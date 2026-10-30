"""K4: fused LoRA path for the adapted projections (q_proj / v_proj, r=16).

The base (peft 0.20.0 vanilla LoRA) computes, per adapted linear:

    result = base_layer(x)                       # bnb Linear4bit -> K5 (fp32 out)
    x      = _cast_input_dtype(x, A.weight.dtype)
    x      = dropout(x)                          # F.dropout, MUST keep the RNG draws
    result = result + B(A(x)) * scaling          # 3 GEMM launches (A, B, add-mul)

Under the trainer's fp16 autocast the two ``F.linear`` calls run in fp16 while
the ``result`` tensor (the base layer's output) stays fp32: the base adds
``result(fp32) + fp16(B(A(x)) * scaling)`` in fp32.  This module replaces the
three small launches with two Triton launches and keeps every cast of the base:

* forward : ``t = x @ A^T`` (M, R) and ``out = result + s * t @ B^T`` in one
  kernel (R = 16 fits in registers; the LoRA branch is rounded to fp16 before
  the add, like the base; ``t`` is kept for the backward);
* backward: ``dt = (s*dy) @ B``; then ``dx = dt @ A``, ``dA = dt^T x``,
  ``dB = (s*dy)^T t`` -- one kernel, four MODE specializations; the gradient of
  ``result`` is ``dy`` itself, so the base layer's backward (K5) still runs.

The dropout call is peft's own ``F.dropout`` invoked with the same value the
base drops, so the RNG stream (and therefore every later number) is unchanged.
Only the plain, unmerged, single-adapter vanilla path is fast-pathed; anything
else (merged state, extra adapters, kwargs, r != 16, non-CUDA float input)
falls back to the original peft implementation.
"""

import threading

import torch
import triton
import triton.language as tl

BM = 64
BN = 64
BK = 64
_R = 16  # the task's LoRA rank

_lora_ok = {}
_lora_failed = {}
_lora_lock = threading.Lock()
_probe_gens = {}


@triton.jit
def lora_fwd_kernel(
    result_ptr,   # (M, OUT) native dtype (fp32 in the task run)
    x_ptr,        # (M, IN) fp16   (the dropped input, autocast dtype)
    a_ptr,        # (R, IN) fp16
    b_ptr,        # (OUT, R) fp16
    t_ptr,        # (M, R) fp16 out
    out_ptr,      # (M, OUT) native dtype out
    M, IN, OUT,
    stride_rm, stride_xm,
    scaling,
    R: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    ro = pid_n * BN + tl.arange(0, BN)
    rr = tl.arange(0, R)
    mmask = rm < M

    acc = tl.zeros((BM, R), dtype=tl.float32)
    for k0 in range(0, IN, BK):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + rm[:, None] * stride_xm + rk[None, :],
                    mask=mmask[:, None], other=0.0)
        a = tl.load(a_ptr + rk[:, None] + rr[None, :] * IN).to(tl.float16)  # (BK, R)
        acc = tl.dot(x, a, acc)
    t16 = acc.to(tl.float16)
    tl.store(t_ptr + rm[:, None] * R + rr[None, :], t16, mask=mmask[:, None])

    b = tl.load(b_ptr + ro[None, :] * R + rr[:, None]).to(tl.float16)  # (R, BN)
    acc2 = tl.dot(t16, b)
    res = tl.load(result_ptr + rm[:, None] * stride_rm + ro[None, :],
                  mask=mmask[:, None], other=0.0).to(tl.float32)
    # the base rounds the LoRA branch to the compute dtype (fp16) before the
    # fp32 add: result + fp16(B_out * scaling)
    branch = (scaling * acc2).to(tl.float16).to(tl.float32)
    tl.store(out_ptr + rm[:, None] * OUT + ro[None, :],
             (res + branch).to(out_ptr.dtype.element_ty), mask=mmask[:, None])


@triton.jit
def lora_bwd_kernel(
    p1_ptr, p2_ptr, out_ptr,
    M, IN, OUT,
    stride_p1m,
    scaling,
    R: tl.constexpr,
    MODE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    if MODE == 0:
        # dt = (scaling * dy) @ B   (M, R);  p1 = dy (M, OUT), p2 = B (OUT, R)
        # the base multiplies the fp16 gradient by the scalar first (fp16
        # rounding), then runs the GEMM -- mirror that order.
        pid = tl.program_id(0)
        rm = pid * BM + tl.arange(0, BM)
        rr = tl.arange(0, R)
        mmask = rm < M
        acc = tl.zeros((BM, R), dtype=tl.float32)
        for k0 in range(0, OUT, BK):
            rk = k0 + tl.arange(0, BK)
            dy = tl.load(p1_ptr + rm[:, None] * stride_p1m + rk[None, :],
                         mask=mmask[:, None], other=0.0)
            dys = (dy * scaling).to(tl.float16)
            b = tl.load(p2_ptr + rk[:, None] * R + rr[None, :]).to(tl.float16)
            acc = tl.dot(dys, b, acc)
        tl.store(out_ptr + rm[:, None] * R + rr[None, :],
                 acc.to(tl.float16), mask=mmask[:, None])
    elif MODE == 1:
        # dx = dt @ A   (M, IN);  p1 = dt (M, R), p2 = A (R, IN)
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        ri = pid_n * BN + tl.arange(0, BN)
        rr = tl.arange(0, R)
        mmask = rm < M
        dt = tl.load(p1_ptr + rm[:, None] * R + rr[None, :], mask=mmask[:, None], other=0.0)
        a = tl.load(p2_ptr + rr[:, None] * IN + ri[None, :]).to(tl.float16)  # (R, BN)
        acc = tl.dot(dt, a)
        tl.store(out_ptr + rm[:, None] * IN + ri[None, :], acc.to(tl.float16),
                 mask=mmask[:, None])
    elif MODE == 2:
        # dA = dt^T @ x   (R, IN);  p1 = x (M, IN), p2 = dt (M, R)
        pid = tl.program_id(0)
        ri = pid * BM + tl.arange(0, BM)
        rr = tl.arange(0, R)
        imask = ri < IN
        acc = tl.zeros((R, BM), dtype=tl.float32)
        for m0 in range(0, M, BK):
            rm = m0 + tl.arange(0, BK)
            x = tl.load(p1_ptr + rm[:, None] * IN + ri[None, :],
                        mask=rm[:, None] < M, other=0.0)
            dt = tl.load(p2_ptr + rm[:, None] * R + rr[None, :]).to(tl.float16)
            acc = tl.dot(tl.trans(dt), x, acc)
        tl.store(out_ptr + rr[:, None] * IN + ri[None, :], acc.to(tl.float16),
                 mask=imask[None, :])
    else:
        # dB = (scaling * dy)^T @ t   (OUT, R);  p1 = dy (M, OUT), p2 = t (M, R)
        pid = tl.program_id(0)
        ro = pid * BM + tl.arange(0, BM)
        rr = tl.arange(0, R)
        omask = ro < OUT
        acc = tl.zeros((BM, R), dtype=tl.float32)
        for m0 in range(0, M, BK):
            rm = m0 + tl.arange(0, BK)
            dy = tl.load(p1_ptr + rm[:, None] * OUT + ro[None, :],
                         mask=rm[:, None] < M, other=0.0)
            dys = (dy * scaling).to(tl.float16)
            t = tl.load(p2_ptr + rm[:, None] * R + rr[None, :]).to(tl.float16)
            acc = tl.dot(tl.trans(dys), t, acc)
        tl.store(out_ptr + ro[:, None] * R + rr[None, :],
                 acc.to(tl.float16), mask=omask[:, None])


def _probe_gen(device):
    """A dedicated RNG stream for the probes: a seeded side generator never
    advances the global RNG, so the run's dropout draws stay in place."""
    gen = _probe_gens.get(device)
    if gen is None:
        gen = torch.Generator(device=device).manual_seed(0x5EED)
        _probe_gens[device] = gen
    return gen


def _lora_probe(self):
    """Compare the fused path against plain torch on deterministic 16-row inputs
    (forward values and the A/B gradients); True when they agree."""
    try:
        a_w = self.lora_A["default"].weight
        b_w = self.lora_B["default"].weight
        scaling = float(self.scaling["default"])
        m = 16
        gen = _probe_gen(a_w.device)
        a16 = a_w.to(torch.float16)
        b16 = b_w.to(torch.float16)
        x_p = ((torch.rand(m, a_w.shape[1], device=a_w.device, generator=gen) * 2 - 1)
               * 0.5).to(torch.float16)
        res_p = ((torch.rand(m, b_w.shape[0], device=a_w.device, generator=gen) * 2 - 1)
                 * 0.5).to(torch.float32)

        out = FusedLoRA.apply(res_p, x_p, a16, b16, scaling)
        ref = res_p + (x_p @ a16.t()) @ b16.t() * scaling
        if not bool(torch.allclose(out, ref, rtol=2e-2, atol=2e-2)):
            return False

        gy = torch.randn(m, b_w.shape[0], device=a_w.device, generator=gen).to(torch.float16)
        A_f, B_f = a16.clone().requires_grad_(True), b16.clone().requires_grad_(True)
        A_r, B_r = a16.clone().requires_grad_(True), b16.clone().requires_grad_(True)
        FusedLoRA.apply(res_p, x_p, A_f, B_f, scaling).backward(gy)
        (res_p + (x_p @ A_r.t()) @ B_r.t() * scaling).backward(gy)
        return (bool(torch.allclose(A_f.grad, A_r.grad, rtol=2e-2, atol=2e-2))
                and bool(torch.allclose(B_f.grad, B_r.grad, rtol=2e-2, atol=2e-2)))
    except Exception:
        return False


class FusedLoRA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, result, x_drop, A16, B16, scaling):
        m, i_in = x_drop.shape
        out_dim = B16.shape[0]
        r = A16.shape[0]
        t = torch.empty(m, r, device=x_drop.device, dtype=torch.float16)
        out = torch.empty(m, out_dim, device=result.device, dtype=result.dtype)
        lora_fwd_kernel[(triton.cdiv(m, BM), triton.cdiv(out_dim, BN))](
            result, x_drop, A16, B16, t, out, m, i_in, out_dim,
            result.stride(0), x_drop.stride(0), scaling,
            R=r, BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=3,
        )
        ctx.save_for_backward(x_drop, A16, B16, t)
        ctx.scaling = scaling
        ctx.shape = (m, i_in, out_dim, r)
        return out

    @staticmethod
    def backward(ctx, dy):
        x_drop, A16, B16, t = ctx.saved_tensors
        m, i_in, out_dim, r = ctx.shape
        s = ctx.scaling
        dev = dy.device
        dy16 = dy.to(torch.float16).contiguous()
        dt = torch.empty(m, r, device=dev, dtype=torch.float16)
        lora_bwd_kernel[(triton.cdiv(m, BM),)](
            dy16, B16, dt, m, i_in, out_dim, dy16.stride(0), s,
            R=r, MODE=0, BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=3,
        )
        dx = torch.empty(m, i_in, device=dev, dtype=torch.float16)
        lora_bwd_kernel[(triton.cdiv(m, BM), triton.cdiv(i_in, BN))](
            dt, A16, dx, m, i_in, out_dim, dt.stride(0), s,
            R=r, MODE=1, BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=3,
        )
        dA = torch.empty(i_in, r, device=dev, dtype=torch.float16)
        lora_bwd_kernel[(triton.cdiv(i_in, BM),)](
            x_drop, dt, dA, m, i_in, out_dim, x_drop.stride(0), s,
            R=r, MODE=2, BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=3,
        )
        dB = torch.empty(out_dim, r, device=dev, dtype=torch.float16)
        lora_bwd_kernel[(triton.cdiv(out_dim, BM),)](
            dy16, t, dB, m, i_in, out_dim, dy16.stride(0), s,
            R=r, MODE=3, BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=3,
        )
        # the base-layer output gets dy, exactly like the peft add node: this
        # keeps the gradient flowing into the (frozen, K5) base GEMM.
        return dy, dx, dA, dB, None


def _can_fast(self, x):
    try:
        ok = (
            isinstance(x, torch.Tensor)
            and x.is_cuda
            and x.dtype in (torch.float16, torch.float32)
            and x.dim() >= 1
            and not self.disable_adapters
            and not self.merged
            and list(self.active_adapters) == ["default"]
            and "default" in self.lora_A
            and "default" in self.lora_B
            and self.lora_variant.get("default") is None
            and self.lora_dropout is not None
            and "default" in self.lora_dropout
        )
        if not ok:
            return False
        a = self.lora_A["default"]
        b = self.lora_B["default"]
        a_w, b_w = a.weight, b.weight
        ok = (
            a.bias is None and b.bias is None
            and a_w.shape[0] == _R and b_w.shape[1] == _R
            and a_w.shape[1] == x.shape[-1]
            and a_w.is_cuda and b_w.is_cuda
            and isinstance(self.scaling["default"], (int, float))
        )
        return ok
    except Exception:
        return False


def _fused_lora(self, x):
    key = id(self)
    if key in _lora_failed:
        return None
    if key not in _lora_ok:
        with _lora_lock:
            if key not in _lora_ok:
                if _lora_probe(self):
                    _lora_ok[key] = True
                else:
                    print("[solution] fused lora: probe mismatch for "
                          f"{type(self).__name__}; using the original peft path", flush=True)
                    _lora_failed[key] = True
                    return None
    a = self.lora_A["default"]
    b = self.lora_B["default"]
    dropout = self.lora_dropout["default"]
    scaling = float(self.scaling["default"])
    x_cast = self._cast_input_dtype(x, a.weight.dtype)
    x_drop = dropout(x_cast)  # the SAME RNG draws as the base
    result = self.base_layer(x)
    out = FusedLoRA.apply(
        result.reshape(-1, result.shape[-1]),
        x_drop.to(torch.float16).reshape(-1, x_drop.shape[-1]),
        a.weight.to(torch.float16),
        b.weight.to(torch.float16),
        scaling,
    )
    return out.reshape(result.shape).to(result.dtype)


def patch_lora_linear():
    """peft LoRA Linear.forward -> fused kernel on the vanilla fast path,
    original implementation otherwise."""
    try:
        from peft.tuners.lora import layer as lora_layer
    except Exception:
        return
    lora_linear = getattr(lora_layer, "Linear", None)
    original = getattr(lora_linear, "forward", None)
    if original is None or getattr(original, "_soup_fused_lora", False):
        return

    def forward(self, x, *args, **kwargs):
        if args or kwargs:
            return original(self, x, *args, **kwargs)
        if not _can_fast(self, x):
            return original(self, x)
        try:
            out = _fused_lora(self, x)
            return original(self, x) if out is None else out
        except Exception as exc:  # never break a run over an optimization
            print(f"[solution] fused lora fell back: {exc!r}", flush=True)
            return original(self, x)

    forward._soup_fused_lora = True
    lora_layer.Linear.forward = forward


def precompile():
    """Compile the forward and all four backward MODEs before the first timed
    step, in the task's actual dtypes (fp32 result / input, fp16 A-B-t)."""
    if not torch.cuda.is_available():
        return
    m, i_in, out_dim = 16, 512, 1024
    try:
        x = torch.randn(m, i_in, device="cuda", dtype=torch.float16) * 0.1
        result = torch.randn(m, out_dim, device="cuda", dtype=torch.float32) * 0.1
        A = torch.randn(_R, i_in, device="cuda", dtype=torch.float16) * 0.1
        B = torch.randn(out_dim, _R, device="cuda", dtype=torch.float16) * 0.1
        out = FusedLoRA.apply(result, x, A, B, 2.0)
        torch.autograd.backward(out, torch.randn_like(out))
        del x, result, A, B, out
        torch.cuda.synchronize()
    except Exception:
        pass
