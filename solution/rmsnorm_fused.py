"""Fused RMSNorm for Qwen2 (hidden 1536), forward and backward, one launch each.

The base (transformers 5.16.1 Qwen2RMSNorm.forward) is:

    x32  = x.to(fp32)
    var  = x32.pow(2).mean(-1, keepdim=True)
    y    = self.weight * (x32 * rsqrt(var + eps)).to(x.dtype)

In the task run the residual stream is fp32 (``prepare_model_for_kbit_training``
upcasts the norm weights to fp32 and ``weight(fp32) * fp32`` stays fp32), so the
norm reads fp32, works in fp32 and writes fp32 -- five small kernel launches per
norm (56 norms per step) plus fp32 temporaries.  This replaces it with one
element-wise-ish kernel per direction:

* forward:  read x once, fp32 accumulation, write y (fp32, like the base);
* backward: one pass, dx = inv * (dy*w - mean(dy*w*x) * x) with fp32 accumulation.

For an fp16 input the base additionally rounds the normalized value back to fp16
before the weight multiply (the ``.to(x.dtype)`` step); the ``X16`` flag
reproduces that.  Only Qwen2.5-1.5B's hidden size (1536 = 1024 + 512) is
specialized; anything else falls back to the original implementation.
"""

import threading

import torch
import triton
import triton.language as tl

_HIDDEN = 1536  # Qwen2.5-1.5B hidden_size
_B1, _B2 = 1024, 512  # 1536 = 1024 + 512, both powers of two -> no masking

_norm_ok = {}
_norm_failed = {}
_norm_lock = threading.Lock()
_probe_gens = {}


def _probe_gen(device):
    """A dedicated RNG stream for the probes: a seeded side generator never
    advances the global RNG, so the run's dropout draws stay in place."""
    gen = _probe_gens.get(device)
    if gen is None:
        gen = torch.Generator(device=device).manual_seed(0x5EED)
        _probe_gens[device] = gen
    return gen


@triton.jit
def rmsnorm_fwd_kernel(x_ptr, w_ptr, y_ptr, inv_ptr, eps,
                       X16: tl.constexpr,
                       H: tl.constexpr, B1: tl.constexpr, B2: tl.constexpr):
    row = tl.program_id(0)
    i1 = tl.arange(0, B1)
    i2 = B1 + tl.arange(0, B2)
    x1 = tl.load(x_ptr + row * H + i1).to(tl.float32)
    x2 = tl.load(x_ptr + row * H + i2).to(tl.float32)
    var = (tl.sum(x1 * x1, axis=0) + tl.sum(x2 * x2, axis=0)) / H
    inv = tl.rsqrt(var + eps)
    tl.store(inv_ptr + row, inv)
    w1 = tl.load(w_ptr + i1)
    w2 = tl.load(w_ptr + i2)
    xh1 = x1 * inv
    xh2 = x2 * inv
    if X16:
        # the base casts the normalized value back to the input dtype before
        # the weight multiply (fp16 input only)
        xh1 = xh1.to(tl.float16).to(tl.float32)
        xh2 = xh2.to(tl.float16).to(tl.float32)
    tl.store(y_ptr + row * H + i1, w1 * xh1)
    tl.store(y_ptr + row * H + i2, w2 * xh2)


@triton.jit
def rmsnorm_bwd_kernel(dy_ptr, x_ptr, w_ptr, inv_ptr, dx_ptr,
                       H: tl.constexpr, B1: tl.constexpr, B2: tl.constexpr):
    row = tl.program_id(0)
    i1 = tl.arange(0, B1)
    i2 = B1 + tl.arange(0, B2)
    x1 = tl.load(x_ptr + row * H + i1).to(tl.float32)
    x2 = tl.load(x_ptr + row * H + i2).to(tl.float32)
    dy1 = tl.load(dy_ptr + row * H + i1).to(tl.float32)
    dy2 = tl.load(dy_ptr + row * H + i2).to(tl.float32)
    w1 = tl.load(w_ptr + i1)
    w2 = tl.load(w_ptr + i2)
    inv = tl.load(inv_ptr + row)
    xh1 = x1 * inv  # normalized input, as in the forward
    xh2 = x2 * inv
    dw1 = dy1 * w1
    dw2 = dy2 * w2
    c = (tl.sum(dw1 * xh1, axis=0) + tl.sum(dw2 * xh2, axis=0)) / H
    tl.store(dx_ptr + row * H + i1, ((dw1 - c * xh1) * inv).to(dx_ptr.dtype.element_ty))
    tl.store(dx_ptr + row * H + i2, ((dw2 - c * xh2) * inv).to(dx_ptr.dtype.element_ty))


class FusedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps):
        shape = x.shape
        n = x.numel() // shape[-1]
        y = torch.empty(shape, device=x.device, dtype=torch.float32)
        inv = torch.empty(n, device=x.device, dtype=torch.float32)
        rmsnorm_fwd_kernel[(n,)](x, w, y, inv, eps, X16=(x.dtype == torch.float16),
                                 H=_HIDDEN, B1=_B1, B2=_B2)
        ctx.save_for_backward(x, w, inv)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, inv = ctx.saved_tensors
        shape = x.shape
        n = x.numel() // shape[-1]
        dx = torch.empty(shape, device=x.device, dtype=x.dtype)
        rmsnorm_bwd_kernel[(n,)](dy.contiguous(), x, w, inv, dx,
                                 H=_HIDDEN, B1=_B1, B2=_B2)
        return dx, None, None


def fused_rmsnorm(x, weight, eps):
    return FusedRMSNorm.apply(x, weight, eps)


def _rmsnorm_probe(self, x, original):
    """Compare the fused path against the original implementation on
    deterministic 16-row inputs (values and the input gradient); True on match."""
    try:
        m = 16
        gen = _probe_gen(x.device)
        x_p = ((torch.rand(m, _HIDDEN, device=x.device, generator=gen) * 2 - 1)
               * 0.5).to(x.dtype)
        eps = float(self.variance_epsilon)
        ref = original(self, x_p)
        out = fused_rmsnorm(x_p, self.weight, eps)
        if not bool(torch.allclose(out, ref, rtol=1e-4, atol=1e-4)):
            return False
        x_ref = x_p.clone().requires_grad_(True)
        x_my = x_p.clone().requires_grad_(True)
        dy = torch.randn(m, _HIDDEN, device=x.device, generator=gen)
        original(self, x_ref).backward(dy)
        fused_rmsnorm(x_my, self.weight, eps).backward(dy)
        return bool(torch.allclose(x_my.grad, x_ref.grad, rtol=1e-3, atol=1e-3))
    except Exception:
        return False


def patch_qwen2_rmsnorm():
    """Qwen2RMSNorm.forward -> fused kernel when shapes match, original otherwise."""
    try:
        from transformers.models.qwen2 import modeling_qwen2
    except Exception:
        return

    norm_cls = getattr(modeling_qwen2, "Qwen2RMSNorm", None)
    original = getattr(norm_cls, "forward", None)
    if original is None or getattr(original, "_soup_fused_rmsnorm", False):
        return

    def forward(self, hidden_states):
        try:
            ok = (
                isinstance(hidden_states, torch.Tensor)
                and hidden_states.is_cuda
                and hidden_states.dtype in (torch.float16, torch.float32)
                and hidden_states.dim() >= 1
                and hidden_states.shape[-1] == _HIDDEN
                and getattr(self, "weight", None) is not None
                and self.weight.dtype == torch.float32
            )
        except Exception:
            ok = False
        if not ok:
            return original(self, hidden_states)
        key = id(self)
        if key in _norm_failed:
            return original(self, hidden_states)
        if key not in _norm_ok:
            with _norm_lock:
                if key not in _norm_ok:
                    if _rmsnorm_probe(self, hidden_states, original):
                        _norm_ok[key] = True
                    else:
                        print("[solution] fused rmsnorm: probe mismatch for "
                              f"{type(self).__name__}; using the original path", flush=True)
                        _norm_failed[key] = True
                        return original(self, hidden_states)
        return fused_rmsnorm(hidden_states, self.weight, float(self.variance_epsilon))

    forward._soup_fused_rmsnorm = True
    modeling_qwen2.Qwen2RMSNorm.forward = forward


def precompile():
    """Compile both input-dtype variants before the first timed step (the residual
    stream of the task model is fp32, the fp16 variant keeps other setups warm)."""
    if not torch.cuda.is_available():
        return
    n, h = 2, _HIDDEN
    try:
        w = torch.randn(h, device="cuda", dtype=torch.float32) * 0.1 + 1.0
        for dtype in (torch.float32, torch.float16):
            x = (torch.randn(n, h, device="cuda", dtype=dtype) * 0.1)
            y = FusedRMSNorm.apply(x, w, 1e-6)
            torch.autograd.backward(y, torch.randn(n, h, device="cuda", dtype=torch.float32))
            del x, y
        torch.cuda.synchronize()
    except Exception:
        pass
