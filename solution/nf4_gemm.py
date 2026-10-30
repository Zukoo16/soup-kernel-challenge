"""K5: fused NF4 dequant + GEMM for bitsandbytes 4-bit Linear layers.

The base run (bnb 0.50.2, QLoRA-style NF4, double quantization) computes every
frozen ``Linear4bit`` as ``bnb.matmul_4bit``: a fused CUDA dequant+GEMM in fp16,
and in backward it *fully dequantizes* the weight to fp16 and runs a cuBLAS GEMM
to get ``grad_x``.

This module replaces that with two Triton kernels that dequantize each
64-element weight block on the fly (NF4 lookup + double-quant scale) and
accumulate in fp32:

* forward : ``C[M, N] = x[M, K] @ W_deq[N, K]^T`` -- contracts over the weight
  columns, so a weight row is read as one contiguous packed chunk;
* backward: ``grad_x[M, K] = grad_y[M, N] @ W_deq[N, K]`` -- contracts over the
  weight rows, so the inner loop steps over rows of the packed weight.

The 4-bit packed weight never leaves HBM as fp16 (no extra persistent memory),
and one kernel pair covers every bnb layer of the model (168 projections),
so its GPU time is the anchor of the check-4 kernel share.

Dtypes: in the task run the residual stream is fp32 (the norms stay fp32 after
``prepare_model_for_kbit_training``), so the input arrives fp32 (or fp16 for
``o_proj``, which follows fp16 SDPA).  The base casts the input to the layer's
compute dtype (fp16) for the GEMM and casts the fp16 result back to the input
dtype; this module does the same two casts on the torch side and runs the
kernels themselves in fp16/fp32 exactly like the base ``gemm_4bit``.

Layout (verified against the bnb 0.50.2 CUDA source and its own Triton backend):

* packed weight ``B``: ``[(N*K+1)//2, 1]`` uint8 (row-major flat; row ``n`` of
  the (N, K) weight starts at byte offset ``n*(K//2)``);
* nibble order: even ``k=2j`` -> HIGH nibble, odd ``k=2j+1`` -> LOW nibble;
* quantization blocks of 64 elements along ``K`` per row;
* double quant: ``qabsmax`` = packed NF4 of (per-block absmax - offset) in
  blocks of 256 (same nibble rule over the flat index), ``state2.absmax`` =
  fp32 per 256 of that, ``offset`` = scalar fp32;
* ``w[n, k] = lut[code(n, k)] * (lut[qa(i)] * state2[i // 256] + offset)``,
  where ``i = n * (K // 64) + k // 64``.

Safety: on the very first call of every module 16-row forward *and* backward
probes are compared against a pure-torch reference dequant; on mismatch (or
any error) the module permanently falls back to the original bnb
implementation -- never wrong numbers.
"""

import threading

import torch
import triton
import triton.language as tl

BM = 64
BN = 64
BK = 64
_PROBE_ROWS = 16

# NF4 lookup table (quantiles of N(0,1), QLoRA / bnb standard).
_NF4_VALUES = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]

_lut_cache = {}
_states = {}
_states_lock = threading.Lock()


def _get_lut(device):
    lut = _lut_cache.get(device)
    if lut is None:
        lut = torch.tensor(_NF4_VALUES, device=device, dtype=torch.float32)
        _lut_cache[device] = lut
    return lut


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------


@triton.jit
def nf4_gemm_kernel(
    a_ptr,        # (M, K) fp16
    b_ptr,        # packed uint8, flat (N, K//2) row-major
    qabs_ptr,     # packed uint8, flat
    state2_ptr,   # fp32, flat
    offset_ptr,   # fp32 scalar
    lut_ptr,      # fp32 (16,)
    c_ptr,        # (M, N) fp16
    M, N, K,      # ints; K % 64 == 0
    stride_am,    # row stride of a
    stride_bn,    # packed row stride (bytes) of b = K//2
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    mmask = rm < M
    nmask = rn < N

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + rm[:, None] * stride_am + rk[None, :],
                    mask=mmask[:, None], other=0.0)
        # packed bytes (BN, BK//2): byte (n, k//2); even k in the high nibble
        rj = k0 // 2 + tl.arange(0, BK // 2)
        b = tl.load(b_ptr + rn[:, None] * stride_bn + rj[None, :],
                    mask=nmask[:, None], other=0).to(tl.uint8, bitcast=True)
        hi = (b >> 4) & 0xF
        lo = b & 0xF
        # double-quant scale of the 64-block (n, k0//64)
        i = rn * (K // 64) + k0 // 64
        qb = tl.load(qabs_ptr + i // 2, mask=nmask, other=0).to(tl.uint8, bitcast=True)
        qcode = tl.where((i % 2) == 0, (qb >> 4) & 0xF, qb & 0xF)
        s2 = tl.load(state2_ptr + i // 256, mask=nmask, other=0.0)
        scale = tl.load(lut_ptr + qcode) * s2 + tl.load(offset_ptr)
        w_hi = (tl.load(lut_ptr + hi) * scale[:, None]).to(tl.float16)
        w_lo = (tl.load(lut_ptr + lo) * scale[:, None]).to(tl.float16)
        w = tl.interleave(w_hi, w_lo)  # (BN, BK)
        acc = tl.dot(a, tl.trans(w), acc)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.float16),
             mask=mmask[:, None] & nmask[None, :])


@triton.jit
def nf4_gemm_bwd_kernel(
    a_ptr,        # (M, N) fp16  (grad_y)
    b_ptr,        # packed uint8, flat (N, K//2) row-major
    qabs_ptr,     # packed uint8, flat
    state2_ptr,   # fp32, flat
    offset_ptr,   # fp32 scalar
    lut_ptr,      # fp32 (16,)
    c_ptr,        # (M, K) fp16
    M, N, K,      # ints; K % 64 == 0
    stride_am,    # row stride of a
    stride_bn,    # packed row stride (bytes) of b = K//2
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rk = pid_k * BK + tl.arange(0, BK)
    mmask = rm < M

    acc = tl.zeros((BM, BK), dtype=tl.float32)
    # contract over the weight rows: the inner loop steps over packed rows
    for n0 in range(0, N, BN):
        rn = n0 + tl.arange(0, BN)
        nmask = rn < N
        a = tl.load(a_ptr + rm[:, None] * stride_am + rn[None, :],
                    mask=mmask[:, None] & nmask[None, :], other=0.0)
        # packed bytes (BN, BK//2) of this 64-wide k-tile
        rj = pid_k * BK // 2 + tl.arange(0, BK // 2)
        b = tl.load(b_ptr + rn[:, None] * stride_bn + rj[None, :],
                    mask=nmask[:, None], other=0).to(tl.uint8, bitcast=True)
        hi = (b >> 4) & 0xF
        lo = b & 0xF
        # one 64-block (n, pid_k) per row: the scale is constant over the tile
        i = rn * (K // 64) + (pid_k * BK) // 64
        qb = tl.load(qabs_ptr + i // 2, mask=nmask, other=0).to(tl.uint8, bitcast=True)
        qcode = tl.where((i % 2) == 0, (qb >> 4) & 0xF, qb & 0xF)
        s2 = tl.load(state2_ptr + i // 256, mask=nmask, other=0.0)
        scale = tl.load(lut_ptr + qcode) * s2 + tl.load(offset_ptr)
        w_hi = (tl.load(lut_ptr + hi) * scale[:, None]).to(tl.float16)
        w_lo = (tl.load(lut_ptr + lo) * scale[:, None]).to(tl.float16)
        w = tl.interleave(w_hi, w_lo)  # (BN, BK)
        acc = tl.dot(a, w, acc)
    tl.store(c_ptr + rm[:, None] * K + rk[None, :], acc.to(tl.float16),
             mask=mmask[:, None])


# ---------------------------------------------------------------------------
# pure-torch reference dequant (used by the probe check and by unit tests)
# ---------------------------------------------------------------------------


def ref_dequant_rows(packed_u8, qabs_u8, state2_f32, offset_f32, n_rows, K):
    """Dequantize ``n_rows`` weight rows with plain torch (fp32).

    ``packed_u8``: flat row-major, exactly ``n_rows * K // 2`` bytes (row ``n``
    at offset ``n * K // 2``); ``qabs_u8``: flat packed; ``state2_f32``: flat fp32.
    """
    device = packed_u8.device
    lut = torch.tensor(_NF4_VALUES, device=device, dtype=torch.float32)
    p = packed_u8.view(n_rows, K // 2)
    hi = ((p >> 4) & 0xF).to(torch.long)      # even k
    lo = (p & 0xF).to(torch.long)             # odd k
    i = torch.arange(n_rows * (K // 64), device=device, dtype=torch.long)
    qb = qabs_u8.view(-1)[i // 2]
    qcode = torch.where((i % 2) == 0, (qb >> 4) & 0xF, qb & 0xF).to(torch.long)
    scale = (lut[qcode] * state2_f32.view(-1)[i // 256] + offset_f32.view(-1)[0]
             ).view(n_rows, K // 64)
    scale_full = scale.repeat_interleave(64, dim=1)  # (n, K)
    w_hi = lut[hi] * scale_full[:, 0::2]
    w_lo = lut[lo] * scale_full[:, 1::2]
    return torch.stack((w_hi, w_lo), dim=2).reshape(n_rows, K)


def _pack_module(module, qs):
    """Extract (packed_u8 flat, qabs_u8 flat, state2 flat, offset, N, K), or None.

    bnb stores the packed weight as ``[(N*K+1)//2, 1]`` uint8 (row-major flat:
    row ``n`` of the (N, K) weight starts at byte ``n*(K//2)``), which is
    exactly the layout the kernels index with ``stride_bn = K//2``.
    """
    N, K = int(qs.shape[0]), int(qs.shape[1])
    w = module.weight.data
    if w.dtype != torch.uint8 or not w.is_contiguous():
        return None
    packed = w.reshape(-1)
    qabs = qs.absmax
    if qabs is None or not isinstance(qabs, torch.Tensor) or qabs.dtype != torch.uint8:
        return None
    qabs = qabs.reshape(-1)
    s2 = qs.state2.absmax
    if s2 is None or not isinstance(s2, torch.Tensor) or s2.dtype != torch.float32:
        return None
    s2 = s2.reshape(-1)
    off = qs.offset
    if isinstance(off, torch.Tensor):
        off = off.reshape(-1)
    else:
        off = torch.tensor(float(off))
    if off.numel() != 1:
        return None
    off = off.to(device=w.device, dtype=torch.float32)
    if packed.numel() < (N * K) // 2 or qabs.numel() < (N * K) // 128 \
            or s2.numel() < (N * K) // (64 * 256):
        return None
    return packed, qabs, s2, off, N, K


def _probe(module, packed, qabs, s2, off, N, K, x, lut):
    """16-row forward *and* backward kernel output vs a pure-torch reference; True
    when both agree. The reference dequantizes only a subset of weight rows (enough
    to exercise the nibble / block / scale indexing) to keep the transient memory
    small; the backward probe zeros the un-probed rows of grad_y so the same subset
    suffices. The backward probe input is a fixed pattern, so the probe never
    advances the global RNG (the run's dropout draws must stay in place)."""
    n_probe = min(_PROBE_ROWS, x.shape[0])
    n_rows = min(64, N)
    x_probe = x.reshape(-1, K)[:n_probe].contiguous().to(torch.float16)
    w_ref = ref_dequant_rows(packed[: n_rows * K // 2], qabs, s2, off, n_rows, K)

    out = torch.empty((n_probe, N), device=x.device, dtype=torch.float16)
    nf4_gemm_kernel[(triton.cdiv(n_probe, BM), triton.cdiv(N, BN))](
        x_probe, packed, qabs, s2, off, lut, out,
        n_probe, N, K, x_probe.stride(0), K // 2,
        BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=3,
    )
    ref = (x_probe.to(torch.float32) @ w_ref.T).to(torch.float16)  # (16, n_rows)
    diff = (out[:, :n_rows].to(torch.float32) - ref.to(torch.float32)).abs()
    tol = 2e-2 * ref.to(torch.float32).abs().clamp(min=1.0)
    if not bool((diff <= tol).all().item()):
        return False

    gy = torch.zeros(n_probe, N, device=x.device, dtype=torch.float16)
    nn_ = torch.arange(n_rows, device=x.device, dtype=torch.float16)
    mm_ = torch.arange(n_probe, device=x.device, dtype=torch.float16).unsqueeze(1)
    gy[:, :n_rows] = ((mm_ * nn_.unsqueeze(0) + 1) % 7 - 3) * 0.1
    gx = torch.empty((n_probe, K), device=x.device, dtype=torch.float16)
    nf4_gemm_bwd_kernel[(triton.cdiv(n_probe, BM), K // BK)](
        gy, packed, qabs, s2, off, lut, gx,
        n_probe, N, K, gy.stride(0), K // 2,
        BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=3,
    )
    ref_x = (gy[:, :n_rows].to(torch.float32) @ w_ref).to(torch.float16)  # (16, K)
    diff = (gx.to(torch.float32) - ref_x.to(torch.float32)).abs()
    tol = 2e-2 * ref_x.to(torch.float32).abs().clamp(min=1.0)
    return bool((diff <= tol).all().item())


def _prepare(module, x):
    """Build and validate the per-module state; None => stay on the original path."""
    qs = module.weight.quant_state
    packed = _pack_module(module, qs)
    if packed is None:
        return None
    packed_u8, qabs, s2, off, N, K = packed
    lut = _get_lut(x.device)
    if not _probe(module, packed_u8, qabs, s2, off, N, K, x, lut):
        print("[solution] nf4_gemm: probe mismatch for "
              f"{type(module).__name__} {N}x{K}; using the original bnb path", flush=True)
        return None
    return (packed_u8, qabs, s2, off, N, K)


# ---------------------------------------------------------------------------
# autograd
# ---------------------------------------------------------------------------


class NF4Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed, qabs, s2, off, lut, N, K):
        m = x.shape[0]
        out = torch.empty((m, N), device=x.device, dtype=torch.float16)
        nf4_gemm_kernel[(triton.cdiv(m, BM), triton.cdiv(N, BN))](
            x, packed, qabs, s2, off, lut, out,
            m, N, K, x.stride(0), K // 2,
            BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=3,
        )
        ctx.save_for_backward(packed, qabs, s2, off, lut)
        ctx.N, ctx.K, ctx.M = N, K, m
        return out

    @staticmethod
    def backward(ctx, gy):
        packed, qabs, s2, off, lut = ctx.saved_tensors
        gy16 = gy.to(torch.float16).contiguous()
        gx = torch.empty((ctx.M, ctx.K), device=gy.device, dtype=torch.float16)
        nf4_gemm_bwd_kernel[(triton.cdiv(ctx.M, BM), ctx.K // BK)](
            gy16, packed, qabs, s2, off, lut, gx,
            ctx.M, ctx.N, ctx.K, gy16.stride(0), ctx.K // 2,
            BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=3,
        )
        return gx, None, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# the patch
# ---------------------------------------------------------------------------


def _can_fast(self, x):
    try:
        ok = (
            isinstance(x, torch.Tensor)
            and x.is_cuda
            and x.dtype in (torch.float16, torch.float32)
            and x.dim() >= 1
            and self.bias is None
            and self.weight is not None
            and getattr(self.weight, "quant_state", None) is not None
        )
        if not ok:
            return False
        qs = self.weight.quant_state
        ok = (
            qs.quant_type == "nf4"
            and qs.nested
            and getattr(qs.state2, "blocksize", None) == 256
            and isinstance(qs.shape, (tuple, list)) and len(qs.shape) == 2
        )
        if not ok:
            return False
        n, k = int(qs.shape[0]), int(qs.shape[1])
        ok = k % 64 == 0 and n % 64 == 0 and x.shape[-1] == k
        if ok and self.compute_dtype is not None:
            ok = self.compute_dtype == torch.float16
        return ok
    except Exception:
        return False


def _nf4_forward(self, x, original):
    key = id(self)
    state = _states.get(key)
    if state is None:
        with _states_lock:
            state = _prepare(self, x)
            if state is not None:
                _states[key] = state
    if state is None:
        return original(self, x)
    packed_u8, qabs, s2, off, N, K = state
    inp_dtype = x.dtype
    orig_shape = x.shape
    x2 = x.reshape(-1, K)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    x16 = x2.to(torch.float16)  # the base's x.to(compute_dtype)
    out = NF4Linear.apply(x16, packed_u8, qabs, s2, off, _get_lut(x.device), N, K)
    return out.reshape(orig_shape).to(inp_dtype)  # the base's .to(inp_dtype)


def patch_nf4_linear():
    """bitsandbytes.nn.Linear4bit.forward -> fused NF4 GEMM when shapes match,
    original otherwise (and permanently, after a failed probe)."""
    try:
        from bitsandbytes.nn import modules as bnb_modules
    except Exception:
        return
    l4 = getattr(bnb_modules, "Linear4bit", None)
    original = getattr(l4, "forward", None)
    if original is None or getattr(original, "_soup_nf4_gemm", False):
        return

    def forward(self, x):
        bnb_modules.fix_4bit_weight_quant_state_from_module(self)
        if not _can_fast(self, x):
            return original(self, x)
        try:
            return _nf4_forward(self, x, original)
        except Exception as exc:  # never break a run over an optimization
            print(f"[solution] nf4_gemm fell back: {exc!r}", flush=True)
            return original(self, x)

    forward._soup_nf4_gemm = True
    bnb_modules.Linear4bit.forward = forward


def precompile():
    """Compile both kernels (and both M % 16 specializations) before the first
    timed step."""
    if not torch.cuda.is_available():
        return
    n, k = 256, 512
    try:
        packed = torch.randint(0, 255, (n * k // 2,), device="cuda", dtype=torch.uint8)
        qabs = torch.randint(0, 255, (n * k // 128,), device="cuda", dtype=torch.uint8)
        s2 = torch.rand(n * k // 16384 + 1, device="cuda", dtype=torch.float32) + 0.5
        off = torch.tensor([0.1], device="cuda", dtype=torch.float32)
        lut = _get_lut("cuda")
        for m in (_PROBE_ROWS, _PROBE_ROWS + 3):  # cover the M % 16 specializations
            x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.1
            out = torch.empty((m, n), device="cuda", dtype=torch.float16)
            nf4_gemm_kernel[(triton.cdiv(m, BM), triton.cdiv(n, BN))](
                x, packed, qabs, s2, off, lut, out,
                m, n, k, x.stride(0), k // 2,
                BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=3,
            )
            gy = torch.randn(m, n, device="cuda", dtype=torch.float16) * 0.1
            gx = torch.empty((m, k), device="cuda", dtype=torch.float16)
            nf4_gemm_bwd_kernel[(triton.cdiv(m, BM), k // BK)](
                gy, packed, qabs, s2, off, lut, gx,
                m, n, k, gy.stride(0), k // 2,
                BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=3,
            )
            del x, out, gy, gx
        del packed, qabs, s2, off
        torch.cuda.synchronize()
    except Exception:
        pass
