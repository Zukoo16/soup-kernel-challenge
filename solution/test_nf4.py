"""CPU tests for the K5 NF4 dequant math (the part that must be bit-exact with bnb).

The Triton GEMM kernel itself can only run on a GPU; these tests validate the
index/nibble/double-quant math that the kernel implements, by quantizing random
data with a pure-torch replica of the bnb 0.50.2 quantization kernels and
checking that the reference dequant recovers it inside the NF4 error bound.
"""

import torch
import torch.nn.functional as F

from solution import nf4_gemm

LUT = torch.tensor(nf4_gemm._NF4_VALUES, dtype=torch.float32)
GAP_MAX = (LUT[1:] - LUT[:-1]).max().item()


def _nearest_lut(x):
    """Index of the nearest NF4 level (what bnb's dQuantizeNF4 binary search does)."""
    return (x.unsqueeze(-1) - LUT).abs().argmin(dim=-1)


def quantize_torch(w, blocksize=64):
    """Pure-torch replica of bnb 0.50.2 quantize_4bit + compress_statistics=True.

    ``w``: (N, K) fp32. Returns (packed (N, K//2) uint8, qabs packed uint8 flat,
    state2 absmax fp32 flat, offset scalar tensor, per-64-block absmax fp32).
    """
    n, k = w.shape
    assert k % blocksize == 0
    nb = n * (k // blocksize)
    assert nb % 2 == 0
    wb = w.reshape(-1, blocksize)             # (nb, 64)
    absmax = wb.abs().amax(dim=1)             # (nb,)
    code = _nearest_lut((wb / absmax[:, None]).clamp(-1, 1))  # (nb, 64)
    even, odd = code[:, 0::2], code[:, 1::2]
    packed_row = (even << 4) | odd            # even element in the HIGH nibble
    packed = packed_row.reshape(n, k // 2).to(torch.uint8)

    offset = absmax.mean().reshape(-1)        # scalar fp32 tensor
    shifted = (absmax - offset).reshape(-1)   # (nb,) flat, row-major
    b2 = 256
    pad = (b2 - nb % b2) % b2
    s2b = F.pad(shifted, (0, pad)).reshape(-1, b2)
    s2_absmax = s2b.abs().amax(dim=1)         # (ceil(nb/256),)
    code2 = _nearest_lut((s2b / s2_absmax[:, None].clamp(min=1e-30)).clamp(-1, 1)).reshape(-1)[:nb]
    qpacked = ((code2[0::2] << 4) | code2[1::2]).to(torch.uint8)
    return packed, qpacked, s2_absmax, offset, absmax


def test_ref_dequant_roundtrip_bound():
    """The dequantized values stay inside the NF4 weight+scale error bound."""
    torch.manual_seed(0)
    n, k = 16, 256
    w = torch.randn(n, k) * 0.05
    packed, qpacked, s2, offset, absmax = quantize_torch(w)
    w_deq = nf4_gemm.ref_dequant_rows(packed.view(-1), qpacked, s2, offset, n, k)

    bound = 0.5 * GAP_MAX * (absmax + (absmax - offset).abs()).repeat_interleave(64)
    err = (w_deq - w).abs().reshape(-1)
    assert (err <= bound + 1e-6).all(), (
        f"{(err > bound + 1e-6).sum().item()} elements outside the NF4 error bound; "
        f"max err {err.max().item():.4f}"
    )


def test_ref_dequant_brute_force():
    """Vectorized reference equals a per-element python loop (small shape)."""
    torch.manual_seed(1)
    n, k = 3, 128
    w = torch.randn(n, k) * 0.2
    packed, qpacked, s2, offset, _ = quantize_torch(w)

    flat = qpacked.view(-1)
    out = torch.empty(n, k)
    for i in range(n):
        for j in range(k):
            byte = packed[i, j // 2].item()
            code = (byte >> 4) if j % 2 == 0 else (byte & 0xF)
            bi = i * (k // 64) + j // 64
            qb = flat[bi // 2].item()
            qcode = (qb >> 4) if bi % 2 == 0 else (qb & 0xF)
            scale = LUT[qcode] * s2[bi // 256] + offset[0]
            out[i, j] = LUT[code] * scale

    w_deq = nf4_gemm.ref_dequant_rows(packed.view(-1), qpacked, s2, offset, n, k)
    assert torch.equal(w_deq, out), f"max diff {(w_deq - out).abs().max().item()}"


def test_nibble_parity_crafted():
    """Even positions come from the high nibble, odd from the low nibble."""
    torch.manual_seed(2)
    n, k = 2, 64
    # constant sign pattern so the even/odd codes differ from each other
    w = torch.zeros(n, k)
    for i in range(n):
        for j in range(k):
            w[i, j] = (0.5 if j % 2 == 0 else -0.25) * (1.0 if i % 2 == 0 else 0.5)
    packed, qpacked, s2, offset, absmax = quantize_torch(w)

    w_deq = nf4_gemm.ref_dequant_rows(packed.view(-1), qpacked, s2, offset, n, k)
    bound = 0.5 * GAP_MAX * (absmax + (absmax - offset).abs()).repeat_interleave(64)
    err = (w_deq - w).abs().reshape(-1)
    assert (err <= bound + 1e-6).all(), f"nibble parity wrong: max err {err.max().item():.4f}"
