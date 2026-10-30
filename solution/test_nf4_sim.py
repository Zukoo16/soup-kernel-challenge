"""CPU simulation of the NF4 GEMM kernels' exact index arithmetic.

The Triton kernels themselves need a GPU, but the part that can silently go wrong
(nibble order, the double-quant scale index, the backward orientation) is plain
indexing.  These tests re-implement the kernels' loads and index math step by step
in pure Python and check them against the pure-torch reference dequant (itself
validated against a per-element brute force in test_nf4.py).  A transcription bug
in the kernel's indexing breaks the simulation-vs-reference comparison here, on a
CPU, before anything runs on the GPU.
"""

import torch

from solution import nf4_gemm
from solution.test_nf4 import quantize_torch

LUT = torch.tensor(nf4_gemm._NF4_VALUES, dtype=torch.float32)


def _simulate_fwd(x, packed, qabs, s2, off, N, K):
    """Element-wise Python mirror of nf4_gemm_kernel (fp32 throughout)."""
    M = x.shape[0]
    out = torch.zeros(M, N)
    for m in range(M):
        for n in range(N):
            acc = 0.0
            for k0 in range(0, K, 64):  # BK == 64
                for j in range(32):  # BK // 2 bytes
                    byte = int(packed[n * (K // 2) + k0 // 2 + j])
                    hi = (byte >> 4) & 0xF
                    lo = byte & 0xF
                    i = n * (K // 64) + k0 // 64
                    qb = int(qabs[i // 2])
                    qcode = (qb >> 4) & 0xF if i % 2 == 0 else qb & 0xF
                    scale = LUT[qcode].item() * s2[i // 256].item() + off.item()
                    w_hi = LUT[hi].item() * scale
                    w_lo = LUT[lo].item() * scale
                    # interleave: even k = high nibble, odd k = low nibble
                    acc += float(x[m, k0 + 2 * j]) * w_hi
                    acc += float(x[m, k0 + 2 * j + 1]) * w_lo
            out[m, n] = acc
    return out


def _simulate_bwd(gy, packed, qabs, s2, off, N, K):
    """Element-wise Python mirror of nf4_gemm_bwd_kernel (fp32 throughout).

    grad_x[m, k] = sum_n gy[m, n] * W[n, k] -- the contraction over the weight
    ROWS, which is the orientation that used to be wrong.
    """
    M = gy.shape[0]
    out = torch.zeros(M, K)
    for m in range(M):
        for k0 in range(0, K, 64):  # BK == 64, one quant block per k-tile
            for j in range(32):
                for n in range(N):
                    byte = int(packed[n * (K // 2) + k0 // 2 + j])
                    hi = (byte >> 4) & 0xF
                    lo = byte & 0xF
                    i = n * (K // 64) + k0 // 64  # == n * (K // 64) + (k0 * 64) // 64
                    qb = int(qabs[i // 2])
                    qcode = (qb >> 4) & 0xF if i % 2 == 0 else qb & 0xF
                    scale = LUT[qcode].item() * s2[i // 256].item() + off.item()
                    w_hi = LUT[hi].item() * scale
                    w_lo = LUT[lo].item() * scale
                    out[m, k0 + 2 * j] += float(gy[m, n]) * w_hi
                    out[m, k0 + 2 * j + 1] += float(gy[m, n]) * w_lo
    return out


def test_simulated_fwd_matches_reference():
    torch.manual_seed(0)
    n, k = 64, 128
    w = torch.randn(n, k) * 0.05
    packed, qpacked, s2, offset, _ = quantize_torch(w)
    x = torch.randn(8, k) * 0.1
    got = _simulate_fwd(x, packed.view(-1), qpacked, s2, offset, n, k)
    w_ref = nf4_gemm.ref_dequant_rows(packed.view(-1), qpacked, s2, offset, n, k)
    ref = x @ w_ref.T
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)


def test_simulated_bwd_matches_reference():
    torch.manual_seed(1)
    n, k = 64, 128
    w = torch.randn(n, k) * 0.05
    packed, qpacked, s2, offset, _ = quantize_torch(w)
    gy = torch.randn(8, n) * 0.1
    got = _simulate_bwd(gy, packed.view(-1), qpacked, s2, offset, n, k)
    w_ref = nf4_gemm.ref_dequant_rows(packed.view(-1), qpacked, s2, offset, n, k)
    ref = gy @ w_ref
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)


def test_simulated_bwd_nonsquare_orientation():
    """N != K on purpose: a forward-oriented backward would index out of bounds
    (or read the wrong rows) and cannot match the reference here."""
    torch.manual_seed(2)
    n, k = 32, 256
    w = torch.randn(n, k) * 0.05
    packed, qpacked, s2, offset, _ = quantize_torch(w)
    gy = torch.randn(4, n) * 0.1
    got = _simulate_bwd(gy, packed.view(-1), qpacked, s2, offset, n, k)
    w_ref = nf4_gemm.ref_dequant_rows(packed.view(-1), qpacked, s2, offset, n, k)
    ref = gy @ w_ref
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)
