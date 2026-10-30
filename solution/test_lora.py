"""CPU tests for the K4 fused-LoRA math: the forward formula and the four
backward gradients the Triton kernel computes (validated against autograd)."""

import torch

import pytest


def _close(a, b, msg):
    assert torch.allclose(a, b, atol=1e-5), f"{msg}: max diff {(a - b).abs().max().item()}"


@pytest.mark.parametrize("m, i_in, out_dim, r", [(8, 32, 40, 4), (16, 64, 32, 16), (3, 128, 16, 16)])
def test_lora_grad_formulas(m, i_in, out_dim, r):
    torch.manual_seed(3)
    x = torch.randn(m, i_in, requires_grad=True)
    A = (torch.randn(r, i_in) * 0.1).requires_grad_(True)
    B = (torch.randn(out_dim, r) * 0.1).requires_grad_(True)
    result = torch.randn(m, out_dim, requires_grad=True)
    s = 2.0

    t = x @ A.T
    out = result + s * (t @ B.T)
    dy = torch.randn(m, out_dim)
    out.backward(dy)

    dt = s * (dy @ B)
    _close(x.grad, dt @ A, "dx = dt @ A")
    _close(A.grad, dt.T @ x, "dA = dt^T x")
    _close(B.grad, (s * dy).T @ t, "dB = (s dy)^T t")
    _close(result.grad, dy, "dresult = dy")
    _close(out, result + s * (t @ B.T), "forward formula")
