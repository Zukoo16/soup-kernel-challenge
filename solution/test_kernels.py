"""Correctness of your kernels against plain PyTorch, forward and backward.

Run on a GPU: `python -m pytest solution -q`. Add a test like this for every kernel you write.
"""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOLERANCE = {torch.float16: 2e-3, torch.float32: 1e-4}


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 8960), (3, 77, 1001)])
def test_swiglu_matches_pytorch(dtype, shape):
    from solution.swiglu import swiglu

    torch.manual_seed(0)
    gate = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    up = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    ref = torch.nn.functional.silu(gate.float()) * up.float()
    ref.backward(grad.float())
    ref_dgate, ref_dup = gate.grad.float(), up.grad.float()
    gate.grad = up.grad = None

    out = swiglu(gate, up)
    out.backward(grad)
    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref, rtol=tol, atol=tol)
    torch.testing.assert_close(gate.grad.float(), ref_dgate, rtol=tol, atol=tol)
    torch.testing.assert_close(up.grad.float(), ref_dup, rtol=tol, atol=tol)


# ---------------------------------------------------------------------------
# Fused cross-entropy (the lm_head logits pipeline)
# ---------------------------------------------------------------------------


def _make_targets(labels):
    """Same mapping as solution.ce_fused.fused_ce: target at position t = labels[i, t+1]."""
    b, t = labels.shape
    targets = torch.full((b, t), -100, dtype=torch.int32, device=labels.device)
    targets[:, : t - 1] = labels[:, 1:].to(torch.int32)
    return targets.reshape(-1)


@pytest.mark.parametrize("shape", [(4, 512, 151936), (2, 33, 2048)])
def test_fused_ce_matches_pytorch(shape):
    from solution.ce_fused import fused_ce

    torch.manual_seed(0)
    b, t, v = shape

    def reference():
        logits = (torch.randn(b, t, v, device="cuda", dtype=torch.float16) * 2).requires_grad_(True)
        labels = torch.randint(0, v, (b, t), device="cuda", dtype=torch.long)
        labels[:, : t // 3] = -100

        shift_logits = logits[:, :-1, :].float()  # ForCausalLMLoss does logits.float()
        shift_labels = labels[:, 1:]
        loss = F.cross_entropy(
            shift_logits.reshape(-1, v), shift_labels.reshape(-1), ignore_index=-100)
        loss.backward()
        x = logits.detach().float()
        return logits.detach(), labels, loss.detach(), logits.grad, x

    logits, labels, ref_loss, ref_dlogits, x = reference()
    targets = _make_targets(labels)

    logits2 = logits.clone().requires_grad_(True)
    ce, ent, amax = fused_ce(logits2, labels)

    valid = targets != -100
    my_loss = ce[valid].sum() / valid.sum()
    torch.testing.assert_close(my_loss.item(), ref_loss.item(), rtol=1e-4, atol=1e-5)

    ce_view = ce.view(b, t)  # position (i, t) predicts labels[i, t+1]; last column invalid

    # per-position CE (fp32 reference, same input values)
    ref_ce = F.cross_entropy(
        x[:, :-1, :].reshape(-1, v), labels[:, 1:].reshape(-1),
        ignore_index=-100, reduction="none")
    torch.testing.assert_close(ce_view[:, : t - 1].reshape(-1), ref_ce, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(ce_view[:, t - 1], torch.zeros_like(ce_view[:, t - 1]),
                               rtol=0, atol=0)

    # entropy (fp32 reference)
    ref_ent = -(F.softmax(x, dim=-1) * F.log_softmax(x, dim=-1)).sum(dim=-1).reshape(-1)
    torch.testing.assert_close(ent, ref_ent, rtol=1e-3, atol=1e-3)

    # argmax
    torch.testing.assert_close(amax, x.argmax(dim=-1).reshape(-1).int(), rtol=0, atol=0)

    # backward: dlogits = (softmax - onehot) / count, stored in fp16
    my_loss.backward()
    torch.testing.assert_close(logits2.grad, ref_dlogits, rtol=2e-2, atol=2e-2)


# ---------------------------------------------------------------------------
# Fused RMSNorm (Qwen2, hidden 1536)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(4, 512, 1536), (2, 37, 1536)])
def test_fused_rmsnorm_matches_pytorch(shape):
    from solution.rmsnorm_fused import FusedRMSNorm, _HIDDEN

    b, t, h = shape
    assert h == _HIDDEN
    eps = 1e-6
    torch.manual_seed(0)

    # reference: the exact formula of transformers 5.16.1 Qwen2RMSNorm.forward
    x_ref = (torch.randn(shape, device="cuda", dtype=torch.float16) * 0.5).requires_grad_(True)
    w = torch.randn(h, device="cuda", dtype=torch.float32) * 0.1 + 1.0
    x32 = x_ref.float()
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    y_ref = w * (x32 * torch.rsqrt(var + eps)).to(torch.float16)  # fp32, like the base
    dy = torch.randn(shape, device="cuda", dtype=torch.float32)
    y_ref.backward(dy)
    ref_dx = x_ref.grad

    x = x_ref.detach().clone().requires_grad_(True)
    y = FusedRMSNorm.apply(x, w, eps)
    torch.testing.assert_close(y, y_ref.detach(), rtol=1e-4, atol=1e-5)
    y.backward(dy)
    torch.testing.assert_close(x.grad, ref_dx, rtol=2e-2, atol=2e-2)


# ---------------------------------------------------------------------------
# Fused LoRA path (forward + the four backward gradients, vs autograd)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m, i_in, out_dim", [(8, 1536, 1536), (16, 1536, 256), (3, 8960, 1536)])
def test_fused_lora_matches_pytorch(m, i_in, out_dim):
    from solution.lora_fused import FusedLoRA

    torch.manual_seed(0)
    r, s = 16, 2.0
    # base dtypes: the base layer output stays fp32, the LoRA branch runs fp16
    result = (torch.randn(m, out_dim, device="cuda", dtype=torch.float32) * 0.1
              ).requires_grad_(True)
    x32 = (torch.randn(m, i_in, device="cuda", dtype=torch.float32) * 0.1
           ).requires_grad_(True)
    A = torch.randn(r, i_in, device="cuda", dtype=torch.float16) * 0.1
    B = torch.randn(out_dim, r, device="cuda", dtype=torch.float16) * 0.1

    # the base expression, through plain autograd (fp16 GEMMs, fp32 add)
    def base_expr(res, x):
        t = x.to(torch.float16) @ A.t()
        return res + (t @ B.t()) * s

    dy = torch.randn(m, out_dim, device="cuda", dtype=torch.float32)

    out_ref = base_expr(result, x32)
    out_ref.backward(dy)
    ref_result_grad, ref_x_grad = result.grad, x32.grad

    result2 = result.detach().clone()
    x32_2 = x32.detach().clone()
    out = FusedLoRA.apply(result2, x32_2.to(torch.float16), A, B, s)
    out.backward(dy)
    torch.testing.assert_close(out, out_ref.detach(), rtol=2e-2, atol=2e-2)
    # the base layer must receive dy (the gradient is not cut)
    torch.testing.assert_close(result2.grad, ref_result_grad, rtol=2e-2, atol=2e-2)
    # the input receives the A-path gradient (fp16, then cast back by the graph)
    torch.testing.assert_close(x32_2.grad.float(), ref_x_grad, rtol=2e-2, atol=2e-2)


# ---------------------------------------------------------------------------
# Fused NF4 dequant + GEMM (forward and backward, vs a torch dequant reference)
# ---------------------------------------------------------------------------


def _quantize_for_nf4_test(w, blocksize=64):
    from solution.test_nf4 import quantize_torch

    return quantize_torch(w, blocksize=blocksize)


@pytest.mark.parametrize("n, k", [(256, 512), (512, 1024), (128, 8960)])
def test_fused_nf4_gemm_matches_pytorch(n, k):
    """Non-square shapes on purpose: a transposed orientation in the backward
    kernel would read the wrong packed rows and fail here."""
    from solution import nf4_gemm

    torch.manual_seed(0)
    m = 24
    w = torch.randn(n, k, device="cuda") * 0.05
    packed, qpacked, s2, offset, _ = _quantize_for_nf4_test(w.to(torch.float32).cpu())
    packed = packed.to("cuda")
    qpacked = qpacked.to("cuda")
    s2 = s2.to("cuda")
    offset = offset.to("cuda")
    lut = nf4_gemm._get_lut("cuda")

    w_deq = nf4_gemm.ref_dequant_rows(packed, qpacked, s2, offset, n, k)
    x = (torch.randn(m, k, device="cuda") * 0.1).to(torch.float16)

    out = nf4_gemm.NF4Linear.apply(x, packed, qpacked, s2, offset, lut, n, k)
    ref = (x.float() @ w_deq.t()).to(torch.float16)
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)

    gy = torch.randn(m, n, device="cuda", dtype=torch.float16) * 0.1
    out.backward(gy)
    ref_gx = (gy.float() @ w_deq).to(torch.float16)
    torch.testing.assert_close(x.grad.float(), ref_gx.float(), rtol=2e-2, atol=2e-2)
