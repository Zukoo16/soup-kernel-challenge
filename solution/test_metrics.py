"""CPU-testable math of the fused CE pipeline.

The Triton kernels themselves are tested on a GPU in test_kernels.py; this file checks the
torch-side math (indexing, reductions, metric formulas) against the exact base formulas,
so an index bug is caught before it ever touches a GPU.
"""

import sys
import types

import pytest
import torch
import torch.nn.functional as F


def _stats_from_logits(logits, labels):
    """Pure-torch reference of what the fused forward kernel computes (fp32)."""
    from solution.ce_fused import build_targets

    x = logits.float()
    targets = build_targets(labels)
    lse = torch.logsumexp(x, dim=-1)  # (b, t)
    valid = targets >= 0
    xt = x.reshape(-1, x.shape[-1]).gather(1, targets.clamp(min=0).unsqueeze(1)).squeeze(1)
    ce = torch.where(valid, lse.reshape(-1) - xt, torch.zeros_like(xt)).reshape(-1)
    p = torch.exp(x - lse.unsqueeze(-1))
    ent = (lse - (p * x).sum(-1)).reshape(-1)
    amax = x.argmax(dim=-1).reshape(-1).int()
    return ce, ent, amax, targets


def _base_row_losses(logits, labels):
    """soup_cli 0.77.0 rewind_hf.row_losses, pure torch (CPU-safe)."""
    row_loss, row_tokens = [], []
    with torch.no_grad():
        shift_labels = labels[:, 1:]
        for i in range(shift_labels.size(0)):
            targets_i = shift_labels[i]
            token_ce = F.cross_entropy(
                logits[i, :-1, :].detach(), targets_i, reduction="none",
                ignore_index=-100).float()
            m = (targets_i != -100).to(token_ce.dtype)
            tokens = m.sum()
            total = (token_ce * m).sum()
            row_loss.append(float(total / tokens.clamp(min=1.0)))
            row_tokens.append(int(tokens.item()))
    return row_loss, row_tokens


def _make_batch(b=4, t=33, v=512, seed=0):
    torch.manual_seed(seed)
    logits = torch.randn(b, t, v) * 2
    labels = torch.randint(0, v, (b, t), dtype=torch.long)
    labels[:, : t // 3] = -100
    attention_mask = torch.ones(b, t, dtype=torch.long)
    attention_mask[:, t - 8:] = 0
    return logits, labels, attention_mask


def test_metrics_from_stats_matches_base_formulas():
    from solution.ce_fused import metrics_from_stats

    logits, labels, attention_mask = _make_batch()
    ce, ent, amax, targets = _stats_from_logits(logits, labels)
    loss, entropy, accuracy, row_loss, row_tokens = metrics_from_stats(
        ce, ent, amax, targets, labels, attention_mask)

    # base loss: ForCausalLMLoss -- sum over valid / num_items_in_batch
    shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    shift_labels = labels[:, 1:].reshape(-1)
    n_items = shift_labels.ne(-100).sum()
    ref_loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100,
                               reduction="sum") / n_items
    # fp32 rounding of two different algorithms (online lse vs CE-internal); the
    # harness itself allows 1% on the first step, so 1e-5 here is 5 orders tighter.
    assert loss.item() == pytest.approx(ref_loss.item(), rel=1e-5)

    # base entropy metric (TRL): masked mean of the per-position entropy
    x = logits.float()
    ref_ent = -(F.softmax(x, dim=-1) * F.log_softmax(x, dim=-1)).sum(dim=-1)
    ref_entropy = (ref_ent * attention_mask).sum() / attention_mask.sum()
    assert entropy.item() == pytest.approx(ref_entropy.item(), rel=1e-5)

    # base accuracy metric (TRL): argmax agreement on shift_labels != -100
    predictions = x.argmax(dim=-1)[:, :-1]
    shift_labels_2d = labels[:, 1:]
    mask = shift_labels_2d != -100
    ref_acc = ((predictions == shift_labels_2d) & mask).sum() / mask.sum()
    assert accuracy.item() == pytest.approx(ref_acc.item(), rel=1e-9)

    # base rewind row losses
    ref_row_loss, ref_row_tokens = _base_row_losses(logits, labels)
    assert row_tokens.tolist() == ref_row_tokens
    for got, ref in zip(row_loss.tolist(), ref_row_loss):
        assert got == pytest.approx(ref, rel=1e-5)


def test_metrics_from_stats_all_invalid_batch():
    from solution.ce_fused import metrics_from_stats

    b, t, v = 2, 17, 64
    logits = torch.randn(b, t, v)
    labels = torch.full((b, t), -100, dtype=torch.long)
    attention_mask = torch.ones(b, t, dtype=torch.long)
    ce, ent, amax, targets = _stats_from_logits(logits, labels)
    loss, entropy, accuracy, row_loss, row_tokens = metrics_from_stats(
        ce, ent, amax, targets, labels, attention_mask)
    assert loss.item() == 0.0
    assert row_tokens.tolist() == [0, 0]
    assert row_loss.tolist() == [0.0, 0.0]


def test_row_losses_wrapper(monkeypatch):
    import solution.ce_fused as mod

    # stub the soup_cli modules the patcher imports
    pkg = types.ModuleType("soup_cli")
    trainer = types.ModuleType("soup_cli.trainer")
    rewind = types.ModuleType("soup_cli.trainer.rewind_hf")
    rewind.row_losses = _base_row_losses
    monkeypatch.setitem(sys.modules, "soup_cli", pkg)
    monkeypatch.setitem(sys.modules, "soup_cli.trainer", trainer)
    monkeypatch.setitem(sys.modules, "soup_cli.trainer.rewind_hf", rewind)
    monkeypatch.setattr(mod, "_pending", types.SimpleNamespace())
    mod.patch_row_losses()
    wrapper = rewind.row_losses
    assert getattr(wrapper, "_soup_fused_ce", False)

    logits, labels, _attn = _make_batch(b=2, t=17, v=64)
    ce, ent, amax, targets = _stats_from_logits(logits, labels)
    _loss, _ent, _acc, row_loss, row_tokens = mod.metrics_from_stats(
        ce, ent, amax, targets, labels, torch.ones_like(labels))
    mod._pending.targets = targets
    mod._pending.ce = ce
    mod._pending.shape = tuple(labels.shape)
    mod._pending.row_loss = row_loss
    mod._pending.row_tokens = row_tokens

    got_loss, got_tokens = wrapper(logits, labels)
    ref_loss, ref_tokens = _base_row_losses(logits, labels)
    assert got_tokens == ref_tokens
    for got, ref in zip(got_loss, ref_loss):
        assert got == pytest.approx(ref, rel=1e-9)

    # stale / shape-mismatched stats -> fall back to the original implementation
    other_logits = torch.randn(1, 5, 64)
    other_labels = torch.randint(0, 64, (1, 5), dtype=torch.long)
    got = wrapper(other_logits, other_labels)
    ref = _base_row_losses(other_logits, other_labels)
    assert got[1] == ref[1]
    for a, b_ in zip(got[0], ref[0]):
        assert a == pytest.approx(b_, rel=1e-9)

    # second call after the stats were consumed -> also the original implementation
    got2 = wrapper(other_logits, other_labels)
    assert got2 == got
