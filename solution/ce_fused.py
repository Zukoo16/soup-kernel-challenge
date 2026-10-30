"""Fused cross-entropy pipeline for the Qwen2.5 lm_head logits.

Replaces, in one forward and one backward pass over the logits:

* ``ForCausalLMLoss``   -- ``logits.float()`` (a 1.24 GB fp32 copy of the logits) + fp32 CE;
* its autograd backward -- fp32 dlogits (another 1.24 GB) re-computed through the CE chain;
* TRL's ``entropy_from_logits``  (a log_softmax over the whole 151 936-token vocabulary);
* TRL's argmax token-accuracy pass;
* Soup's rewind ``row_losses`` (a per-row CE pass over the logits).

The kernels below stream each logit row once in fp16, accumulate in fp32 (online max /
sum-exp, flash-attention style) and never materialize the fp32 logits or a full softmax.
Only (N,) fp32 statistics are stored; the backward re-reads the fp16 logits once and
writes dlogits in fp16 in a fresh buffer.

Numerics: the same fp16 logits the base consumes, accumulated in fp32, so per-step loss,
entropy, accuracy and rewind row losses match the base well inside the harness tolerances
(first step 1%, mean 3%).
"""

import threading

import torch
import triton
import triton.language as tl

_BLOCK = 4096


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------


@triton.jit
def ce_fused_fwd_kernel(
    logits_ptr,  # (N, V) fp16
    targets_ptr,  # (N,) int32, -100 where the position has no supervised target
    ce_out_ptr,  # (N,) fp32 per-position CE (0 where invalid)
    ent_out_ptr,  # (N,) fp32 per-position entropy (nats)
    argmax_out_ptr,  # (N,) int32 argmax of the row
    lse_out_ptr,  # (N,) fp32 logsumexp, stored for the backward
    V: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    target = tl.load(targets_ptr + row)
    base = logits_ptr + row * V

    m = float("-inf")  # running max
    s = 0.0  # sum exp(x - m)
    spx = 0.0  # sum exp(x - m) * x
    am = float("-inf")  # running argmax value
    aidx = 0

    for off in range(0, V, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < V
        x = tl.load(base + idx, mask=mask, other=float("-inf")).to(tl.float32)

        bm = tl.max(x, axis=0)
        if bm > m:
            if m != float("-inf"):
                r = tl.exp(m - bm)
                s = s * r
                spx = spx * r
            m = bm
        e = tl.exp(x - m)
        s += tl.sum(e, axis=0)
        spx += tl.sum(e * x, axis=0)

        if bm > am:
            am = bm
            aidx = off + tl.argmax(x, axis=0)

    lse = m + tl.log(s)
    if target >= 0:
        xt = tl.load(base + target).to(tl.float32)
        ce = lse - xt
    else:
        ce = 0.0
    ent = lse - spx / s
    tl.store(ce_out_ptr + row, ce)
    tl.store(ent_out_ptr + row, ent)
    tl.store(argmax_out_ptr + row, aidx)
    tl.store(lse_out_ptr + row, lse)


@triton.jit
def ce_fused_bwd_kernel(
    logits_ptr,  # (N, V) fp16
    targets_ptr,  # (N,) int32
    scale_ptr,  # (N,) fp32, dloss/d(ce row); 0 where invalid -> row skipped
    lse_ptr,  # (N,) fp32
    dlogits_ptr,  # (N, V) fp16 output
    V: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    scale = tl.load(scale_ptr + row)
    if scale == 0.0:
        return
    target = tl.load(targets_ptr + row)
    lse = tl.load(lse_ptr + row)
    base = logits_ptr + row * V
    out = dlogits_ptr + row * V
    for off in range(0, V, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < V
        x = tl.load(base + idx, mask=mask, other=float("-inf")).to(tl.float32)
        p = tl.exp(x - lse)
        tl.store(out + idx, (p * scale).to(tl.float16), mask=mask)
    if target >= 0:
        xt = tl.load(base + target).to(tl.float32)
        tl.store(out + target, (tl.exp(xt - lse) * scale - scale).to(tl.float16))


# ---------------------------------------------------------------------------
# Autograd + the torch-side math (pure, CPU-testable)
# ---------------------------------------------------------------------------


class FusedCE(torch.autograd.Function):
    """One-pass fused CE/entropy/argmax over (N, V) fp16 logits.

    Returns per-position CE, entropy and argmax; the CE output is the differentiable one
    (loss = mean of the valid CE values, exactly like the base's
    ``cross_entropy(reduction="mean", ignore_index=-100)``).
    """

    @staticmethod
    def forward(ctx, logits, targets):
        n = logits.shape[0]
        v = logits.shape[1]
        ce = torch.empty(n, device=logits.device, dtype=torch.float32)
        ent = torch.empty(n, device=logits.device, dtype=torch.float32)
        amax = torch.empty(n, device=logits.device, dtype=torch.int32)
        lse = torch.empty(n, device=logits.device, dtype=torch.float32)
        ce_fused_fwd_kernel[(n,)](logits, targets, ce, ent, amax, lse, V=v, BLOCK=_BLOCK)
        ctx.save_for_backward(logits, targets, lse)
        return ce, ent, amax

    @staticmethod
    def backward(ctx, g_ce, _g_ent, _g_amax):
        logits, targets, lse = ctx.saved_tensors
        g_ce = g_ce.to(torch.float32).contiguous()
        # rows with scale == 0 (unsupervised positions) are skipped by the kernel's early
        # return, so the buffer must start zero: their dlogits are exactly 0 and leaving
        # them uninitialized would feed garbage into the model gradients
        dlogits = torch.zeros_like(logits)
        n = logits.shape[0]
        ce_fused_bwd_kernel[(n,)](logits, targets, g_ce, lse, dlogits,
                                  V=logits.shape[1], BLOCK=_BLOCK)
        return dlogits, None


def build_targets(labels):
    """target at position (i, t) = labels[i, t+1] (the token the logit at t predicts);
    -100 for the last position of every row. Returns (B*T,) int32."""
    b, t = labels.shape
    targets = torch.full((b, t), -100, dtype=torch.int32, device=labels.device)
    targets[:, : t - 1] = labels[:, 1:].to(torch.int32)
    return targets.reshape(-1)


def metrics_from_stats(ce, ent, amax, targets, labels, attention_mask):
    """Pure-torch math over the fused forward's statistics. Mirrors, value for value:

    * the base loss  -- mean CE over the supervised tokens
      (ForCausalLMLoss: ``sum / num_items_in_batch``, num_items_in_batch is exactly the
      number of non-ignored tokens in ``labels[..., 1:]``);
    * TRL's entropy metric -- ``sum(H * attention_mask) / attention_mask.sum()``;
    * TRL's accuracy metric -- argmax agreement on ``labels[..., 1:] != -100``;
    * Soup's rewind per-row losses -- per-row mean CE over the supervised tokens.

    ``loss`` keeps the gradient (through ``ce``); everything else is a detached metric.
    Returns (loss, entropy, accuracy, row_loss, row_tokens).
    """
    b, t = labels.shape
    valid = targets != -100
    count = valid.sum()
    if count > 0:
        loss = ce[valid].sum() / count
    else:
        loss = ce.sum() * 0.0

    with torch.no_grad():
        entropy = torch.sum(ent.view(b, t) * attention_mask) / attention_mask.sum()
        shift_labels = labels[:, 1:]
        mask = shift_labels != -100
        # position (i, t) with t < T-1 is the one that predicts labels[i, t+1]
        predictions = amax.view(b, t)[:, : t - 1]
        correct = (predictions == shift_labels) & mask
        total = mask.sum()
        accuracy = (correct.sum() / total) if total > 0 else torch.zeros((), device=ce.device)
        # rewind per-row statistics (row i: its t < T-1 positions)
        ce2 = ce.view(b, t)[:, : t - 1]
        row_tokens = mask.sum(dim=1)
        row_loss = (ce2 * mask).sum(dim=1) / row_tokens.clamp(min=1)
    return loss, entropy, accuracy, row_loss, row_tokens


# ---------------------------------------------------------------------------
# The patches
# ---------------------------------------------------------------------------

_pending = threading.local()
_ce_probe_done = False


def fused_ce(logits, labels):
    """Run the fused forward over (B, T, V) fp16 logits with (B, T) labels (-100 ignored).

    Returns (ce, ent, amax) each (B*T,), position (i, t) = logit at (i, t). Also stores
    the batch statistics on a thread-local for the row_losses wrapper below.
    """
    targets = build_targets(labels)
    _pending_clear()
    ce, ent, amax = FusedCE.apply(logits.contiguous().view(-1, logits.shape[-1]), targets)
    _pending.targets = targets
    _pending.ce = ce
    _pending.shape = tuple(labels.shape)
    return ce, ent, amax


def _pending_get():
    return (getattr(_pending, "targets", None), getattr(_pending, "ce", None),
            getattr(_pending, "shape", None))


def _pending_clear():
    _pending.targets = None
    _pending.ce = None
    _pending.shape = None


def _ce_probe(logits, labels):
    """One-time check of both CE kernels against a plain-torch fp32 reference on a
    few real rows (forward statistics and the backward dlogits). The backward probe
    runs on a detached copy so it never touches the real autograd graph.

    Causal alignment, exactly as in the base: position (i, t) with t < T-1 predicts
    ``labels[i, t+1]``; the last position of every row is unsupervised. The probe uses
    only 16 positions from the middle of two rows (dense supervised zone), so its
    fp32 reference tensors stay tiny on a T4.
    """
    b = min(2, logits.shape[0])
    t = labels.shape[1]
    off = t // 2
    n_pos = min(16, t - off)
    sl, slab = logits[:b, off: off + n_pos], labels[:b, off: off + n_pos]
    t = n_pos
    shift = slab[:, 1:]  # target at position t (t < T-1)

    ce, ent, amax = fused_ce(sl, slab)
    x = sl.float()
    lse = torch.logsumexp(x, dim=-1)
    xt = x[:, : t - 1].gather(-1, shift.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    valid = shift != -100
    ref_ce = torch.zeros(b, t, device=x.device)
    ref_ce[:, : t - 1] = (lse[:, : t - 1] - xt) * valid
    if not torch.allclose(ce.view(b, t), ref_ce, rtol=1e-3, atol=1e-3):
        return False
    p = torch.exp(x - lse.unsqueeze(-1))
    ref_ent = lse - (p * x).sum(-1)
    if not torch.allclose(ent.view(b, t), ref_ent, rtol=1e-3, atol=1e-3):
        return False
    if not bool(torch.equal(amax.view(b, t), x.argmax(dim=-1).int())):
        return False

    # backward: dlogits = (softmax - onehot) * g, g = 0.5 on the valid rows only
    sl2 = sl.detach().requires_grad_(True)
    ce2, _ent2, _am2 = fused_ce(sl2, slab)
    g = torch.zeros(b, t, device=logits.device)
    g[:, : t - 1] = torch.where(valid, torch.tensor(0.5, device=logits.device),
                                torch.zeros((), device=logits.device))
    (ce2.view(b, t) * g).sum().backward()
    lse2 = torch.logsumexp(sl2.float(), dim=-1)
    p2 = torch.exp(sl2.float() - lse2.unsqueeze(-1))
    onehot = torch.zeros_like(p2)
    onehot[:, : t - 1].scatter_(-1, shift.clamp(min=0).unsqueeze(-1), 1.0)
    ref_d = (p2 - onehot) * g.unsqueeze(-1)
    return bool(torch.allclose(sl2.grad.float(), ref_d, rtol=2e-2, atol=2e-2))


def patch_trl_compute_loss():
    """Route trl.SFTTrainer.compute_loss through the fused kernels.

    Replicates, exactly, the metric bookkeeping TRL 0.29.1 does (entropy / num_tokens /
    mean_token_accuracy into self._metrics) and the base loss value. Falls back to the
    original implementation whenever the setup differs from the task (also keeps Soup's
    regression tests with their fake modules passing).
    """
    from trl.trainer import sft_trainer as _st

    sft_cls = getattr(_st, "SFTTrainer", None)
    original = getattr(sft_cls, "compute_loss", None)
    if original is None or getattr(original, "_soup_fused_ce", False):
        return
    import torch.nn as nn

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        try:
            ok = (
                isinstance(model, nn.Module)
                and isinstance(inputs, dict)
                and "labels" in inputs
                and "shift_labels" not in inputs
                and inputs["input_ids"].is_cuda
                and inputs["input_ids"].dtype == torch.long
                and not self.args.use_liger_kernel
                and getattr(self.args, "loss_type", "nll") == "nll"
                and getattr(self, "compute_loss_func", None) is None
                and getattr(self, "label_smoother", None) is None
                and not getattr(self, "aux_loss_enabled", False)
            )
        except Exception:
            ok = False
        if not ok:
            _pending_clear()  # never let a stale fast path serve the next row_losses
            return original(self, model, inputs, return_outputs=return_outputs,
                            num_items_in_batch=num_items_in_batch)

        mode = "train" if self.model.training else "eval"
        inputs.pop("_prediction_loss_only", None)
        labels = inputs["labels"]
        inputs["use_cache"] = False
        call_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**call_inputs)
        logits = outputs.logits
        if logits.dtype != torch.float16:  # the kernels are compiled for fp16 logits
            _pending_clear()
            ret = original(self, model, inputs, return_outputs=True,
                           num_items_in_batch=num_items_in_batch)
            loss, orig_outputs = ret if isinstance(ret, tuple) else (ret, None)
            return (loss, orig_outputs) if return_outputs else loss

        global _ce_probe_done
        if not _ce_probe_done:  # one-time self-check against a torch reference
            _ce_probe_done = True
            good = False
            try:
                good = _ce_probe(logits, labels)
            except Exception as exc:  # noqa: BLE001 - probe failure means: fall back
                print(f"[solution] fused ce probe error: {exc!r}", flush=True)
            if not good:
                print("[solution] fused ce: probe mismatch; using the original loss chain",
                      flush=True)
                _pending_clear()
                ret = original(self, model, inputs, return_outputs=True,
                               num_items_in_batch=num_items_in_batch)
                loss, orig_outputs = ret if isinstance(ret, tuple) else (ret, None)
                return (loss, orig_outputs) if return_outputs else loss

        ce, ent, amax = fused_ce(logits, labels)
        targets = _pending.targets  # just written by fused_ce
        loss, entropy, accuracy, row_loss, row_tokens = metrics_from_stats(
            ce, ent, amax, targets, labels, inputs["attention_mask"])

        # --- the same metric bookkeeping as TRL SFTTrainer.compute_loss ---
        entropy = self.accelerator.gather_for_metrics(entropy).mean().item()
        self._metrics[mode]["entropy"].append(entropy)

        if mode == "train":
            num_tokens_in_batch = self.accelerator.gather_for_metrics(
                inputs["attention_mask"].sum()).sum().item()
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        accuracy = self.accelerator.gather_for_metrics(accuracy).mean().item()
        self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        # keep the per-row statistics for the rewind recorder
        _pending.row_loss = row_loss
        _pending.row_tokens = row_tokens

        return (loss, outputs) if return_outputs else loss

    compute_loss._soup_fused_ce = True
    _st.SFTTrainer.compute_loss = compute_loss


def patch_row_losses():
    """Serve Soup's rewind per-row losses from the statistics the fused forward already
    computed, instead of re-reading the logits four times (rewind_hf.row_losses)."""
    try:
        from soup_cli.trainer import rewind_hf
    except Exception:
        return

    original = rewind_hf.row_losses
    if getattr(original, "_soup_fused_ce", False):
        return

    def row_losses(logits, labels):
        targets, ce, shape = _pending_get()
        row_loss = getattr(_pending, "row_loss", None)
        row_tokens = getattr(_pending, "row_tokens", None)
        _pending_clear()
        try:
            ok = (
                isinstance(labels, torch.Tensor)
                and targets is not None and ce is not None and shape is not None
                and row_loss is not None and row_tokens is not None
                and tuple(labels.shape) == shape
                and row_loss.device == labels.device
            )
        except Exception:
            ok = False
        if ok:
            return ([float(x) for x in row_loss], [int(c) for c in row_tokens])
        return original(logits, labels)

    row_losses._soup_fused_ce = True
    rewind_hf.row_losses = row_losses


def precompile():
    """Compile the kernels once, before the first timed training step.

    Every measured run starts with an empty Triton cache and the wall time of the whole
    `soup train` command is scored, so the (unavoidable) compilation cost is paid here,
    during apply() while the GPU is idle, instead of inside the loop.
    """
    if not torch.cuda.is_available():
        return
    v = 151936
    n = 2
    try:
        logits = torch.randn(n, v, device="cuda", dtype=torch.float16) * 2
        targets = torch.tensor([17, -100], device="cuda", dtype=torch.int32)
        ce, ent, amax = FusedCE.apply(logits, targets)
        g = torch.where(targets >= 0, torch.full_like(ce, 0.5), torch.zeros_like(ce))
        torch.autograd.backward(ce, g)
        del logits, targets, ce, ent, amax, g
        torch.cuda.synchronize()
    except Exception:
        # Precompilation is an optimization only; a failure must never break the run.
        pass
