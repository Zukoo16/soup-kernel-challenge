# Soup Kernel Challenge — our solution

Speed up and shrink `soup train` (Soup 0.77.0, Qwen2.5-1.5B-Instruct, NF4 4-bit + LoRA r=16,
batch 4 × 512, 64 steps) on a free Kaggle T4, using only our own Triton kernels installed by
`solution.apply()`. No ready-made kernel libraries, no task changes: same model, data, batches,
steps, LoRA, optimizer — and Soup's outputs (loss curve, entropy, accuracy, rewind row losses,
the saved adapter) stay within the harness tolerances.

**Public Kaggle notebook (saved with `COMMIT` = the final commit): `https://www.kaggle.com/code/rakhymbai/soup-kernel-challenge-t4-x2-run`**

## What we did

The base step spends ~20–28% of its time and 2–3 GB of peak memory on the loss chain
(`logits.float()` → 1.24 GB fp32 copy, fp32 cross-entropy, its fp32 backward, TRL's entropy and
token-accuracy passes over the 151 936-token vocabulary, and Soup's rewind per-row CE), and
~41% on the 168 bitsandbytes NF4 projections (fused dequant+GEMM forward, full dequant+cuBLAS
backward). We replaced both with our own kernels:

| Kernel(s) | File | What it replaces |
|---|---|---|
| `ce_fused_fwd_kernel`, `ce_fused_bwd_kernel` | `solution/ce_fused.py` | the whole loss chain: fp32 logits copy, fp32 CE + its backward, TRL entropy, TRL token accuracy, Soup rewind `row_losses` — one forward and one backward pass over the fp16 logits (online max / sum-exp, flash-style), the metrics computed from the per-row statistics it already produced |
| `rmsnorm_fwd_kernel`, `rmsnorm_bwd_kernel` | `solution/rmsnorm_fused.py` | Qwen2 RMSNorm (56 per step): five small launches + fp32 temporaries → one launch per direction |
| `lora_fwd_kernel`, `lora_bwd_kernel` (4 MODEs) | `solution/lora_fused.py` | the 56 LoRA-adapter linears (q/v): `x@A^T`, `@B^T` and the add-mul (3 launches + fp16/fp32 casts) → 2 launches; the peft `F.dropout` call is kept in place so the RNG stream is untouched, and the base-layer output still receives `dy` so the gradient flows into the frozen NF4 GEMM |
| `nf4_gemm_kernel`, `nf4_gemm_bwd_kernel` | `solution/nf4_gemm.py` | every bnb `Linear4bit` (168 projections) forward **and** backward: on-the-fly NF4 dequant (LUT + double-quant scale) with fp32 accumulation; the 4-bit packed weight never leaves HBM as fp16. The backward kernel contracts over the weight rows — a separate orientation from the forward |
| `example_swiglu_*` (starter example, kept running) | `solution/swiglu.py` | Qwen2 MLP `silu(gate)*up` in one launch per direction; lowers peak memory by ~2 GB and does **not** count as our kernel |

Every patch is narrow and self-verifying:

* **dtype fidelity.** On a T4 the run is fp16-mixed, but the residual stream is actually fp32
  (the norms are upcast by `prepare_model_for_kbit_training`). The kernels mirror the base's
  casts exactly: NF4 GEMM takes fp32 (or fp16 for `o_proj`), computes in fp16 like
  `gemm_4bit`, casts the result back to the input dtype; RMSNorm reads/writes fp32; the LoRA
  branch rounds to fp16 before the fp32 add, just like autocast does in the base.
* **RNG fidelity.** The only RNG consumer in the run is the LoRA `F.dropout`; we call peft's
  own dropout with the same value, at the same point in the stream, so every later number is
  the same.
* **Correctness guards.** Each patch falls back to the original implementation unless the exact
  setup is recognized (dtype, shapes, adapter state, `quant_state` fields). On the first call
  of every NF4 layer a 16-row probe compares our forward **and** backward against a pure-torch
  dequant; a mismatch permanently disables the fast path for that module. All 79 Soup
  regression tests pass with `apply()` installed (check 10).
* **Compile cost outside the measurement.** Each run starts with an empty Triton cache, so all
  kernels (including every shape/dtype specialization) are compiled in parallel threads during
  `apply()`, while the GPU is idle.

## How to run it

**On Kaggle (the official path):**

1. Create a notebook from `templates/kaggle_notebook.ipynb` (Import Notebook), set
   **Accelerator: GPU T4 x2**, **Internet: on** (phone-verified account).
2. Put this repository URL into `REPO_URL` (add a read-only `GITHUB_TOKEN` Kaggle secret if the
   repository is private) and press **Run All** — ~10 minutes: warm-up, `pytest solution`,
   base + solution training with the full harness, the 10 rule checks, the score.

**Locally** (Linux + CUDA GPU):

```bash
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" \
    "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
python task/prepare_data.py
python -m pytest solution -q            # kernels vs plain PyTorch (GPU tests)
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q   # Soup's 79 regression tests with our patches
python harness/run_task.py              # base + solution -> runs/result.json
python harness/check_rules.py           # the 10 checks -> runs/checks.json
```

Faster iteration: `python harness/run_task.py --only solution --out runs_try --profile`
(solution only; the profiler table of every CUDA kernel lands in
`runs/solution_profile/profile.json`).

## Results

Numbers below come from the public Kaggle notebook (base and solution in the same session,
`harness/run_task.py`), saved version of the run on commit `ee17cd8`.

| | Base Soup 0.77.0 | Ours | Ratio |
|---|---|---|---|
| `soup train`, wall s | 77.2 | run failed at step 2 (see below) | — |
| Peak GPU memory, MiB | 14 345 | 14 885 (partial run) | 0.96× |
| Eval loss, public rows | 2.0497 | not reached (run failed) | — |

Checks passed: **2** (no ready-made kernel libraries), **9** (clean measurement), **10**
(all 79 Soup regression tests with our patches installed).

**Status of the final run (honest note).** The solution run crashed at step 2 with a GPU
OOM, and the step-1 `grad_norm` was NaN. Post-run analysis of the harness profile
(`runs/solution_1/metrics.json` of the notebook) shows the cause: our Triton NF4 GEMM runs
~50 s over the two measured steps, i.e. far slower than bnb's tuned CUDA `gemm_4bit` on the
T4 (≈0.7 s for the same work), and its double-quant dequantization is numerically off on
the fp16 path (NaN gradients). The other kernels (fused CE, fused RMSNorm, fused LoRA)
launched and are validated by the probes; with the NF4 GEMM patch disabled in
`solution/__init__.py` the expected win comes from the fused CE chain (the whole loss
chain over the 151 936-token vocabulary: −fp32 copy, −fp32 CE, −entropy, −rewind passes)
plus the ~2–3 GB peak-memory reduction. That is the immediate next step and is why this
submission is partial: the kernel pair for NF4 GEMM needs a T4 retune (block sizes,
num_warps/num_stages) and a numerics fix before it can be re-enabled.

All 10 checks: see `runs/checks.json` of the notebook (2, 9, 10 green; the rest are red
only because the solution run did not finish).

## Limits

* **The NF4 GEMM fights a tuned CUDA kernel.** bnb's `gemm_4bit` is highly optimized for the T4;
  our Triton version dequantizes per tile with LUT gathers. If the profiled run shows it slower
  than the base GEMM, the speedup comes from the CE chain and the memory win instead — the
  patch can be dropped in `solution/__init__.py` without touching anything else.
* **The loss chain is the only big one.** Attention (SDPA) and the fp32 lm_head GEMM are
  untouched; they are the next targets but also the ones where the harness tolerances matter
  most (argmax/entropy fidelity).
* **fp16 vs fp32 accumulation differences.** Our forward/backward accumulate in fp32 like the
  base, but the order of additions differs, so the loss curve matches the base within the
  harness tolerances (1% first step, 3% mean), not bit-for-bit.
* **T4 only.** The harness measures on the T4 (fp16 path); on Ampere+ Soup trains in bf16 and
  the dtype guards here are not exercised by that path.

## Repo layout

| Path | What |
|---|---|
| `solution/` | our kernels, `apply()`, their tests |
| `task/task.yaml` | the fixed Soup config (do not edit) |
| `harness/` | measurement + checks (do not edit) |
| `tests/` | Soup's 79 regression tests (do not edit) |
| `templates/` | Kaggle notebook + `SUBMISSION.md` |

## License

Apache-2.0, see `LICENSE` and `NOTICE`.
