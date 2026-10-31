# Submission: <team name>

## Links
- Repository, final commit: https://github.com/Zukoo16/soup-kernel-challenge/commit/25592fe29ff06b93745e03752c7f511e378df570
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link must
  be in your README):
  https://www.kaggle.com/code/<you>/<notebook>?scriptVersionId=<version>
- Demo video (up to 3 minutes): <link>
- Presentation: <link>

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| | | | | |

## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `ce_fused_fwd_kernel` | `solution/ce_fused.py` | fp32 logits copy, fp32 CE forward + backward, TRL entropy, TRL token accuracy, Soup rewind row losses | _from the notebook_ |
| `ce_fused_bwd_kernel` | `solution/ce_fused.py` | the fp32 CE backward (dlogits recompute) | _from the notebook_ |
| `rmsnorm_fwd_kernel` | `solution/rmsnorm_fused.py` | Qwen2 RMSNorm forward (56 per step) | _from the notebook_ |
| `rmsnorm_bwd_kernel` | `solution/rmsnorm_fused.py` | Qwen2 RMSNorm backward | _from the notebook_ |
| `lora_fwd_kernel` | `solution/lora_fused.py` | LoRA `x@A^T` + `@B^T` + add-mul on 56 q/v projections | _from the notebook_ |
| `lora_bwd_kernel` (MODE 0–3) | `solution/lora_fused.py` | LoRA backward: dx, dA, dB and the scaled dy@B | _from the notebook_ |
| `nf4_gemm_kernel` | `solution/nf4_gemm.py` | bnb NF4 dequant+GEMM forward on all 168 projections | _from the notebook_ |
| `nf4_gemm_bwd_kernel` | `solution/nf4_gemm.py` | bnb full-dequant + cuBLAS backward on all 168 projections | _from the notebook_ |

Ideas taken from other kernels, with links (we wrote the code ourselves):

- the online max / sum-exp scheme of the cross-entropy forward is the standard
  flash-attention logsumexp technique (no code taken).

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | | | speedup x |
| Peak GPU memory, MiB | | | base/ours x |
| Eval loss (public rows) | | | % vs base |

- [ ] All 10 checks of `harness/check_rules.py` passed (if not: which failed and why)
- [ ] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
Which operations we sped up (the loss chain over the 151 936-token vocabulary and the 168 NF4
projections, forward and backward), how the kernels work (flash-style online logsumexp, on-the-fly
NF4 dequant with double-quant scales in both GEMM orientations, dtype/RNG fidelity vs the base),
results against the base (speed, memory, quality), limits and next steps.

## Limitations
- The Triton NF4 GEMM fights bnb's tuned CUDA kernel on the T4; if it is slower in the profiled
  run, the win comes from the CE chain and the ~2–3 GB memory reduction instead (the patch can be
  disabled in `solution/__init__.py` without touching anything else).
- SDPA attention and the fp32 lm_head are untouched: the next targets, where the harness
  tolerances matter most.
- Loss curve matches the base within the harness tolerances (1% first step, 3% mean), not
  bit-for-bit (different fp32 accumulation order).
- Validated on the T4 (fp16 path); the bf16 path of Ampere+ GPUs is not exercised here.
