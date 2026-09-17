# Qwen3-235B frozen SpecDec results

## Methodology

- Target: `Qwen/Qwen3-235B-A22B`
- Base recipe: `grpo-qwen3-235b-16n4g.yaml`
- Hardware: 16 Lyris GB200 nodes, four GPUs per node
- Workload: 16 prompts × 32 generations, 8,192-token maximum sequence length
- Precision and runtime: BF16, inherited Triton MoE, generation TP8,
  `enforce_eager=false`, `FULL_AND_PIECEWISE` CUDA Graphs
- Baseline: the unmodified recipe concurrency; no SpecDec
- SpecDec: frozen drafter, method-aware CUDA Graph buckets, `max_num_seqs=64`
- Aggregation window: W&B steps 3–20 inclusive
- W&B project: `nvidia/sna-specdec`
- Submitted walltime: one hour for gates and three hours for 20-step runs. The
  successful baseline gate required 378 seconds of setup and 371 seconds for
  its first step, leaving approximately 50 minutes of margin for 20 steps.

The baseline intentionally preserves the official performance recipe without a
`max_num_seqs` override. The SpecDec arms use the S64 cap required by their
captured request-width buckets. Therefore, the primary table is an optimized
method comparison rather than a one-variable ablation. This distinction must
remain visible in any published result.

## Performance

| Configuration | Job | W&B | Gen TPS/GPU | Gen speedup | Generation time | E2E step time | E2E speedup | Acceptance | Mean accepted length |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline, no SpecDec | 3086570 | [9mt3ocw9](https://wandb.ai/nvidia/sna-specdec/runs/9mt3ocw9); running | — | 1.000x | — | — | 1.000x | — | — |
| EAGLE-3 K3 | 3086771 | gate pending | — | — | — | — | — | — | — |
| EAGLE-3 K5 | 3086775 | gate pending | — | — | — | — | — | — | — |
| DFlash B8 K5 | 3086779 | gate pending | — | — | — | — | — | — | — |
| DFlash B8 K7 | 3086783 | gate pending | — | — | — | — | — | — | — |
| DSpark B8 K5 | 3086787 | gate pending | — | — | — | — | — | — | — |
| DSpark B8 K7 | 3086791 | gate pending | — | — | — | — | — | — | — |
| DFlash B16 K11 | 3086795 | gate pending | — | — | — | — | — | — | — |
| DFlash B16 K13 | 3086799 | gate pending | — | — | — | — | — | — | — |
| DSpark B16 K11 | 3086803 | gate pending | — | — | — | — | — | — | — |
| DSpark B16 K13 | 3086807 | gate pending | — | — | — | — | — | — | — |

## Correctness checks

Before ranking any arm, compare the same Steps 3–20 window for:

- reward
- mean generated tokens per sample
- policy KL and generation KL/error metrics, when present
- loss, approximate entropy, and token probability-ratio diagnostics
- returned and missing W&B steps and the valid count for every reported metric

Do not report a speedup for a failed, truncated, or configuration-mismatched
run. Use `waiting baseline` until the complete baseline window is available.

## Baseline gate receipt

The one-step NUMA-fix gate completed end to end and is a runtime sanity check,
not the final performance result:

- Job: `3086466`
- W&B: <https://wandb.ai/nvidia/sna-specdec/runs/zshyatn8>
- Total step time: 370.57 seconds
- Generation: 149.99 seconds
- Generation throughput: 328.75 tokens/s/GPU
- E2E throughput: 133.06 tokens/s/GPU
- Mean generation length: 6,064.03 tokens
- Reward: 0.6504
- Generation KL error: 0.0056

## Baseline 20-step startup receipt

The full no-SpecDec baseline started on Lyris at 06:34:41 PDT in job
`3086570`. The first five minutes completed Ray head and 16-node worker
initialization without a host-memory OOM or traceback. W&B run
[9mt3ocw9](https://wandb.ai/nvidia/sna-specdec/runs/9mt3ocw9) is live. Final
performance remains pending until the complete Steps 3–20 window is available.

Step 1 completed successfully and closely reproduced the independent gate:

| Diagnostic | 1-step gate | Full-run Step 1 |
|---|---:|---:|
| Total step time | 370.57 s | 371.41 s |
| Generation time | 149.99 s | 150.21 s |
| Generation throughput/GPU | 328.75 tok/s | 324.58 tok/s |
| E2E throughput/GPU | 133.06 tok/s | 131.27 tok/s |
| Reward | 0.6504 | 0.6621 |
| Mean generation length | 6,064.03 | 5,994.68 |
| Generation KL error | 0.0056 | 0.0056 |

These are startup diagnostics, not the final release comparison. The final
table continues to require the complete Steps 3–20 window.

## Ptyche matched one-step diagnostics

The first matched Ptyche gates completed without an OOM or traceback. Their
generation lengths and quality diagnostics are close enough to treat this as a
runtime sanity check, but not as the final performance result.

| Configuration | W&B | Gen TPS/GPU | Gen speedup | E2E step time | E2E TPS/GPU | E2E speedup | Reward | Mean gen length | Gen KL error | Acceptance rate | Mean accepted length |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline, no SpecDec | [65bxq2im](https://wandb.ai/nvidia/sna-specdec/runs/65bxq2im) | 329.73 | 1.000x | 371.52 s | 132.66 | 1.000x | 0.6562 | 6,061.13 | 0.0056 | — | — |
| DFlash B8 K5 | [68u2ugzd](https://wandb.ai/nvidia/sna-specdec/runs/68u2ugzd) | 455.03 | 1.380x | 332.70 s | 148.36 | 1.118x | 0.6504 | 6,069.33 | 0.0056 | 34.51% | 2.725 |

DFlash K5 reduced the one-step generation time from 149.47 to 108.48 seconds
(1.378x) and total step time by 10.4%. Final speedups still require the matched
20-step Steps 3–20 window.
