# NeMo-RL SpecDec Slowdown Watchlist

Updated: 2026-06-24 11:12 PDT

This file records SpecDec runs that are slower than their matched baseline. Comparisons use step>=2 averages when available and only count rows as `explicit` when the run manifest/log clearly identifies `policy.generation.vllm_cfg.enforce_eager`.

## Confirmed `enforce_eager=false` Slowdowns

| Model | Mode | Method | Job | Baseline | Steps | E2E step speedup | Gen time speedup | E2E throughput speedup | Gen throughput speedup | Acceptance | Mean accept len | W&B |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-30B-A3B | sync | PARD K16 | 2182146 | 2182145 | 19/20 | 0.57x | 0.36x | 0.57x | 0.36x | 12.6% | 2.90 | [run](https://wandb.ai/nvidia/nemo-rl-perfcfg-specdec-lyris/runs/5yg0y4re) |
| Qwen3-30B-A3B | sync | Suffix K32 | 2182148 | 2182145 | 19/20 | 0.76x | 0.61x | 0.75x | 0.60x | 22.2% | 2.31 | [run](https://wandb.ai/nvidia/nemo-rl-perfcfg-specdec-lyris/runs/egbzz2wt) |
| Qwen3-32B | sync | PARD K8 | 2196842 | 2175017 | 19/20 | 0.91x | 0.84x | 0.90x | 0.84x | 27.1% | 3.16 | [run](https://wandb.ai/nvidia/nemo-rl-perfcfg-specdec-lyris/runs/ji0n20hg) |
| Qwen3-32B | sync | PARD K12 | 2196844 | 2175017 | 7/20 partial | 0.90x | 0.83x | 0.90x | 0.83x | 17.9% | 3.13 | [run](https://wandb.ai/nvidia/nemo-rl-perfcfg-specdec-lyris/runs/g2knlfxu) |
| Qwen3-32B | sync | PARD K16 | 2196588 | 2175017 | 19/20 | 0.87x | 0.82x | 0.85x | 0.80x | 12.5% | 2.99 | [run](https://wandb.ai/nvidia/nemo-rl-perfcfg-specdec-lyris/runs/nw0hirdp) |

Primary action: treat PARD as the highest-priority performance bug. The main symptom is that accepted length is nonzero, but generation throughput is still below baseline, so the overhead is likely in the PARD/vLLM integration path rather than only in model quality.

## Current Code Follow-Up: Qwen3-32B PARD K1

These runs are still live, but step>=2 metrics are now available, so cold-start step 1 is excluded. Baseline is Qwen3-32B sync job `2175017`, step>=2: E2E `539.87s`, generation `335.03s`, E2E throughput `779.28 tok/s/GPU`, generation throughput `1256.62 tok/s/GPU`.

| Variant | Job | Completed steps used | E2E step speedup | Gen time speedup | E2E throughput speedup | Gen throughput speedup | Acceptance | Mean accept len | W&B |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| PARD K1, target TP2 / draft TP2 | 2199223 | step 2-4 | 1.07x | 1.11x | 1.05x | 1.09x | 75.5% | 1.75 | [run](https://wandb.ai/nvidia/nemo-rl-perfcfg-specdec-lyris/runs/uc0y3ksb) |
| PARD K1, target TP1 / draft TP1 override | 2199224 | step 2 only | 1.42x | 1.80x | 1.36x | 1.71x | 75.8% | 1.75 | [run](https://wandb.ai/nvidia/nemo-rl-perfcfg-specdec-lyris/runs/9an1nij1) |

Interpretation so far: PARD K1 improved materially versus the K8/K12/K16 slowdown rows. This points away from a pure CUDA graph regression and toward K-dependent PARD overhead, parallel draft scheduling overhead, or draft TP/target TP interaction. The TP1/draftTP1 result is especially promising but needs more completed steps before it should replace the baseline K-sweep conclusion.

## Confirmed `enforce_eager=true` Slowdowns

No matched-baseline `enforce_eager=true` slowdown is confirmed in the current parsed data.

The explicit true K-sweep rows in `docs/lyris_nemorl_perfcfg_specdec_combined_latest.csv` are all above baseline:

| Model | Mode | Method | Job | E2E throughput speedup | Gen throughput speedup |
|---|---|---:|---:|---:|---:|
| Qwen3-30B-A3B | sync | Eagle-3 K9 | 2177875 | 1.46x | 2.01x |
| Qwen3-30B-A3B | async-1off | Eagle-3 K9 | 2177876 | 1.30x | 1.31x |
| Qwen3-32B | sync | Eagle-3 K5 | 2177869 | 1.33x | 1.72x |
| Qwen3-32B | sync | Eagle-3 K7 | 2177873 | 1.29x | 1.53x |
| Qwen3-32B | sync | Eagle-3 K9 | 2177877 | 1.18x | 1.36x |

Caveat: the later Qwen3-32B `cudagraphoff` sync baseline job `2191508` failed, so jobs `2191509/2191510/2191511` cannot be used as a matched-baseline comparison until that baseline is rerun.

## Historical Rows Needing Reclassification

These are slower-than-baseline rows from older summaries where `enforce_eager` was not recorded in the summary row. They should remain on the debug list, but should not be used as true/false evidence until the original logs are classified.

| Model | Mode | Method | Job | E2E step speedup | Gen time speedup | E2E throughput speedup | Gen throughput speedup | Source |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3-30B-A3B | sync | PARD K5 | 2175012 | 0.84x | 0.77x | 0.84x | 0.78x | `docs/lyris_nemorl_qwen30_qwen32_pr2879_step20_speedups_20260622.csv` |
| Qwen3-30B-A3B | async-1off | PARD K5 | 2175016 | 0.95x | n/a | 0.95x | 0.95x | same |
| Qwen3-30B-A3B | async-1off | Suffix K32 | 2175014 | 0.92x | n/a | 0.91x | 0.91x | same |
| Qwen3-32B | sync | PARD K5 | 2175020 | 0.96x | 0.91x | 0.96x | 0.91x | same |
| Qwen3-32B | sync | Suffix K32 | 2175018 | 0.72x | 0.60x | 0.73x | 0.60x | same |
| Qwen3-32B | async-1off | PARD K5 | 2175024 | 0.93x | n/a | 0.95x | 0.95x | same |
| Qwen3-32B | async-1off | Suffix K32 | 2175022 | 0.46x | n/a | 0.45x | 0.45x | same |

## Debug Notes

- PARD K8/K12/K16 on Qwen3-32B gets mean accepted length around 3, but still loses generation throughput. That points to implementation overhead, lack of an optimized fast path, CUDA graph capture coverage, or extra synchronization rather than only acceptance quality.
- Qwen3-30B-A3B PARD K16 is much worse than baseline with only 12.6% token acceptance; this combines low acceptance with overhead and should be treated separately from the Qwen3-32B case.
- Suffix slowdowns should be tracked separately from PARD. They may be scheduler/cache/speculative bookkeeping overhead rather than drafter-model overhead.
- Eagle-3 is currently the control path: both explicit true and explicit false Eagle rows are the ones that are most consistently faster than baseline.
- CUDA graph hypothesis: both effects can be true at once. `enforce_eager=false` makes the baseline stronger through vLLM compile/CUDAGraph capture, while PARD may not receive the same benefit if the generic `draft_model` plus `parallel_drafting=true` path introduces graph breaks, extra synchronization, or non-trivial draft/verify scheduling overhead.

CSV companion: `docs/nemorl_specdec_slowdown_watchlist_20260624.csv`.
