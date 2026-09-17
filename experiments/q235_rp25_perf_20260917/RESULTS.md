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

The baseline intentionally preserves the official performance recipe without a
`max_num_seqs` override. The SpecDec arms use the S64 cap required by their
captured request-width buckets. Therefore, the primary table is an optimized
method comparison rather than a one-variable ablation. This distinction must
remain visible in any published result.

## Performance

| Configuration | Job | W&B | Gen TPS/GPU | Gen speedup | Generation time | E2E step time | E2E speedup | Acceptance | Mean accepted length |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline, no SpecDec | 3086570 | pending | — | 1.000x | — | — | 1.000x | — | — |
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
