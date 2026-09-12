# Speculative Decoding Technical Blog Figures

## Figure production plan

| Figure | Purpose | Production method | State |
|---|---|---|---|
| 1 | Compare EAGLE-3, DFlash, and DSpark candidate generation before shared target verification | Matplotlib publication PNG; Archify interactive companion | Publication draft complete |
| 2 | Show target-only RL step-time share and Amdahl-law opportunity | Publication bar/curve chart from canonical timer export | Waiting for matched baseline data |
| 3 | Show acceptance drift and actual online-update/refit events | Time-series plot from canonical W&B/CSV export | Waiting for completed cadence study |
| 4 | Show packed-token, drafter-loss, distributed update, and refit flow | Archify workflow/data-flow diagram | Design pending release interface freeze |
| 5 | Compare matched generation TPS/GPU and speedup | Publication bar chart | Waiting for release cohort |
| 6 | Compare end-to-end GRPO time breakdown | Stacked bar chart | Waiting for non-overlapping timers |
| 7 | Show cadence freshness versus update overhead | Scatter/efficient-frontier chart | Waiting for complete matched windows |
| 8 | Show speedup by generated-length bin | Grouped bar chart | Waiting for DAPO/long-context results |

## Figure 1 artifacts

- `render_specdec_method_comparison.py`: deterministic Matplotlib renderer with text-boundary auditing.
- `specdec_eagle3_dflash_dspark_compact.png`: current compact 1480 × 830 publication image.
- `specdec_eagle3_dflash_dspark_compact.layout.json`: render receipt confirming exact dimensions and zero detected text overflow.
- `specdec_eagle3_dflash_dspark_concepts.dataflow.json`: editable Archify source.
- `specdec_eagle3_dflash_dspark_concepts.html`: self-contained interactive artifact with light/dark modes and SVG/PNG export.
- `specdec_eagle3_dflash_dspark_concepts.png`: superseded Archify-derived PNG retained for comparison.
- `specdec_eagle3_dflash_dspark_concepts.visual-check.*`: automated viewport evidence used for visual review.

Figure 1 teaches one invariant: all three methods use target verification as the correctness boundary. Their primary difference is the proposal path—autoregressive depth for EAGLE-3, one-pass block prediction for DFlash, and a parallel block with a lightweight predecessor/confidence path for DSpark.

The conceptual details are grounded in the original [EAGLE-3](https://arxiv.org/abs/2503.01840), [DFlash](https://arxiv.org/abs/2602.06036), and [DSpark](https://arxiv.org/abs/2607.05147) papers and cross-checked against the [vLLM parallel-drafting overview](https://vllm.ai/blog/2026-07-28-speculators-parallel-drafting). The publication labels use the paper terminology: autoregressive, parallel block-diffusion, and semi-autoregressive.

Do not create performance charts from placeholder values. Each publication chart must use a canonical export, a declared averaging window, a matched target-only baseline, and quality checks.
