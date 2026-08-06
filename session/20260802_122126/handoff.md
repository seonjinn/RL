# Handoff

## Resume From Here

The corrected Nano 20-step four-axis matrix is submitted on OCI-HSG from exact
source `e95e40325`. The campaign manifest is under
`/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/sna-cg-study/nemotron-thd-te-graph-20260805/runs/nano-4axis-matrix/`.
The original 12 jobs failed before NeMo-RL because the stale 2026-08-01 image
could not supply Python 3.13.14. The corrected jobs explicitly use the known-good
2026-08-05 nightly ending in runtime job `5884993`. All corrected rows loaded
the config and initialized generation workers without fatal errors.

## Next Actions

- Monitor corrected jobs `5913139`, `5913180`, `5913182`, `5913184`, `5913186`,
  `5913188`, `5913190`, `5913192`, `5913194`, `5913196`, `5913198`, and
  `5913200` through policy initialization, first optimizer step, and completion.
- Collect W&B, CUDA Graph telemetry, peak memory, and correctness metrics into the HTML report.

## Watch Outs

- Use exactly three successful CUDA Graph warmups and disable checkpoints.
- Use `batch`, not `backfill`; do not request exclusive access or partial-node GPU allocations.
- Keep source SHA, nested Bridge/MCore SHAs, container, recipe, seed, and dispatcher identical across rows.
- `moe_preprocess` alone is invalid; it must be paired with `moe_router`.
- Do not assume all-enabled is fastest. Require high cache-hit rate, no bank resets, and correctness parity.
- Do not compare stochastic reward alone as correctness evidence; include fixed-input graph/eager parity and logprob/KL diagnostics.
- Do not use the direct launcher's stale default container; pass the `5884993` nightly explicitly.
