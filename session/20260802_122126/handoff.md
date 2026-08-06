# Handoff

## Resume From Here

The active worktree and local `main` are both at `4ed047b48`, which merges
latest NeMo-RL main with the persistent CUDA Graph bank implementation. The
next campaign is a matched Nano 20-step four-axis matrix on OCI-HSG. Existing
scope leaves are sufficient; do not add an ad-hoc launcher.

## Next Actions

- Push the exact source as `experiment/nano-cg-4axis-matrix-20260805`.
- Create or refresh a clean OCI-HSG checkout recursively at that pushed SHA.
- Dry-run, then submit baseline plus the 11 valid CUDA Graph scope subsets.
- Monitor for at least five minutes and record job IDs in `experiments.tsv`.
- Collect W&B, CUDA Graph telemetry, peak memory, and correctness metrics into the HTML report.

## Watch Outs

- Use exactly three successful CUDA Graph warmups and disable checkpoints.
- Use `batch`, not `backfill`; do not request exclusive access or partial-node GPU allocations.
- Keep source SHA, nested Bridge/MCore SHAs, container, recipe, seed, and dispatcher identical across rows.
- `moe_preprocess` alone is invalid; it must be paired with `moe_router`.
- Do not assume all-enabled is fastest. Require high cache-hit rate, no bank resets, and correctness parity.
- Do not compare stochastic reward alone as correctness evidence; include fixed-input graph/eager parity and logprob/KL diagnostics.
