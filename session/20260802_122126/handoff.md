# Handoff

## Resume From Here

MCore candidate `2dbad0a2d` merges current main and retains the HybridEP/TE
capture fix. Bridge `2f6338610` merges current Bridge main and pins that MCore
candidate; both are pushed. NeMo-RL integration commit `4e5f9bac7` pins that
Bridge candidate. ptyche job `2551742` passed the exact 16-GPU gate on the
immediately preceding candidate `fc718cf4c`, so the merged candidate must repeat
the gate. The next step is to refresh ptyche, re-attest, and re-run. Do not
submit NeMo-RL through the existing
`MCORE_CANDIDATE_SHA` path alone: `run_nemorl_scope.sub` currently clears
`PYTHONPATH`, so it would silently run integration MCore `4013232a9`.

## Next Actions

- Re-run exact ptyche MCore correctness and refresh source/runtime attestation.
- Submit matched 20-step Nano baseline and graph rows, then collect W&B and
  CUDA Graph telemetry into the HTML report.

## Watch Outs

- Use exactly three successful CUDA Graph warmups and disable checkpoints.
- Use `batch`, not `backfill`; use whole ptyche GB200 nodes without explicit GRES flags.
- Keep source SHA, nested Bridge/MCore SHAs, container, recipe, seed, and dispatcher identical across rows.
- `moe_preprocess` alone is invalid; pair it with `moe_router`.
- Do not infer convergence from 20 steps; require deterministic graph/eager parity and longer matched soak evidence.
- Keep exact candidate provenance in nested gitlinks; do not rely on an unused snapshot variable.
