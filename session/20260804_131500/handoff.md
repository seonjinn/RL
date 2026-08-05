# Handoff

## Resume From Here

The matched PR 3477 A/B is complete. Results and interpretation boundaries are
recorded in `experiments/pr3477_refit_ab/RESULTS.md`.

## Next Actions

- Decide whether the two required vLLM 0.25 runtime fixes should be added to the
  PR 3477 branch.
- Investigate the non-blocking Ray interpreter-shutdown failure if clean SLURM
  completion is required for future NCCL runs.

## Watch Outs

- Do not attribute the historical trainer-prequant result to current PR 3477.
- Keep PR 3478's transfer/update effect separate from total-refit and E2E claims.
