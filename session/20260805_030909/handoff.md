# Handoff

## Resume From Here

The BF16-training/NVFP4-rollout implementation and GCP launcher are ready, but
no GPU smoke has run yet. Start with W4A16 legacy and NCCL-Reshard on GCP-NRT.

## Next Actions

- Pull the new launcher commit in the prepared GCP clone.
- Run focused container tests and `sbatch --test-only` before submission.
- Monitor each submitted job for at least five minutes.

## Watch Outs

- Do not label prior QARL/QAT results as BF16-training rollout-only validation.
- W4A4 must use a provenance-checked calibration artifact; do not use dummy
  scales.
