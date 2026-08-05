# Handoff

## Resume From Here

The BF16-training/NVFP4-rollout implementation is committed and pushed, but no
GPU smoke has run yet. Start with W4A16 legacy and NCCL-Reshard on GCP-NRT.

## Next Actions

- Run focused local tests and commit this session checkpoint.
- Prepare the remote branch and run `sbatch --test-only` before submission.
- Monitor each submitted job for at least five minutes.

## Watch Outs

- Do not label prior QARL/QAT results as BF16-training rollout-only validation.
- W4A4 must use a provenance-checked calibration artifact; do not use dummy
  scales.
