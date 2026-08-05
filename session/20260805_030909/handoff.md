# Handoff

## Resume From Here

The first W4A16 legacy and NCCL-Reshard GPU smokes reached vLLM initialization
but failed because a repo-relative quantization recipe was resolved from the
Ray actor working directory. A focused path-normalization fix is pending target-
container validation and resubmission.

## Next Actions

- Commit and push the path-normalization fix.
- Pull it into the prepared GCP clone and run focused container tests.
- Run `sbatch --test-only`, resubmit both W4A16 arms, and monitor for at least
  five minutes.

## Watch Outs

- Do not label prior QARL/QAT results as BF16-training rollout-only validation.
- W4A4 must use a provenance-checked calibration artifact; do not use dummy
  scales.
- Do not compare legacy versus NCCL timing until both jobs complete the same
  training-step window.
