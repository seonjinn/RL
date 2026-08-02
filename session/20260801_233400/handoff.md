# Handoff

## Resume From Here

Jobs `486954` and `486955` completed the matched 20-step legacy versus transform-aware NCCL-Reshard A/B on GCP-NRT with exit code `0:0`. Steps 3-20 show a refit reduction from `4.7956 s` to `0.7867 s` and an E2E-throughput gain of 4.43%. The NCCL arm is a Python exact-transfer fallback over NCCL communicators because the image has no native `nccl.m2n` package.

## Next Actions

- Commit and push `RESULTS.md` and the session records.
- Package an ABI-compatible `nccl.m2n` wrapper/library before a native-M2N follow-up benchmark.

## Watch Outs

- Do not describe this as compiled/native M2N performance.
- Both arms already use trainer-side prequantization; the A/B isolates transport.
- Preserve the PR 3294 legacy branch and experiment results separately.
