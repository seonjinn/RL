# Handoff

## Resume From Here

Jobs `486954` and `486955` completed the matched 20-step legacy versus transform-aware NCCL-Reshard A/B on GCP-NRT with exit code `0:0`. Steps 3-20 show a refit reduction from `4.7956 s` to `0.7867 s` and an E2E-throughput gain of 4.43%. Jobs `487298` and `487299` then isolated the receiver-side PR 3294 work on top of NCCL: refit improved from `4.138 s` to `0.887 s`, and throughput improved by 2.21%. The NCCL arm is a Python exact-transfer fallback over NCCL communicators because the image has no native `nccl.m2n` package.

## Next Actions

- Package an ABI-compatible `nccl.m2n` wrapper/library before a native-M2N follow-up benchmark.
- Design a generic storage-transform plan and codec registry in a separate PR after first hardening unsupported storage-pair validation.

## Watch Outs

- Do not describe this as compiled/native M2N performance.
- Both arms already use trainer-side prequantization; the A/B isolates transport.
- The post-NCCL receiver A/B also holds prequantization and transport constant; it isolates batched MoE shuffle and loader-route caching.
- Preserve the PR 3294 legacy branch and experiment results separately.
- Do not claim arbitrary cross-precision support from the current MXFP8-specific implementation.
