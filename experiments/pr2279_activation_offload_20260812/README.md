# PR 2279 activation CPU-offload evidence

This experiment validates NeMo-RL PR #2279 on the official Qwen3-30B-A3B
4-node/16-GPU GB200 performance workload.

The OFF and ON arms keep the same Transformer Engine CUDA-graph path and the
same `NVTE_CPU_OFFLOAD_V1=1` environment. The only factor is
`fine_grained_activation_offloading=false/null` versus
`true/["moe_act"]`. This dependency-matched design avoids attributing CUDA-graph
speed or CPU-offload protocol changes to activation offload itself.

## Acceptance gates

- Exact PR runtime: `2f39df66d6fd0a0b1b53cf472eb1599c6c05dfce`
- Exact evidence source: pass the immutable harness commit as
  `EXPECTED_SOURCE_COMMIT`; the submitter rejects runtime changes relative to
  the PR runtime above
- Both arms complete at least three optimizer updates with finite loss/gradient
- ON logs a non-zero Megatron activation-offload summary
- Same prompt/token workload across arms
- Report warmup-excluded policy, logprob, generation, refit, E2E throughput
- Report peak GPU/host memory from job telemetry when available
- Do not claim speedup from a single sequential pair; use replicated alternating
  or concurrent pairs for claim-ready results

## Failure ledger

| Cluster | Job | Result | Root cause | Resolution |
| --- | --- | --- | --- | --- |
| Pre-Tyche | 2572885 | Harness failure | `sbatch --wrap` used `/bin/sh`; Bash `pipefail` was invalid | Changed focused unit wrapper to POSIX `set -eu` |
| Lyris | 2676543 | Harness failure | Same `/bin/sh` issue | Changed focused unit wrapper to POSIX `set -eu` |
| Lyris | 2676553 | Image failure | 2026-07-26 nightly `uv` could not resolve Python 3.13.14 | Stage a fresh immutable nightly before performance submission |

## Current jobs

- Pre-Tyche focused unit retry: `2572912`
- Lyris focused unit retry: `2676553` (failed; old image, superseded by staging)
