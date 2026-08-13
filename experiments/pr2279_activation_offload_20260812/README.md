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
- Current evidence source: `42eb7c0f742ebb3006a880f8fab807bff29d81fe`
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
| Pre-Tyche | 2572912 | Setup timeout | TE source build targeted seven CUDA architectures and exceeded the 30-minute job limit before tests started | Reused the build cache and resubmitted with a one-hour limit as 2573195 |
| Lyris | 2676585 | Passed | Fresh nightly image staged with immutable metadata and SHA256 | Image SHA256 `aa7621512376f5562a950708625f0db011f2a240d4fc1a612083489df4d8e8ab` |
| Lyris | 2676732 | Smoke harness failure | The base image intentionally lacks the mcore-extra TE Python package; the smoke asserted too much | Validate base image uv/Python/torch/ray/CUDA only; TE is built in the source-pinned mcore venv |
| Lyris | 2676760 | Passed | Fresh image recognized 4 GB200 GPUs with torch 2.11.0+cu130, CUDA 13.0, Ray 2.56.1 | Use the immutable image for focused unit and performance jobs |
| GitHub CI | run 31650398312 | Infrastructure failure | Six functional shards hit the same flash-attn wheel HTTP/2 `refused stream` fetch error | No source change; rerun the external-network failures after approval |

## Current jobs

- Pre-Tyche focused unit retry: `2573195` (running; cached TE build)
- Lyris focused unit: `2676792` (running; `NVTE_CUDA_ARCHS=100`)
