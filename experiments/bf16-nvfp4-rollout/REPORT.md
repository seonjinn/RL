# BF16 Training with NVFP4 Rollout Results

## Scope

This campaign validates a plain BF16 Qwen3-30B-A3B Megatron trainer with real
ModelOpt NVFP4 rollout. It is not a QARL or QAT run.

- Cluster: GCP-NRT, 8 B200 GPUs per node
- Workload: OpenMathInstruct-2, 64 prompts x 32 generations, GBS 2048
- Maximum sequence length: 4096
- Importance sampling: enabled; previous-policy logprob forward retained
- Source: `14cf3a9cc1177d9303cdc214c7d10c3fe1193b10`
- Container: `nemo_rl_nightly_20260730_483099.sqsh`
- Smoke length: two GRPO steps; checkpointing disabled

## PR 3477 Stacked Validation

The BF16-to-NVFP4 implementation was also stacked on PR 3477 in the isolated
branch `sna/pr3477-bf16-nvfp4-stacked`. The validated code commit is
`fa9dd78ddbf47e91dde5959eee9aad5b546dc46d`, based on PR 3477 commit
`6f57c1b79504245fc8211028e504465045315f34`.

| Mode | Job | Source | Outcome |
| --- | ---: | --- | --- |
| BF16 train + W4A16 rollout | 498118 | `743c18d8f` | Two NCCL-Reshard refits and two GRPO steps passed |
| BF16 train + W4A4 rollout | 498209 | `fa9dd78dd` | Two NCCL-Reshard refits and two GRPO steps passed |

Commits after `743c18d8f` only harden W4A4 calibration-config provenance. The
final calibration test run passed 37 tests in job 498206; the broader stacked
refit test group passed 276 tests with four dependency-based skips in job
498194.

Logs:

- W4A16: `/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/experiments/bf16-nvfp4-rollout/results/pr3477-stacked-final4-w4a16-743c18d8f`
- W4A4: `/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/experiments/bf16-nvfp4-rollout/results/pr3477-stacked-final6-w4a4-fa9dd78dd`
- Final calibration tests: `/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/experiments/bf16-nvfp4-rollout/results/pr3477-stacked-calibration-fa9dd78dd`

## W4A16

Both paths completed two GRPO steps and passed the post-run checks for two
finite losses and two refit timing records.

| Metric, two-step mean | Legacy colocated | NCCL Reshard |
| --- | ---: | ---: |
| Transfer and update | 53.67 s | 34.37 s |
| Total refit | 63.61 s | 34.37 s |
| E2E step time | 338.22 s | 436.82 s |
| E2E throughput | 1,228.93 tok/s/GPU | 954.52 tok/s/GPU |

NCCL Reshard reduced transfer and update by 36.0% and total refit by 46.0%.
The E2E rows are not a transport-only comparison: the legacy run uses all 16
GPUs for colocated training and generation with Megatron EP16, while the NCCL
run splits the same 16 GPUs into one 8-GPU trainer node and one 8-GPU
generation node with Megatron EP8.

| Path | Job | W&B | Outcome |
| --- | ---: | --- | --- |
| Legacy colocated | 497530 | [jsru3e8a](https://wandb.ai/nvidia/sna-bf16-nvfp4-rollout/runs/jsru3e8a) | Completed, exit 0 |
| NCCL Reshard | 497532 | [jjohyjez](https://wandb.ai/nvidia/sna-bf16-nvfp4-rollout/runs/jjohyjez) | Completed, exit 0 |

## W4A4

The activation calibration artifact was generated from the exact run snapshot
with `cnn_dailymail`, 16 samples, sequence length 512, and seed 42.

- Size: 1,984,904 bytes
- SHA256: `d3b001f505c78ef9eb625aef153b791aedb380d1bb1e408324b59cf5a5405f19`
- Calibration job: 497614, completed with exit 0
- Artifact: `/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/experiments/bf16-nvfp4-rollout/RL/code_snapshots_nvfp4/14cf3a9cc-20260805-w4a4-calib1-w4a4-nccl/w4a4-nccl/artifacts/qwen3-30ba3b-w4a4.safetensors`

Both paths completed two GRPO steps and passed the post-run checks for two
finite losses and two refit timing records.

| Metric, two-step mean | Legacy colocated | NCCL Reshard |
| --- | ---: | ---: |
| Transfer and update | 75.70 s | 36.15 s |
| Total refit | 94.00 s | 36.15 s |
| E2E step time | 351.45 s | 407.40 s |
| E2E throughput | 1,180.80 tok/s/GPU | 1,027.27 tok/s/GPU |

NCCL Reshard reduced transfer and update by 52.2% and total refit by 61.5%.
The E2E rows are not a transport-only comparison: the legacy run uses all 16
GPUs for colocated training and generation with Megatron EP16, while the NCCL
run splits the same 16 GPUs into one 8-GPU trainer node and one 8-GPU
generation node with Megatron EP8. The two legacy refits were 54.91 s and
96.49 s, so a longer run is required for a stable performance estimate.

| Path | Job | W&B | Outcome |
| --- | ---: | --- | --- |
| Legacy colocated | 497642 | [z9xj4qks](https://wandb.ai/nvidia/sna-bf16-nvfp4-rollout/runs/z9xj4qks) | Completed, exit 0 |
| NCCL Reshard | 497644 | [8s4uiaje](https://wandb.ai/nvidia/sna-bf16-nvfp4-rollout/runs/8s4uiaje) | Completed, exit 0 |

## Cumulative MXFP8 Refit Context

The PR 3477 and PR 3478 effects must be separated by evidence level.

| Evidence | E2E step time | Change from legacy |
| --- | ---: | ---: |
| Strict legacy baseline | 308.45 s | - |
| Strict PR 3477 NCCL Reshard A/B | 304.13 s | -1.40% |
| PR 3477 plus PR 3478 projection | 300.87-301.00 s | -2.42% to -2.46% |
| Cross-campaign observed combined run | 294.69 s | -4.46% |

The projection amortizes PR 3478's 3.321 s saving over 17 actual refits in 18
steps. The 4.46% result is observational because it compares different source
campaigns and averaging windows.
