# Qwen3-235B MXFP8 QKVO Experiment Plan

## Question

Does extending rollout MXFP8 quantization from MoE-only weights to MoE plus
Q/K/V/O projections improve Qwen3-235B-A22B GRPO performance?

## Matrix

| Arm | Rollout precision | QKVO | Refit optimization |
| --- | --- | --- | --- |
| BF16 | BF16 | No | Off |
| MoE baseline | MXFP8 | No | Off |
| MoE optimized | MXFP8 | No | On |
| QKVO baseline | MXFP8 | Yes | Off |
| QKVO optimized | MXFP8 | Yes | On |

## Controlled Setup

- Hardware:
  - Lyris: 16 nodes x 4 GB200 GPUs
  - GCP-NRT: 8 nodes x 8 B200 GPUs
  - Both: 64 GPUs total
- Steps: 20
- Seed: 42
- GBS: 512
- Prompts per step: 16
- Generations per prompt: 32
- Maximum sequence length: 8192
- Importance sampling: enabled with the previous-policy forward retained
- Trainer parallelism: TP2, PP4, CP2, EP16
- vLLM parallelism: TP8 for BF16 and TP4 for both MXFP8 scopes
- Checkpoint saving: disabled
- Dispatcher: `alltoall`, avoiding the unavailable HybridEP runtime
- Stability guards: NVLS disabled and Ray/vLLM distributed timeouts set to 2400s

The vLLM TP difference comes from the repository's validated 235B performance
recipes. MoE-only versus QKVO comparisons remain topology-matched at TP4.

On GCP-NRT, `cluster.segment_size=8` reproduces the topology used by prior
completed 8-node Qwen3-235B jobs. GCP-NRT's Slurm does not expose
`sbatch --segment`, so only the application-side segment is set.

## Metrics

Use steps 3-20 after warmup:

- E2E, generation, logprob, policy-training, and refit time
- E2E, generation, logprob, and policy-training tokens/s/GPU
- transfer-and-update time within refit
- mean rollout reward
- token and sequence probability-product error

## Success Criteria

- All five arms complete 20 steps without checkpoint output.
- MoE-only and QKVO arms use identical topology and algorithm settings.
- QKVO performance is reported relative to MoE-only MXFP8.
- Accuracy diagnostics are reported before recommending QKVO.
