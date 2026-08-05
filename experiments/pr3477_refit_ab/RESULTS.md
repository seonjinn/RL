# PR 3477 and PR 3478 Refit Results

## Summary

PR 3477 enables BF16-training to MXFP8-rollout refit through NCCL Reshard.
The matched GCP-NRT A/B reduced transfer and weight-update time by 51.9% and
total refit time per training step by 51.8%. This reduced E2E step time by 1.4%
and increased E2E throughput by 1.4%.

PR 3478 is an independent optimization inside the receiver load path. Its
matched NCCL-Reshard A/B reduced MXFP8 MoE transfer and weight-update time by
82.6% by batching the four value-dependent expert-layout shuffles.

## PR 3477: BF16 to MXFP8 NCCL Reshard

Matched arithmetic means over steps 3-20:

| Metric | Legacy transport | NCCL Reshard | Change |
|---|---:|---:|---:|
| Transfer + update per actual refit | 8.697 s | 4.186 s | -51.9%, 2.08x faster |
| Total refit per training step | 8.214 s | 3.955 s | -51.8%, 2.08x faster |
| E2E step time | 308.450 s | 304.135 s | -1.4% |
| E2E throughput | 830.875 tok/s/GPU | 842.150 tok/s/GPU | +1.4% |

The transfer metric has 17 matched samples because neither arm performed a
weight transfer at step 11. Total refit, E2E time, and throughput have all 18
requested samples. The total-refit row includes the near-zero step-11 refit and
therefore represents the refit cost amortized over training steps.

### Setup

- Cluster: GCP-NRT, 4 nodes x 8 B200 GPUs
- Placement: 2 trainer nodes and 2 generation nodes
- Model: Qwen3-30B-A3B
- Training: BF16
- Rollout: MoE-only MXFP8; Q/K/V/O remain BF16
- Recipe: `grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml`
- Workload: 64 prompts x 32 generations, GBS 2048, 4096 max sequence length
- Importance sampling: enabled; previous-policy logprob forward retained
- Steps: 20; arithmetic means over steps 3-20
- Source: PR 3477 plus required vLLM 0.25 MXFP8 runtime fixes, commit
  `2198124bca1a83ecc22ee4526cfa7193c412723b`

### Runs

| Arm | Job | W&B | Outcome |
|---|---:|---|---|
| Legacy | 496508 | [37hdhdbt](https://wandb.ai/nvidia/sna-pr3477-refit-ab/runs/37hdhdbt) | Completed, exit 0 |
| NCCL Reshard | 496509 | [i4xg5s7k](https://wandb.ai/nvidia/sna-pr3477-refit-ab/runs/i4xg5s7k) | Completed 20 steps; post-run Ray cleanup exit 1 |

The NCCL arm recorded all 18 requested W&B samples and W&B marks the run
finished. After training and W&B finalization, Ray failed during interpreter
shutdown with `Check failed: !core_worker_process` because a core worker was
already initialized. This post-run cleanup failure does not enter the measured
step window.

## PR 3478: Batched MXFP8 MoE Shuffle

The PR-body matched A/B uses Qwen3-30B-A3B on B200 with NCCL Reshard, 20
steps, and arithmetic means over steps 3-20:

| MXFP8 MoE shuffle | Transfer + update |
|---|---:|
| Per-expert reference | 4.021 s |
| Batched | 0.700 s |
| Change | -82.6%, 5.74x faster |

PR 3478 does not claim a matched total-refit, E2E, or throughput change. Those
effects depend on refit's share of the complete step.

## Interpretation Boundary

The two PR measurements isolate different changes:

- PR 3477 compares legacy transport with receiver-quantized NCCL Reshard.
- PR 3478 compares per-expert and batched MoE layout shuffles within the NCCL
  Reshard path.

The percentages are not directly additive. The older trainer-prequantization
prototype result of 4.796 s to 0.787 s is not a measurement of the current PR
3477 implementation and should not be attributed to it.
