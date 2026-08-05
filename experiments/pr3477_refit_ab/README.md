# PR 3477 Refit Performance A/B

This experiment measures the current PR 3477 receiver-quant implementation on
GCP-NRT. It compares the legacy packed-collective refit with NCCL-Reshard while
holding the source commit, Qwen3-30B-A3B MoE-only MXFP8 recipe, topology, batch,
seed, and step window constant.

## Setup

- Cluster: GCP-NRT, 4 nodes x 8 B200 GPUs
- Placement: 2 trainer nodes and 2 generation nodes
- Training storage: BF16
- Rollout storage: MoE-only MXFP8; Q/K/V/O remain BF16
- Recipe: `grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml`
- Steps: 20; report arithmetic means over steps 3-20
- Batch: 64 prompts x 32 generations, GBS 2048
- Sequence length: 4096
- Importance sampling: enabled with the previous-policy logprob forward retained
- Checkpointing and validation: disabled

## Arms

| Arm | `policy.generation.refit_transport` |
|---|---|
| `legacy` | `null` |
| `nccl` | `nccl_reshard` |

Run the scheduler preflight first, then submit both arms:

```bash
ACTION=test-only ./experiments/pr3477_refit_ab/submit_pair.sh
ACTION=submit ./experiments/pr3477_refit_ab/submit_pair.sh
```

The primary metric is
`timing/train/prepare_for_generation/transfer_and_update_weights`. Also
report total refit, E2E step time, E2E throughput, mean rollout reward, and
generation KL over the same step window.
