# BF16 to MXFP8 NCCL-Reshard A/B

This experiment measures trainer-side MXFP8 prequantization over NCCL-Reshard
on the same source commit, container, model, topology, batch, and seed.

## Comparison

| Mode | Trainer storage | Rollout storage | Transport |
|---|---|---|---|
| `bf16` | BF16 | BF16 | NCCL-Reshard |
| `mxfp8-rollout` | BF16 | MXFP8 | Legacy collective |
| `mxfp8-nccl-prequant` | BF16 | MXFP8 | NCCL-Reshard value + E8M0 scale pair |

All three modes run sequentially in one allocation. The first functional target
is GCP-NRT B200: 4 nodes x 8 GPUs, split into 2 training and 2 generation nodes.
Qwen3-30B-A3B uses trainer EP16 and vLLM TP1. Q/K/V/O projections stay BF16;
MXFP8 applies to eligible MoE weights. Importance-sampling correction is
disabled in all modes and `force_on_policy_ratio=true`.

The isolated transport comparison is `mxfp8-rollout` versus
`mxfp8-nccl-prequant`; both use the same MXFP8 recipe and generation backend.
`bf16` is an end-to-end BF16 reference and may use a different MoE backend.

Use `MAX_STEPS=5` for functional smoke tests and `MAX_STEPS=20` for the reported
A/B. Report `transfer_and_update_weights`, total refit, generation, E2E step
time, and logged tokens/s/GPU over steps 3-20.

For the BF16 versus MXFP8 NCCL prequantization A/B on GCP-NRT:

```bash
CONTAINER=/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/containers/pr3294-nccl-reshard/nemo_rl_nightly_20260727_14418344.sqsh \
ACTION=test-only \
MAX_STEPS=5 \
./experiments/nccl_reshard_pr3294/submit_prequant_ab.sh
```
