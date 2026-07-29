# PR 3294 After NCCL-Reshard

This experiment measures which PR #3294 optimizations remain useful after
main commit `e40aa046e5fd4af30f93c27acdcdb9cc748670ab`.

## Comparison

| Arm | Source | PR #3294 receiver controls |
|---|---|---|
| `baseline` | Exact `e40aa046` main commit | Disabled |
| `optimized` | `sna/pr3294-after-nccl-reshard` | Batched MoE shuffle and loader replay enabled |

Both arms use non-colocated `refit_transport=nccl_reshard`. Trainer-side
prequantization, persistent IPC buffers, and slim colocated offload are disabled
because the NCCL-Reshard path does not execute them.

## Modes

- `bf16`: supported BF16 training and generation.
- `blockwise-fp8`: supported blockwise FP8 parameter storage and FP8 generation.
- `mxfp8-probe`: probes blockwise FP8 trainer storage with vLLM MXFP8 enabled.
  This is intentionally a compatibility smoke test, not a supported performance
  result.

Native Megatron `fp8_recipe=mxfp8` parameter storage is rejected by the current
NCCL-Reshard validator, which only accepts `fp8_recipe=blockwise`.

## Platforms

- GCP-NRT B200: 4 nodes x 8 GPUs, split into 2 training and 2 generation nodes.
- Lyris/OCI-HSG GB200: 5 nodes x 4 GPUs, split into 4 training and 1 generation
  node. This preserves trainer EP16 with the smallest valid allocation.

Use `MAX_STEPS=2` for smoke tests and `MAX_STEPS=20` for reported performance.
Reported steady-state values use steps 3-20.

