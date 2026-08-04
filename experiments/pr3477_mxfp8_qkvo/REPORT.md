# Qwen3-30B-A3B MXFP8 QKVO Analysis

## Conclusion

QKVO refit processing is already small. Quantizing QKV and O projections adds
96 dense MXFP8 modules, but increases dense scale-swizzle GPU time by only
2.71 ms per generation worker. The complete refit increases by 0.26 s per
step, which is 0.09% of the 297.96 s QKVO E2E step time.

The QKVO recipe does not improve generation performance in the current path.
It is 1.35% slower in generation time and 1.33% lower in generation throughput
than the MoE-only recipe. The main actionable issue is not scale swizzling:
NeMo-RL replaces vLLM 0.25's preferred CuTeDSL MXFP8 linear kernel with the
CUTLASS kernel because refit keeps dense weights in `[N, K]` layout.

## Setup

- Cluster: GCP-NRT, four B200 nodes, eight GPUs per node
- Model: `Qwen/Qwen3-30B-A3B`
- Training GPUs: 16
- Generation GPUs: 16
- Generation TP/PP/EP: 1/1/1
- Workload: 64 prompts, 32 generations per prompt, GBS 2048
- Importance sampling: enabled; previous-policy logprobs retained
- Refit transport: NCCL Reshard, with the Python xferdtensor fallback
- Steps: five; step 1 treated as warmup and steps 2-5 averaged
- Container: `nemo_rl_nightly_20260730_483099.sqsh`

Recipes:

- MoE only: `grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml`
- MoE + QKVO: `grpo-qwen3-30ba3b-4n4g-mxfp8-qkvo-rollout.yaml`

The QKVO overlay sets `quantization_ignored_layer_kws: []`. The MoE-only
recipe excludes `q_proj`, `k_proj`, `v_proj`, and `o_proj`.

## Stable Results

Steps 2-5, mean values:

| Metric | MoE only | MoE + QKVO | QKVO delta |
|---|---:|---:|---:|
| E2E step time | 294.69 s | 297.96 s | +1.11% |
| Generation time | 50.32 s | 50.99 s | +1.35% |
| Logprob time | 115.52 s | 116.56 s | +0.90% |
| Policy training time | 126.35 s | 127.74 s | +1.10% |
| Refit transfer/update | 0.91 s | 1.17 s | +0.26 s |
| E2E throughput | 869.45 tok/s/GPU | 859.93 tok/s/GPU | -1.09% |
| Generation throughput | 10,184.96 tok/s/GPU | 10,049.67 tok/s/GPU | -1.33% |

Only generation and refit configuration changed. The approximately 1 s
logprob and training differences are run-to-run noise or token-shape effects,
not work introduced by QKVO refit.

## Dense Refit Breakdown

| Metric | MoE only | MoE + QKVO | Increment |
|---|---:|---:|---:|
| Dense modules processed | 49 | 145 | +96 |
| Raw E8M0 scale data | 9.65 MiB | 36.65 MiB | +27.00 MiB |
| Scale-swizzle GPU time | 4.80 ms | 7.52 ms | +2.71 ms |
| Scale-swizzle CPU submit time | 4.83 ms | 13.27 ms | +8.45 ms |
| Misc receive/load critical time | 0.38 s | 0.42 s | +0.04 s |

The added shapes are exactly two fused dense projections per transformer
layer across 48 layers:

- Fused QKV: 48 tensors of shape `[5120, 2048]`
- O projection: 48 tensors of shape `[2048, 4096]`

The current flow sends these weights through the BF16 misc broadcast. Each
generation worker quantizes them to MXFP8, then swizzles the E8M0 scales. The
layout operation itself is not a useful optimization target: eliminating all
of the added scale-swizzle GPU time would save less than 0.001% of E2E time.

## Recommended Direction

1. Keep the CuTeDSL MXFP8 dense kernel active during refit. Adapt the refit
   loader to update stable CuTeDSL-compatible `[K, N]` storage, or write the
   canonical `[N, K]` update into persistent staging and transform directly
   into the live parameter.
2. Benchmark CuTeDSL versus CUTLASS for the observed QKV/O shapes and rollout
   token-count distribution before changing the loader contract.
3. Consider moving QKV/O from misc broadcast into the bulk reshard path only
   after kernel-layout support. The measured upper bound from removing all
   extra QKVO refit work is only 0.26 s per step.

Batching the dense scale swizzles is not recommended on these results. The
additional implementation and correctness surface would target a 2.71 ms GPU
operation while leaving the observed generation regression unchanged.

## Reproduction

Branch: `sna/pr3478-qkvo-analysis`

```bash
MODE=moe-only ACTION=submit MAX_STEPS=5 WANDB_ENABLED=false \
  experiments/pr3477_mxfp8_qkvo/submit_gcp_nrt.sh

MODE=moe-qkvo ACTION=submit MAX_STEPS=5 WANDB_ENABLED=false \
  experiments/pr3477_mxfp8_qkvo/submit_gcp_nrt.sh
```

Jobs and remote logs:

- MoE only, job 492816:
  `/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/experiments/pr3478-qkvo-analysis/results/moe-only-5step-20260803-ab-moe-only/492816-logs/ray-driver.log`
- MoE + QKVO, job 492831:
  `/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/experiments/pr3478-qkvo-analysis/results/moe-qkvo-5step-20260803-ab2-moe-qkvo/492831-logs/ray-driver.log`

Both runs reached all five steps. SLURM reported `FAILED` only after completion
because Ray attempted to initialize a second CoreWorker during interpreter
teardown (`Check failed: !core_worker_process`).
