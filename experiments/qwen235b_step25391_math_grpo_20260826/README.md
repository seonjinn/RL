# Qwen3-235B-A22B Base Math GRPO

This experiment directly inherits the official NeMo-RL GB200 performance recipe,
`grpo-qwen3-235b-32n4g.yaml`, whose target is `Qwen/Qwen3-235B-A22B`.
The benchmark overrides only the 20-step run length and DSpark-specific settings.

The DSpark checkpoint is the step-25391, block-size-8 drafter trained against
the matching Qwen3-235B-A22B Base target. Its architecture, weight size,
hidden/vocabulary shape, and block size are checked before submission.

## Matrix

| Arm | Method | K | Base recipe |
| --- | --- | ---: | --- |
| `baseline` | target-only | 0 | official 32n4g |
| `dspark_k3` | DSpark block 8 | 3 | official 32n4g |
| `dspark_k5` | DSpark block 8 | 5 | official 32n4g |
| `dspark_k7` | DSpark block 8 | 7 | official 32n4g |

The drafters are serving-only in this experiment: `policy.draft` is omitted,
so no online drafter update, drafter optimizer step, or drafter refit is charged
to these measurements. DSpark adds only its checkpoint, K, draft TP,
`FLASH_ATTN`, disabled FlashInfer autotuning, and K-aware CUDA Graph capture
sizes. Target, data, batching, sequence length, parallelism, validation,
checkpointing, and the global CUDA Graph mode remain owned by the official
performance recipe.

The clean product checkout is pinned to `d5c8bfa987025949699f7cfff188b349480bb8b5`.
Megatron source, vLLM overlays, venvs, and caches are staged under node-local
`/raid/scratch`; durable configs, logs, and receipts remain on Lustre. DSpark
uses the source-verified vLLM #48167 runtime patch plus the group-causality
follow-up that is first validated by the Q30B canary.

The default measurement is 20 GRPO steps. `Q235_MAX_STEPS=1` or `3` remains
available for correctness canaries. The launcher accepts only 1, 3, or 20.

## Submission protocol

Run from the immutable remote harness checkout. A passing `sbatch --test-only`
receipt is mandatory before an actual submission of the same arm, source SHA,
config SHA, harness SHA, and step count.

```bash
bash experiments/qwen235b_step25391_math_grpo_20260826/submit_qwen235b_math_grpo.sh \
  --emit-manifest dspark_k3

Q235_MAX_STEPS=20 \
bash experiments/qwen235b_step25391_math_grpo_20260826/submit_qwen235b_math_grpo.sh \
  --test-only dspark_k3

Q235_MAX_STEPS=20 \
bash experiments/qwen235b_step25391_math_grpo_20260826/submit_qwen235b_math_grpo.sh \
  --submit dspark_k3
```

W&B project: `sna-specdec`.
