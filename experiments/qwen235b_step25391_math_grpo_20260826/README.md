# Qwen3-235B-A22B Base Math GRPO

This experiment uses the original NeMo-RL performance-recipe target,
`Qwen/Qwen3-235B-A22B`. All arms use the same OpenMathInstruct-2 samples, seed,
sampling settings, batch sizes, sequence packing, and 32-node GB200 allocation.

The DSpark checkpoint is the step-25391, block-size-8 drafter trained against
the matching Qwen3-235B-A22B Base target. Its architecture, weight size,
hidden/vocabulary shape, and block size are checked before submission.

## Matrix

| Arm | Method | K | CUDA Graph mode |
| --- | --- | ---: | --- |
| `baseline` | target-only | 0 | FULL_AND_PIECEWISE |
| `dspark_k3` | DSpark block 8 | 3 | FULL_AND_PIECEWISE |
| `dspark_k5` | DSpark block 8 | 5 | FULL_AND_PIECEWISE |
| `dspark_k7` | DSpark block 8 | 7 | FULL_AND_PIECEWISE |

The drafters are serving-only in this experiment: `policy.draft` is omitted,
so no online drafter update, drafter optimizer step, or drafter refit is charged
to these measurements. Every arm sets `max_num_batched_tokens=8192` and
`max_num_seqs=32`. DSpark uses `FLASH_ATTN` for the drafter and disables
FlashInfer autotuning. Global CUDA Graph mode is FAP; the DSpark draft decode
manager internally uses its supported full-decode graph path.

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
