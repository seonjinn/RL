# Qwen3-235B Thinking step-25391 Math GRPO

This experiment compares a target-only baseline with the Nemotron-v2 DFlash
and DSpark drafters trained through step 25391. All arms use the same
Qwen3-235B-A22B-Thinking-2507 target, OpenMathInstruct-2 samples, seed, sampling
settings, batch sizes, sequence packing, and 32-node GB200 allocation.

## Matrix

| Arm | Method | K | CUDA Graph mode |
| --- | --- | ---: | --- |
| `baseline` | target-only | 0 | PIECEWISE |
| `dflash_k3` | DFlash | 3 | PIECEWISE |
| `dflash_k5` | DFlash | 5 | PIECEWISE |
| `dspark_k3` | DSpark | 3 | FULL_DECODE_ONLY |
| `dspark_k5` | DSpark | 5 | FULL_DECODE_ONLY |

The drafters are serving-only in this experiment: `policy.draft` is omitted,
so no online drafter update, drafter optimizer step, or drafter refit is charged
to these measurements. Every arm sets `max_num_batched_tokens=8192` and
`max_num_seqs=32`. DSpark uses `FLASH_ATTN` for the drafter and disables
FlashInfer autotuning.

The default pilot is three GRPO steps. Set `Q235_MAX_STEPS=1` for the
correctness smoke or `Q235_MAX_STEPS=20` for the measurement run. The launcher
accepts only 1, 3, or 20 steps.

## Submission protocol

Run from the immutable remote harness checkout. A passing `sbatch --test-only`
receipt is mandatory before an actual submission of the same arm, source SHA,
config SHA, harness SHA, and step count.

```bash
bash experiments/qwen235b_step25391_math_grpo_20260826/submit_qwen235b_math_grpo.sh \
  --emit-manifest dflash_k3

Q235_MAX_STEPS=3 \
bash experiments/qwen235b_step25391_math_grpo_20260826/submit_qwen235b_math_grpo.sh \
  --test-only dflash_k3

Q235_MAX_STEPS=3 \
bash experiments/qwen235b_step25391_math_grpo_20260826/submit_qwen235b_math_grpo.sh \
  --submit dflash_k3
```

W&B project: `sna-specdec`.
