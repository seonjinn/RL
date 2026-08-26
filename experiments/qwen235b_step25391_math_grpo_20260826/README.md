# Qwen3-235B-A22B Base Math GRPO

This experiment uses the original NeMo-RL performance-recipe target,
`Qwen/Qwen3-235B-A22B`. All arms use the same OpenMathInstruct-2 samples, seed,
sampling settings, batch sizes, sequence packing, and 32-node GB200 allocation.

The available step-25391 DFlash and DSpark checkpoints were trained against
`Qwen3-235B-A22B-Thinking-2507`; they are not valid Base-target drafters. The
launcher therefore allows the target-only baseline and fails closed for the
four SpecDec arms until matching Base-target checkpoints are supplied.

## Matrix

| Arm | Method | K | CUDA Graph mode |
| --- | --- | ---: | --- |
| `baseline` | target-only | 0 | PIECEWISE |
| `dflash_k3` | DFlash | 3 | PIECEWISE; blocked pending Base drafter |
| `dflash_k5` | DFlash | 5 | PIECEWISE; blocked pending Base drafter |
| `dspark_k3` | DSpark | 3 | FULL_DECODE_ONLY; blocked pending Base drafter |
| `dspark_k5` | DSpark | 5 | FULL_DECODE_ONLY; blocked pending Base drafter |

The drafters are serving-only in this experiment: `policy.draft` is omitted,
so no online drafter update, drafter optimizer step, or drafter refit is charged
to these measurements. Every arm sets `max_num_batched_tokens=8192` and
`max_num_seqs=32`. DSpark uses `FLASH_ATTN` for the drafter and disables
FlashInfer autotuning.

The product checkout contains the prebuilt ARM64 Megatron dataset extension
`megatron/core/datasets/helpers_cpp`. The launcher permits exactly that one
generated file only when its SHA-256 is
`39f37692b828622d8e40d13a683b5d0f511c7c852c7497edce286c7eda28833a`;
any other source or submodule change fails closed.

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
