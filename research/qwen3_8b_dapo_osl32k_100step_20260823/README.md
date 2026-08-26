# Qwen3-8B DAPO OSL32K 100-Step Experiment

This experiment measures end-to-end long-context training performance for three matched arms:

- `baseline-k0`: no speculative decoding
- `dflash-k5`: DFlash with five speculative tokens and always update/refit
- `dspark-k5`: DSpark with five speculative tokens and always update/refit

All arms use the immutable first 64 rows of DAPO-Math-17k, Qwen3-8B, 2K maximum input, 32K maximum output, TP2/PP1/CP1/DP2, global batch size 8, and one four-GPU GB200 node. This directory is separate from the 13-arm cadence screen; it does not add fixed or adaptive cadence arms.

## Segment contract

The optimizer schedule and training horizon remain global: `grpo.max_num_steps=100` and `grpo.max_num_epochs=4`. Four jobs set only the absolute `grpo.segment_stop_step` boundary to 25, 50, 75, and 100. The first three boundaries produce complete resumable checkpoints without closing terminal cadence artifacts. Step 100 produces `checkpoint-runtime.json` and `schedule-runtime.json`.

Every resume verifies the prior keyed segment receipt and rehashes the checkpoint tree, decision-ledger prefix, dataloader state, and optimizer state. The four jobs share one deterministic W&B run ID; segments 50, 75, and 100 require `WANDB_RESUME=must`. SLURM dependencies use `afterok`.

## Local verification

```bash
uv run --no-project python -m unittest \
  research.qwen3_8b_dapo_osl32k_100step_20260823.tests.test_contract -v
bash -n research/qwen3_8b_dapo_osl32k_100step_20260823/run_segment.sh
```

## Submission protocol

Use the full reviewed product and harness SHAs. Test-only scheduling must succeed before actual submission for each arm.

```bash
uv run --no-project python \
  research/qwen3_8b_dapo_osl32k_100step_20260823/harness.py submit \
  --arm baseline-k0 \
  --output-root /lustre/.../qwen3_8b_dapo_osl32k_100step_20260823 \
  --product-sha a28df91a94b623f5108a2992ccac887cc8cbdaab \
  --harness-sha HARNESS_FULL_SHA \
  --test-only
```

After checking the immutable test-only receipt, replace `--test-only` with `--actual`. Repeat for `dflash-k5` and `dspark-k5`. Actual submission is exactly once per arm/product/harness/config identity; a partial scheduler failure retains the lock for manual reconciliation.

No job is submitted by repository tests or rendering commands.
