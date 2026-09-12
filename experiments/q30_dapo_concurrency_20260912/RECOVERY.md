# DAPO concurrency recovery: validation configuration

## Root cause

The initial launcher set `data.validation=null` and
`data.train.split_validation_size=0.0`, but inherited `grpo.val_period=10`
from `examples/configs/grpo_math_1B.yaml`. GRPO setup requires a validation
dataset if periodic, initial, or final validation is enabled, so all seven
launched gates failed before policy/generation model initialization:

```text
AssertionError: Validation dataset is required if validation is enabled
```

This is a launcher/configuration bug, not evidence of OOM, graph fallback,
poor acceptance, or DFlash/DSpark performance. The previous render-only tests
did not compose parent defaults and missed the invalid configuration.

Evidence from actual failed runs:

- [Baseline S16](https://wandb.ai/nvidia/sna-specdec/runs/gyurbsd7), job 7098261.
- [DFlash K5 S32](https://wandb.ai/nvidia/sna-specdec/runs/b4y7l6h9), job 7098277.
- [DSpark K5 S16](https://wandb.ai/nvidia/sna-specdec/runs/gftm4zsr), job 7098269.

Each corresponding durable run directory contains `<jobid>-logs/ray-driver.log`
with the assertion. W&B authentication succeeded; git-code-artifact warnings
were not the terminating error.

## Fix and verification

Revision r2 adds only these behavioral overrides:

```yaml
grpo.val_period: 0
grpo.val_at_start: false
grpo.val_at_end: false
```

Run names include `r2` to distinguish recovery from failed runs. No model,
dataset, GBS, sequence limit, topology, backend, or CUDA Graph shape was changed.

`test_resolved_config.py` uses the production `load_config` and
`parse_hydra_overrides` functions, then resolves OmegaConf interpolation.
All 12 configurations reproduced the invalid validation condition before the
fix. All passed after the fix; the full launcher suite has six passing tests.
This proves configuration consistency, not GPU runtime success.

```bash
uv run --no-project --with hydra-core python -m unittest discover -s experiments/q30_dapo_concurrency_20260912/tests -v
```

## Old-job cleanup

Seven one-step jobs failed, and their seven dependent 20-step jobs were
automatically cancelled. The five remaining gates and five dependent jobs
were explicitly cancelled to avoid executing the known-invalid configuration:

`7098265, 7098267, 7098294, 7098296, 7098298, 7098301, 7098303, 7098305, 7098307, 7098309`.

A filtered scheduler check after cancellation confirmed no active/pending jobs
from the old concurrency cohort. No logs, checkpoints, or result files were
deleted. See [original receipts](SUBMISSIONS.md) for the initial job matrix.

## Recovery submission

Recovery retains the original 12-setting matrix and per-setting gate-to-20-step
dependency. `nemotron_n3_post` is selected through `Q30_DAPO47K_ACCOUNT` because
its observed FairShare was 0.756579, versus 0.667869 for the original account.
Partition remains `batch`, with a four-hour walltime.

Job receipts and the latest observed state are appended after submission.
No successful training step or speedup is claimed yet.
