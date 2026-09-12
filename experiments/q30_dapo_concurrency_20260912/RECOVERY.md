# DAPO concurrency recovery

## r3 outcome and r4 context correction (2026-09-12)

Pilot **7108119** started at 19:14:28 UTC and failed after 5m52s (exit 1).
The previous validation-dataset and float-type errors were passed. The next
failure was vLLM ModelConfig validation, before a completed generation step:

```text
User-specified max_model_len (49152) is greater than the derived max_model_len
(max_position_embeddings=40960.0 ... in model's config.json).
```

[Pilot W&B](https://wandb.ai/nvidia/sna-specdec/runs/e7efi6j1).
The Ray `cannot pickle ... ArgsKwargs` exception is secondary error transport;
the model-context validation is the original failure. This is not an OOM or
evidence of CUDA Graph fallback.

Read-only inspection of the actual staged Base target `config.json` confirmed
`max_position_embeddings=40960` and `rope_scaling=null`. The DAPO workload had
been transplanted with a 49,152-token limit without matching this target.
The native Qwen performance-40K recipe also uses a total length of 40,960,
but has a different dataset/topology: this study is still a DAPO adaptation,
not an unchanged native performance recipe.

Approved r4 correction: input cap 2,048, response cap 38,912, total context
40,960. Reward shaping uses the same response limit; training and logprob
packing budgets use the same total limit. GBS 2048, 128×16 rollouts, TP/EP/CP,
BF16 flashinfer_trtllm, FAP buckets and the nightly container are unchanged.
The aggregate scheduler token budget stays 49,152; unlike max_model_len, it
does not set the length of a single sequence. No long-context validation
bypass, RoPE override, model edit or core-code change is used.

Regression: all 12 rendered/resolved combinations failed the new native-context
bound assertion before the correction (`49152 > 40960`). Additional assertions
require input+response to fit and reward/packing limits to agree. Submit only
one corrected Baseline S16 pilot; expand after actual one-step completion.
Do not mix earlier failed revisions with the new `DAPO40K` run names.

### r4 submission receipt

- Job **7108906**, `nemotron_n3_post` / `batch`, eight 4-GPU nodes, one step.
- Runtime source **c3cfde43c**; remote fast-forward pull completed and recursive
  submodule SHAs remained unchanged. Existing nightly image reused.
- Run: `Qwen3-30BA3B-DAPO40K-Baseline-S16-1step-r4-20260912T194237Z`.
- Artifacts: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/q30-dapo-concurrency-20260912/Qwen3-30BA3B-DAPO40K-Baseline-S16-1step-r4-20260912T194237Z`.
- `sbatch --test-only` passed. Its synthetic ID 7108904 is not a submitted job.
- Snapshot **2026-09-12 19:42:49 UTC**: PENDING, reason None, start N/A.
  No runtime step, W&B run URL or performance result has been confirmed.
- All six local tests passed, including resolved-config checks for all twelve
  settings; Ruff, shell syntax and diff whitespace checks passed.
- No r4 SpecDec or 20-step jobs submitted. Next gate: inspect startup logs and
  monitor at least five minutes once running, then require actual one-step
  completion before expanding the matrix. Pending is not a successful gate.

## Historical r2 outcome and r3 repair (2026-09-12 18:50 UTC)

All twelve r2 gates failed; all twelve dependent measurements were cancelled.
Ray reached 32/32 workers, and the driver passed the previous validation
assertion. It then failed in vLLM target-model configuration, before generation:

```text
TypeError: Field 'router_aux_loss_coef' expected float, got int (value: 0)
```

The error is present in every r2 driver log. The launcher supplied
`policy.hf_config_overrides.router_aux_loss_coef=0`, and the actual vLLM
`hf_overrides` log confirms integer zero was passed to Transformers/HF strict
dataclass validation. [Baseline S16 r2](https://wandb.ai/nvidia/sna-specdec/runs/b0h5aokl)
is one evidence run; it has no completed performance steps.

Revision r3 changes only the numeric type to `0.0` and the run-name revision.
The resolved-config regression test now requires a float at this boundary;
all twelve combinations failed before the fix. The semantic loss coefficient
remains zero. GBS, concurrency, K, length limits, backend and graph shapes are
unchanged. No container, runtime library, model weight or core source is changed.

To avoid another whole-matrix startup failure, the next GPU submission is only
the matched Baseline S16 one-step pilot. The rest of the matrix and 20-step
measurements must wait for actual pilot completion. Local configuration tests
do not substitute for that gate. No speedup is available yet.

The r3 Baseline S16 pilot was submitted as **7108119** on
`nemotron_n3_post` / `batch` at 2026-09-12 18:54 UTC, after sbatch dry-run
success. Source commit: `9c1a4a8cb`. GBS=2048, max_num_seqs=16, one step,
eight 4-GPU GB200 nodes, same nightly container and all prior workload limits.
No r3 SpecDec or 20-step jobs have been submitted yet.

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

All 24 jobs were submitted at 2026-09-12 09:33–09:34 UTC after commit/push,
remote pull, recursive submodule verification, and per-job sbatch dry runs.
Runtime source commit: `1414458af`. Existing nightly container is unchanged.

| max_num_seqs | Method | r2 1-step gate | r2 20-step measurement |
|---:|---|---:|---:|
| 16 | Baseline | 7100287 | 7100289 |
| 16 | DFlash K5 | 7100291 | 7100293 |
| 16 | DSpark K5 | 7100295 | 7100297 |
| 32 | Baseline | 7100299 | 7100301 |
| 32 | DFlash K5 | 7100303 | 7100305 |
| 32 | DSpark K5 | 7100307 | 7100309 |
| 64 | Baseline | 7100311 | 7100314 |
| 64 | DFlash K5 | 7100316 | 7100318 |
| 64 | DSpark K5 | 7100320 | 7100322 |
| 128 | Baseline | 7100324 | 7100326 |
| 128 | DFlash K5 | 7100328 | 7100330 |
| 128 | DSpark K5 | 7100332 | 7100334 |

Snapshot at **2026-09-12 09:34:16 UTC**: all gates PENDING (Priority), all
measurements PENDING (Dependency). There are no completed steps or performance
results yet. Each 20-step job depends only on its own one-step gate.

Updated snapshot at **2026-09-12 09:40:12 UTC**: ten gates are RUNNING for
5 minutes 11–12 seconds; DFlash S128 (7100328) and DSpark S128 (7100332)
remain PENDING (Priority). All twelve 20-step measurements remain PENDING
(Dependency). The first five minutes of allocated-job startup were monitored.
No gate has completed and no performance result is available.

The inspected Baseline S16 and DSpark S32 launcher logs are still waiting for
the Ray head readiness marker. Their `ray-head.log` files were empty when
inspected, and no r2 `ray-driver.log` files were present in the bounded scan.
Thus the validation fix has passed local regression tests but has not yet
been verified past the validation assertion on GPUs. This is startup status,
not proof of healthy training or CUDA Graph replay; inspect subsequent logs
before interpreting a SLURM RUNNING state as benchmark progress.

Artifacts remain under the original durable experiment root, with distinct
`Qwen3-30BA3B-DAPO-<method>-S<seqs>-<steps>step-r2-<timestamp>` directories.
Original failed run directories are preserved. W&B group remains
`q30-dapo-gbs2048-concurrency-20260912`; exclude the original non-r2 failed runs
from performance aggregates.
