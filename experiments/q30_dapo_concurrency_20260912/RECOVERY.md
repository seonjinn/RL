# DAPO concurrency recovery

## S32/S64 SpecDec stage submitted (2026-09-14 05:32 UTC)

User approved DFlash K5 and DSpark K5 at S32/S64 to compare against the
default-concurrency Baseline and existing S16 measurements. All eight jobs
were submitted from source **ebbf3808c**, after local tests (8 passed), Ruff,
shell syntax and whitespace checks, push, remote fast-forward pull, unchanged
recursive submodule verification and individual `sbatch --test-only` checks.
No launcher, library or workload changes were made for this stage.

| Method | S | 1-step gate | 20-step measurement | Gate / measurement timestamp UTC |
|---|---:|---:|---:|---|
| DFlash K5 | 32 | 7134214 | 7134216 | 20260914T053147Z / 20260914T053150Z |
| DFlash K5 | 64 | 7134218 | 7134225 | 20260914T053152Z / 20260914T053154Z |
| DSpark K5 | 32 | 7134227 | 7134229 | 20260914T053156Z / 20260914T053158Z |
| DSpark K5 | 64 | 7134232 | 7134234 | 20260914T053201Z / 20260914T053203Z |

Account `nemotron_n3_post` had FairShare 0.759338 (versus 0.746817 for
`nemotron_sw_post`). Gates use `batch`/4h; measurements use `batch_long`/8h.
Each measurement depends only on successful completion of its own gate, with
kill-on-invalid-dependency enabled; no cross-method serialization. Artifacts
retain `Qwen3-30BA3B-DAPO40K-<method>-S<seqs>-<steps>step-r4-<timestamp>` names
under the existing durable root. Synthetic test-only IDs are not run IDs.

Snapshot 05:32:22 UTC: four gates **PENDING (Priority)** and four measurements
**PENDING (Dependency)**. No runtime/W&B success is asserted; five-minute
startup monitoring is outstanding until allocation. Compare completed Steps
3–20 and check graph dispatch/padding, KV-cache preemption, quality metrics
and realized per-engine concurrency. Config-level bucket envelope tests do
not establish GPU replay coverage. The measured runs are unprofiled; use a
separate diagnostic if ordinary logs cannot establish fallback or padding.

## Default-concurrency control submitted (2026-09-14 03:15 UTC)

User requested the missing comparison against a Baseline without the S16
concurrency cap. Submitted **7131657**, 20 steps, `nemotron_n3_post`,
`batch_long`, 8h, eight 4-GPU nodes, no dependency. Source **5f85218c0**.
Remote fast-forward pull and unchanged recursive submodule SHAs were verified.
The new launcher test first failed because `default` was unsupported, then
all eight tests passed, including resolved-config checks. Ruff, shell syntax,
whitespace checks and `sbatch --test-only` passed. Synthetic dry-run ID
7131656 is not an actual submitted job.

Run: `Qwen3-30BA3B-DAPO40K-Baseline-Sdefault-20step-r4-20260914T031507Z`.
Artifacts are under the existing durable experiment root. Compared with
Baseline S16, only `max_num_seqs` and explicit capture sizes are omitted
(plus run metadata). FAP remains enabled and uses vLLM automatic capture
sizing. Dataset, response/context caps, GBS, training settings, token budget,
backend, hardware and nightly container remain unchanged. No core runtime
code changed. This is not the untouched OpenMath performance recipe.

At 03:15:27 UTC: **PENDING**, reason `Nodes required for job are DOWN,
DRAINED or reserved for jobs in higher priority partitions`; no start estimate.
Runtime default concurrency, graph sizes, W&B URL and five-minute startup
monitoring remain unverified until the job starts. Do not label this RUNNING
or report a default-concurrency speedup yet. Compare completed Steps 3–20
against Baseline S16 and the frozen DFlash/DSpark S16 measurements.

## Runtime progress (2026-09-13 22:14 UTC)

- DSpark gate **7126827** completed successfully (24m36s, exit 0), including
  logprob and policy training. [W&B](https://wandb.ai/nvidia/sna-specdec/runs/qwtk79vd).
- Its dependent 20-step job **7126843** started at 22:06:44 UTC and was still
  RUNNING after seven minutes. Initial five-minute monitoring is complete;
  driver logs show target PIECEWISE/FULL and DSpark FULL captures completing.
  Policy initialization is ongoing; no measurement step is confirmed yet.
  [20-step W&B](https://wandb.ai/nvidia/sna-specdec/runs/5lxpf0zi).
- Baseline 20-step **7126823** completed step 1 and reached step 2 logprobs.
  [W&B](https://wandb.ai/nvidia/sna-specdec/runs/f5f12uza).
- DFlash gate **7126825** completed successfully (22m17s, exit 0), including
  generation, logprobs and policy training. Its 20-step job **7126840** remains
  PENDING after the successful gate; runtime startup monitoring is outstanding.
  [Gate W&B](https://wandb.ai/nvidia/sna-specdec/runs/5nn9tyth).

First-step diagnostics from the driver logs (not Steps 3–20, not a release claim):

| Run | Generation TPS/GPU | E2E TPS/GPU | Generation s | Total step s | Policy s | Logprob s |
|---|---:|---:|---:|---:|---:|---:|
| Baseline 20-step, step 1 | 772.68 | 551.19 | 806.09 | 1130.01 | 223.14 | 82.62 |
| DFlash K5 1-step gate | 1605.48 | 880.94 | 388.10 | 707.29 | 220.66 | 82.62 |
| DSpark K5 1-step gate | 1674.33 | 902.87 | 372.03 | 689.92 | 222.10 | 81.21 |

Mean generated lengths (Baseline, DFlash, DSpark) are 9571.03, 9574.78 and
9571.93 tokens. Mean rewards are -0.1528, -0.1542 and -0.1294; logged
generation KL rounds to 0.0017 for each.
These summaries do not establish output quality equivalence. The DSpark row
and DFlash row are gates, not their 20-step measurements. Final comparisons must use the
matched 20-step jobs and Steps 3–20, with output-quality checks. CUDA Graph
capture completed in the gates; full replay coverage remains unprofiled.

## Pilot passed; 20-step stage (2026-09-13)

Job **7108906** completed (exit 0, allocation elapsed 29m18s), including
generation, logprobs and policy training. [W&B](https://wandb.ai/nvidia/sna-specdec/runs/brpzyvzn).
The single recorded step took 1132.85 s: generation 805.86 s (71.1%), policy
training 227.51 s, logprobs 82.88 s. Logged generation/E2E TPS/GPU were
772.90/549.80. Mean generation length was 9571.03 tokens. These are first-step
diagnostics, not a warmup-excluded benchmark or a completed 40K-output cohort.
FAP piecewise and full-decode captures completed; replay coverage is not yet
profiled. No matched SpecDec speedup exists yet.

User approved 20 steps. Start with only S16: Baseline, frozen DFlash K5, frozen
DSpark K5. Baseline can proceed directly; each SpecDec measurement depends on
its own one-step gate. No serial dependency between methods. Other concurrency
settings remain unsubmitted in this stage.

Read-only partition check: `batch` maximum 4h, `batch_long` maximum 7d.
20 × 1132.85 s is 6.29h before startup; submit measurements with 8h on
`batch_long`, while gates stay 4h on `batch`. This changes scheduling only,
not the workload or container. Checkpointing stays disabled; preemption or
timeout makes the run incomplete and requires restart. Do not call an
interrupted window a completed 20-step benchmark.

### Submitted 20-step stage

Source **0be37fe20**, account `nemotron_n3_post`; remote pull and recursive
submodule checks completed. All five `sbatch --test-only` checks passed.
Seven local tests, Ruff, shell syntax and whitespace checks passed.

| Method | 1-step gate | 20-step measurement | Run timestamp (UTC) |
|---|---:|---:|---|
| Baseline | 7108906 (completed) | 7126823 | 20260913T213237Z |
| DFlash K5 | 7126825 | 7126840 | gate 20260913T213240Z; measurement 20260913T213259Z |
| DSpark K5 | 7126827 | 7126843 | gate 20260913T213242Z; measurement 20260913T213301Z |

All use S16. Artifacts use the same durable root and
`Qwen3-30BA3B-DAPO40K-<arm>-S16-<steps>step-r4-<timestamp>` naming.
Snapshot **2026-09-13 21:33:13 UTC**: both new gates PENDING (Priority),
both SpecDec measurements PENDING (their own afterok dependency). Baseline
measurement PENDING with reason `Nodes required for job are DOWN, DRAINED or
reserved for jobs in higher priority partitions`. This is a scheduling
condition, not an observed application failure. No new run has started;
five-minute runtime monitoring and Steps 3–20 results remain outstanding.
No new W&B run URL is asserted before logger initialization.

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
