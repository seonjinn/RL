# New Draft online canary submissions

Actual submission receipts, September14 2026:

| Job | Method | Scope |
|---|---|---|
| 7152082 | DFlash B8, servingK5, window2048 | Two-step always-online, checkpoint at2 |
| 7152083 | DSpark B8, servingK5, window2048 | Two-step always-online, checkpoint at2 |

Account `nemotron_sw_post`, partition `batch`, one exclusive four-GPU GB200 node
per method, two-hour time limit, no inter-job dependencies. Both test-only
scheduling checks passed before submission; synthetic IDs7152078/7152079 are
not actual run IDs. Git push and remote ff-only pull preceded sbatch.

Source `f7dd041da05c8fc9fd85e4e7970b778e8e483b70`, bundle SHA256
`da831e8f14a0bb5e1183d5000ae4959163dd766d46c30cdfe775c20730e9f101`.
The previously pinned nightly container remains unchanged. Source and actor
environments are staged node-locally, including one target/drafter copy per node.
W&B credential availability was checked without printing its value.

Durable root:
`/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/q8-rp25-swa-20260914/online-f7dd041da/`

Each `dflash/` or `dspark/` directory holds scheduler logs, resolved overrides,
source/asset identities, training log, cadence artifacts and checkpoints.
Planned W&B IDs (existence is not established until Python initializes W&B):

- `nvidia/sna-specdec/q8rp44-dflash-canary-7152082`
- `nvidia/sna-specdec/q8rp44-dspark-canary-7152083`

No200step production arm has been submitted. Before doing so, require real
train→refit→next-rollout evidence, generation-side attention semantics, and
checkpoint/optimizer/cadence resume verification. These canaries do not measure
production speedup and do not validate CP>1 or multi-node execution.

Last live check: 22:09:41 UTC, both jobs RUNNING for4m14s; immutable source
checkout and input staging completed and dependency installation was progressing.
The following SSH checks failed at DNS resolution of the Park hostname. The
five-minute running-monitor requirement was therefore not fully verified; do
not infer that the cluster jobs stopped, passed, or failed from that local
connectivity error. Actual GRPO training metrics are not yet confirmed.

## Resolved status and launcher recovery

At 23:47 UTC, scheduler and logs confirmed both canaries FAILED before GRPO:
7152082 after41m52s, 7152083 after41m34s. Actor preflights passed, then the
hardcoded `/opt/nemo_rl_venv/bin/python` was not executable (ENOENT). No training,
refit, reward, or throughput result was produced. The image metadata identifies
the same pinned August18 nightly, not a newly selected container. Whether its
venv was absent or its interpreter link was masked remains unverified.

Recovery uses the already-created, frozen-lock Megatron environment as the
driver interpreter through `uv run --no-project --no-sync --python`. A real
`examples.run_grpo` import must pass before constructing the remaining runtime.
The renderer writes files synchronously so subprocess failure cannot be hidden
by `mapfile` process substitution. Missing/dangling interpreter tests and both
method rendering tests pass locally; this does not yet prove GPU execution.
The image, model, serving K, attention window and study workload are unchanged.

Recovery submissions (September15 UTC / September14 Pacific): DFlash7154339,
DSpark7154340. Source8f3875f0dab2e708aafe1240cc452cf84f9d809b; bundle SHA256
f8a441658f5e44d57ed8188acfb763192eefc062b0bd43fafa0fcd32775eebde.
Both passed test-only and were independently submitted to nemotron_sw_post/batch.
Durable root is the same experiment prefix with `driver-8f3875f0d/{dflash,dspark}`.
The first scheduler check showed both PENDING, not training success. W&B IDs
remain planned until driver initialization; no200step production job submitted.

Subsequent live check: both jobs RUNNING beyond6minutes, with pinned-source
checkout and dependency builds progressing. The five-minute startup observation
is satisfied, not the full runtime gate. New logs show the image Python is a
symlink to /root/.local/share/uv/python/cpython-3.13-linux-aarch64-gnu/bin/python3.13;
the selected driver no longer relies on that image-internal link. Its underlying
target/mount failure still requires separate diagnosis; no new GRPO step yet.

## Root cause confirmed by same-image A/B probe

The read-only probe in allocation7154339 reproduced ENOENT with default home
mounting: the symlink existed, but its /root interpreter target did not. With
only `--no-container-mount-home` added, the exact target was present and ran
Python3.13.14. Host-home mounting was masking the image's interpreter, not a
missing interpreter in the original nightly. The failure is unrelated to drafter
size, GPU OOM, or model weights. Probe step7154339.1 exited127 as expected.

Future launcher invocations disable home auto-mount and keep UV-managed Python
installs in node-local scratch. Running canaries7154339/7154340 retain source
8f3875f0d and its independent validated-driver fix; they are not retrospectively
relabelled as using the later mount patch. Preserve their ongoing environment builds.

## Driver metadata recovery (September15 UTC)

7154339 and7154340 both FAILED after44m04s/44m02s. Ray and W&B initialized,
but no GRPO step completed. TransferQueue's `_resolve_tq_pin()` called
`importlib.metadata.requires("nemo-rl")` in the MCore environment reused as
the driver. That environment was synced with `--no-install-project`, so its
source imports worked while its project distribution metadata was missing.

Recovery installs the exact checked-out project editable with `--no-deps`
after frozen MCore dependency sync. A preflight compares installed and source
TransferQueue requirements, then calls the actual `_resolve_tq_pin` before Ray.
No data-plane guard is bypassed and no dependency lock, model, image, or study
configuration is changed. The previous home-mount correction remains enabled.

Local regression: an isolated environment with an unresolvable training
dependency receives only project metadata; the helper succeeds without fetching
that dependency. All7 metadata/render/study tests pass. This is not yet proof
of GPU training, update/refit, or checkpoint/resume success. Fresh canaries are
required before the approved eleven-condition200step production matrix.

Recovery canaries submitted: DFlash7157073 and DSpark7157074, independently
on nemotron_sw_post/batch. Source iscc340a27da2ea898d524aa89ff27a66479e106df;
bundle SHA25675d54b3dc972ca27b41419b64bd564b9239d8a1231cec974ce949368e350b05d.
Both test-only checks passed (7157067/7157068 are synthetic planning IDs,
not actual submitted jobs). Commit/push, remote ff-only pull and checksum
verification preceded submission. Durable root: the same experiment prefix
with `metadata-cc340a27d/{dflash,dspark}`. Each canary remains two steps,
always-online, with checkpoint at2; no200step production job submitted yet.

Startup observation completed: at03:19 UTC both jobs were RUNNING for more
than5minutes (DFlash5m25s, DSpark5m22s). Frozen dependency installation/builds
were progressing; no new terminal error was visible. This meets the initial
five-minute monitoring requirement only. Driver metadata preflight, GRPO steps,
update/refit and resume have not yet been confirmed in these fresh runs.

## Checkpoint-transition OOM recovery (September 15 UTC)

Terminal results supersede the startup snapshot above:

- DFlash 7157073 completed two updates/refits and saved step 2, including a
  successful cadence checkpoint receipt. Resume has not been verified.
- DSpark 7157074 completed two updates/refits, then failed while preparing the
  step-2 checkpoint. `grpo_sync.py` called `prepare_for_training`, which reloaded
  optimizer tensors onto CUDA while the colocated vLLM engine remained awake.
  The OOM reported 131.51 GiB for vLLM, 52.07 GiB for the policy process and
  only 23.62 MiB free on a 184.31 GiB GPU; a 46 MiB allocation failed.

The target is Qwen3-8B, not Qwen3-30B-A3B. These figures include runtime/KV-cache
allocations and optimizer state, not just target/drafter checkpoint weights.
The step-end cadence refit wakes generation weights and KV cache; the checkpoint
entry previously omitted the corresponding generation release. DFlash passing
does not establish that this shared lifecycle error is safe for that method.

The patch calls and waits for `finish_generation()` before optimizer onload for
colocated checkpointing, fails closed if release fails, and times preparation
as `checkpointing_prep`. Non-colocated checkpointing is unchanged. No model,
precision, batch size, context length, vLLM memory budget or container changed.
The real checkpoint entry block fails the CPU ordering regression before the
patch and passes afterward; all 11 transition/metadata/render/study checks pass.
This is not GPU OOM-resolution or checkpoint/resume evidence yet. A fresh
DSpark two-step checkpoint gate precedes the approved 200-step matrix.

Recovery submission: **7158090**, `sna-q8-dspark-checkpoint-recovery`,
`nemotron_sw_post / batch`, one exclusive four-GPU GB200 node. Source
`acb7146de1c76afbae8d1194147f9c1640360b44`; immutable bundle SHA256
`60ac65e964f795ef154683e142b33aaa21649d87d8f11f65707f9fcd3942760e`.
Commit/push, remote fast-forward pull, checksum, credential-presence and
test-only checks passed before sbatch. 7158086 is the synthetic test-only ID,
not a submitted job. Durable root:
`/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/q8-rp25-swa-20260914/checkpoint-acb7146de/dspark/`.
W&B ID `q8rp44-dspark-canary-7158090` is planned until W&B initializes.
Submission is confirmed; startup, checkpoint success and resume are not yet.

Startup observation: at 04:36:52 UTC, 7158090 was RUNNING for 5m23s.
The pinned commit was checked out, the container started with its interpreter
visible, and frozen dependency builds were progressing without a new terminal
error. This satisfies five-minute startup monitoring, not GPU training or
checkpoint/resume validation; the corrected checkpoint boundary has not run yet.

## Resume gate (September 15 UTC)

At 05:40 UTC, accounting confirmed DFlash 7157073 and patched DSpark 7158090
both COMPLETED with exit 0 (52m11s / 52m12s). Both saved a successful step-2
cadence checkpoint. DSpark released generation memory before optimizer onload;
the checkpoint-transition OOM did not recur in this two-step recovery run.
This establishes checkpoint save, not 200-step stability or resume correctness.

The explicit `resume-check` launcher continues each existing result directory
from step 2 to step 4. It preserves original launch artifacts and writes the
new logs under `attempts/resume-JOBID`. The target, new rp25-44000 B8 drafter,
K5, GBS8, TP2 training, CP1, generation settings and pinned container are
unchanged. The checkpoint manager restores model, optimizer, dataloader and
cadence state; success requires step-4 evidence bound to the step-2 resume.
The full eleven-condition 200-step matrix remains gated on this GPU validation.

Submitted independently on `nemotron_sw_post / batch`:

| Method | Resume job | Checkpoint source | W&B run ID (after initialization) |
|---|---|---|---|
| DFlash | 7159073 | `metadata-cc340a27d/dflash/checkpoints/step_2` | `q8rp44-dflash-resume-check-7159073` |
| DSpark | 7159074 | `checkpoint-acb7146de/dspark/checkpoints/step_2` | `q8rp44-dspark-resume-check-7159074` |

Paths are relative to the experiment prefix above. Each allocation is one
exclusive four-GPU GB200 node, with a two-hour limit. Source
`79a62bd23641fedf9f704268dbaabf5435ff1197`, source bundle SHA256
`f495423b3a5d6d802473f61169f32d096f4eb43cca380b8cec64085171f61efc`.
Commit/sign-off/push, remote fast-forward pull, credential-presence, bundle
checksum and both test-only checks passed before submission. The test-only
IDs 7159071/7159072 are synthetic and are not submitted jobs. All 13 local
regression tests, shell syntax and whitespace checks passed. At 05:44:35 UTC,
both submitted jobs were PENDING; this is not evidence of successful resume.

Both jobs started at 05:46:52 UTC. During environment bootstrap, before either
attempt created `train.log`, the previous `checkpoint-runtime.json` and
`schedule-runtime.json` were moved into the corresponding `attempts/resume-JOBID`
directory. Their SHA256 values were verified unchanged before and after.
This preserves old terminal summaries and frees their exclusive output names
for step 4. No checkpoint weights, optimizer state, ledger or step-2 receipt
was removed or changed. The launcher now performs this explicit archival for
future resume submissions; the two in-flight jobs use the original pinned
source plus this documented result-directory preparation, not live code edits.

At 05:52:18 UTC, both jobs were RUNNING for 5m26s. The pinned source and
container started and dependency builds were progressing without a new error
in the inspected log tails. Five-minute startup monitoring is complete;
actual checkpoint restore, steps 3–4, and 200-step production remain unverified.

## Independent 200-step submissions (September 15)

06:59 UTC accounting confirmed both resume checks FAILED (exit 1), after
46m59s / 46m55s. Neither failed from CUDA OOM: Megatron's optimizer scheduler
rejected `wd_incr_steps=32` against saved `16`, caused by changing the canary
horizon from two to four steps with GBS8. Actual resumed training remains
unverified. Do not label these jobs successful or reuse them as performance data.

The user then explicitly requested independent parallel runs. The approved
eleven fresh 200-step conditions will no longer wait on resume checks. All
use the unchanged Q8 cadence workload, new rp25-44000 B8/K5 drafters, source
checkpoint-memory fix and independent result roots; baseline stages no drafter.
The shared launcher now accepts `production` plus the exact arm as argument 7.
Use `nemotron_sw_post/batch`, one exclusive four-GB200 node per arm, a three-hour
walltime and no dependency. The queue controls concurrency. Local rendering
checks all eleven arms and verifies the baseline cannot accidentally enable
SpecDec through local path overrides. These runs are diagnostic until completed
and quality/receipt evidence is reviewed.

All eleven submitted at approximately 07:04 UTC; all were RUNNING at 07:05:34
UTC (36 seconds elapsed). No dependencies. Runtime source
`1554f5f51d04acc10aa247e9755a989e75c5cd7f`, bundle SHA256
`1a2b9b35e81a136965672dfec314e46881cfaecb734174b5656f2fd7059e508d`.
Each test-only check passed before any actual submission. Synthetic planning
IDs 7160345–7160355 are not jobs. Actual jobs:

| Arm | Job ID |
|---|---|
| Baseline (no SpecDec) | 7160356 |
| DFlash frozen | 7160357 |
| DFlash always | 7160358 |
| DFlash fixed-5 | 7160359 |
| DFlash fixed-10 | 7160360 |
| DFlash fixed-20 | 7160361 |
| DSpark frozen | 7160362 |
| DSpark always | 7160363 |
| DSpark fixed-5 | 7160364 |
| DSpark fixed-10 | 7160365 |
| DSpark fixed-20 | 7160366 |

Root: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/q8-rp25-swa-20260914/production-1554f5f51-20260915/ARM/`.
W&B: `nvidia/sna-specdec`, run ID `q8rp44-ARM-production-JOBID` after initialization.
Run IDs and stdout are also retained per arm on Lustre.

Separate resume recovery: preserve the saved optimizer scheduler through
`policy.megatron_cfg.scheduler.use_checkpoint_opt_param_scheduler=true` and
`override_opt_param_scheduler=false`, only for resume-check mode. Verified
against pinned Bridge 8c46dc425 and MCore 14346b65a: WD schedule size is
train_iters × GBS, giving 16 vs 32. Target and draft optimizer parameter groups
share this optimizer scheduler; the draft cadence scheduler is separate.
These flags do not change the fresh 200-step cohort. The CPU reproduction and
17 local checks pass; distributed resume has not yet passed. A failed startup
retry may find terminal summaries already archived in the previous attempt;
the launcher now permits their absence without changing checkpoint data.

Scheduler-preserving resume checks submitted independently: DFlash **7160406**,
DSpark **7160407**, source `d7df16db84a9d171c3c7862ea946b71621229cdb`, bundle SHA256
`72f764831c41f7480a7b1a436139842570587151ad75a80a0a2cc7c397e67bb0`.
Same step-2 roots and separate `attempts/resume-JOBID` folders; W&B IDs are now
`q8rp44-dflash-always-resume-check-7160406` and
`q8rp44-dspark-always-resume-check-7160407`.
All pre-submit checks passed; 7160404/7160405 are synthetic test-only IDs.
At 07:07:24 UTC these checks were pending, while all eleven 200-step jobs were
running for 2m25s. No dependency links the two sets. GPU restore success and
200-step quality/performance results are still pending.

07:13:13 UTC observation: all eleven production jobs were RUNNING for 8m15s;
both corrected resume checks were RUNNING for 5m17s. Frozen dependency builds
were progressing in all inspected log tails, without a new terminal error.
Five-minute startup monitoring is complete for all thirteen. No first GRPO
step, corrected resume success or completed 200-step result is claimed yet.

## Current-state resume fix and 300-step rerun (September 19 UTC)

The previous scheduler-preserving resume jobs 7160406 and 7160407 failed before
resumed training because the saved scheduler correctly emitted
`state_version=2`, while `grpo.restore_draft_update_scheduler` still contained a
duplicated `state_version==1` guard. The scheduler implementation itself already
supports and validates both legacy v1 and current v2 states. Source
`24c825f4f7c75f5ee2111891b19798eaa530a12b` removes the stale wrapper guard and
adds a regression test that restores the current state emitted by the live
scheduler. Schema, version, invariant, and configuration validation remain in
`DraftUpdateScheduler.create`.

The same source adds a matched five-step smoke profile and a 300-step profile
with checkpoints at steps 50, 100, 150, 200, 250, and 300. Frozen arms use an
interval of 301 and therefore remain frozen for the full horizon. The workload,
target, rp25-44000 B8 drafters, K5 serving, and other comparison settings are
unchanged. All exact commands passed `sbatch --test-only` before submission.
Jobs use `coreai_dlalgo_nemorl / batch`, one exclusive four-GB200 node each, and
no dependencies.

Five-step matched smoke jobs:

| Arm | Job ID |
|---|---:|
| Baseline | 7279780 |
| DFlash frozen | 7279781 |
| DFlash always | 7279782 |
| DSpark frozen | 7279783 |
| DSpark always | 7279784 |

Three-hundred-step jobs:

| Arm | Job ID |
|---|---:|
| Baseline | 7279785 |
| DFlash frozen | 7279786 |
| DFlash always | 7279787 |
| DFlash fixed-5 | 7279788 |
| DFlash fixed-10 | 7279789 |
| DFlash fixed-20 | 7279790 |
| DSpark frozen | 7279791 |
| DSpark always | 7279792 |
| DSpark fixed-5 | 7279793 |
| DSpark fixed-10 | 7279794 |
| DSpark fixed-20 | 7279795 |

Current-state resume gates use the preserved step-2 checkpoints: DFlash
7279796 and DSpark 7279797. The immutable source bundle SHA256 is
`9bcfd54d4f664199aea2c529832f86d9acc1036fa3ea0320b8a462265d5d1a80`.
New smoke and production results are rooted at
`/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/q8-rp25-swa-20260914/rerun-24c825f4f-20260919/`.
All 18 jobs entered RUNNING together at 2026-09-19 06:59:26 UTC. At this point
they were in SLURM Prolog; no training, resume, or performance success is claimed.
At 07:05:38 UTC all 18 remained RUNNING for 6m12s with no scheduler-level
failure. Each inspected stdout had entered the pinned container and resolved the
image Python; actor-environment installation was still in progress, so driver,
vLLM, MCore, resume, and first-step success were not yet claimed.

## September 19: GBS512 / 32K packed online matrix

The user approved a generation-dominant follow-up using 64 prompts times eight
generations, sequence packing, 32K total length, FULL_AND_PIECEWISE CUDA Graphs,
and `enforce_eager=false`. The matrix compares the standard engine-default
baseline plus S64/S128 matched baselines against DFlash and DSpark K5 at S64 and
S128 with frozen, fixed-10, and always-online schedules. Each independent job is
20 policy steps with checkpoints at steps 5, 10, 15, and 20. There are no SLURM
dependencies.

Source commit: `5f37bf70ce392b63722041b9f30c52c17520a9f5`.
Immutable bundle SHA256:
`18f9289bfcfddc88357af5161860514a11df2bd49775e1ed6a12935a29bfb11c`.
Artifact parent:
`/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/q8-rp25-swa-20260914/gbs512-32k-packed-online-5f37bf70c-20260919`.
All 15 exact commands passed `sbatch --test-only` before submission. Jobs use
`nemotron_n4_post / batch`, one exclusive four-GB200 node, and a four-hour limit.

| Arm | Job ID |
|---|---:|
| Baseline, engine default | 7287122 |
| Baseline S64 | 7287123 |
| Baseline S128 | 7287124 |
| DFlash frozen S64 | 7287125 |
| DFlash frozen S128 | 7287126 |
| DFlash fixed-10 S64 | 7287127 |
| DFlash fixed-10 S128 | 7287128 |
| DFlash always S64 | 7287129 |
| DFlash always S128 | 7287130 |
| DSpark frozen S64 | 7287131 |
| DSpark frozen S128 | 7287132 |
| DSpark fixed-10 S64 | 7287133 |
| DSpark fixed-10 S128 | 7287134 |
| DSpark always S64 | 7287137 |
| DSpark always S128 | 7287138 |

Submission alone is not a performance result. CUDA Graph capture/replay,
resolved concurrency, finite quality metrics, and valid policy steps remain
runtime gates.

At 2026-09-19 15:48 UTC, all 15 jobs remained `RUNNING` after 6m10s or
longer, each on a distinct GB200 node. A bounded scan of all 15 scheduler logs
found no traceback, CUDA OOM, `ModuleNotFoundError`, or explicit failure/error.
The sampled baseline-default, DFlash-always-S128, and DSpark-always-S128 logs
were still installing their frozen runtime dependencies. This satisfies the
five-minute startup observation only; Python driver initialization, CUDA Graph
capture/replay, the first valid GRPO step, checkpoint/resume, quality metrics,
and performance results are not yet established.

### W&B result snapshot (September 20)

W&B contains all 15 runs. All 12 SpecDec runs finished through the 20-step
horizon (`summary_step=21`). The three no-SpecDec baselines are marked crashed:
default and S64 reached `summary_step=18`, while S128 reached 16. Their W&B
runtimes are 11.36--11.56 ks. OCI-HSG SSH was unavailable during this audit,
so the terminal scheduler/log cause is not yet confirmed.

The later scheduler audit confirmed that jobs 7287122--7287124 all ended in
`TIMEOUT` at 04:00:21--04:00:24. They were not OOM or application crashes.
Each baseline produced valid checkpoints through step 15; the 12 SpecDec arms
completed all 20 steps in 03:00:33--03:18:40.

The table below uses the closed Steps 3--15 window, the latest window shared by
all 15 runs. Every value has 13 valid observations. Speedups use the baseline
with the same S64 or S128 cap. Throughput speedups are the preferred comparison
because mean generated length differs across runs.

| Concurrency | Condition | Gen TPS/GPU | Gen TPS speedup | Gen time (s) | E2E time (s) | E2E time speedup | E2E TPS/GPU | E2E TPS speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| S64 | Baseline | 3047.7 | 1.000x | 419.1 | 629.5 | 1.000x | 2034.3 | 1.000x |
| S64 | DFlash frozen | 7157.3 | 2.348x | 171.0 | 379.3 | 1.660x | 3301.4 | 1.623x |
| S64 | DFlash fixed-10 | 7094.4 | 2.328x | 182.9 | 404.0 | 1.558x | 3283.2 | 1.614x |
| S64 | DFlash always | 7308.0 | 2.398x | 171.2 | 391.3 | 1.609x | 3260.2 | 1.603x |
| S64 | DSpark frozen | 6689.1 | 2.195x | 201.0 | 429.2 | 1.467x | 3179.2 | 1.563x |
| S64 | DSpark fixed-10 | 7047.7 | 2.312x | 185.0 | 408.6 | 1.541x | 3249.3 | 1.597x |
| S64 | DSpark always | 6876.8 | 2.256x | 189.1 | 419.6 | 1.500x | 3154.9 | 1.551x |
| S128 | Baseline | 3106.2 | 1.000x | 474.8 | 715.1 | 1.000x | 2052.5 | 1.000x |
| S128 | DFlash frozen | 7134.4 | 2.297x | 181.4 | 399.9 | 1.788x | 3281.4 | 1.599x |
| S128 | DFlash fixed-10 | 7232.8 | 2.328x | 187.5 | 419.9 | 1.703x | 3273.4 | 1.595x |
| S128 | DFlash always | 7255.3 | 2.336x | 190.0 | 448.1 | 1.596x | 3123.9 | 1.522x |
| S128 | DSpark frozen | 7123.1 | 2.293x | 188.4 | 415.7 | 1.720x | 3280.1 | 1.598x |
| S128 | DSpark fixed-10 | 7376.2 | 2.375x | 179.9 | 404.4 | 1.768x | 3345.8 | 1.630x |
| S128 | DSpark always | 7084.6 | 2.281x | 199.3 | 447.2 | 1.599x | 3194.7 | 1.557x |

Reward (0.8125--0.8444 for the aggregated SpecDec rows), approximate entropy
(0.2583--0.2837), and generation KL error (0.000669--0.000698) remain in the
same broad range as their matched baselines. Acceptance is approximately
46.4--47.2% for DFlash and 45.3--46.9% for DSpark, with mean accepted lengths
of about 3.32--3.36 and 3.26--3.35, respectively. The 20-step horizon does not
show a consistent acceptance benefit from always-online updates.

Do not average `policy_kl_error` blindly for this cohort. Isolated outliers
occur at baseline-default step 10 (680.56), baseline-S64 step 3 (10.61), and
DFlash-frozen-S64 step 14 (2.22), while surrounding values are generally near
1e-3. Baseline S128 and the fixed/always SpecDec curves do not show the same
large spikes in Steps 3--15. Root-cause analysis is still required before a
quality sign-off.

## September 20: resumable 300-step GBS512 / 32K packed matrix

Commit `e19ee57ce1c6ae9e82d0d663e0e6000dc02b316c` adds segmented execution
without changing the 300-step optimizer or cadence horizon. The workload and
15 conditions match the 20-step matrix. Baselines stop every 15 steps because
the measured 20-step controls exceeded the four-hour `batch` limit; SpecDec
arms stop every 20 steps. Each stop closes a full policy, optimizer,
dataloader, and cadence checkpoint. The next array stage uses `aftercorr`, so
task index N resumes only task index N. `--kill-on-invalid-dep=yes` prevents a
failed condition from skipping ahead while unrelated conditions continue.

Both array shapes passed `sbatch --test-only`. Actual stages are:

- Stages 1--15, 15 tasks each: jobs 7320372--7320386.
- Stages 16--20, the three slower baseline tasks only: jobs 7320387--7320391.

The immutable source bundle SHA256 is
`5ef7790522505228828c371609dd1f9da0a99aa942b24f5827b300aac0fa1475`.
Submission receipts, scheduler logs, checkpoints, and per-segment attempts are
under
`/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/q8-rp25-swa-20260914/segmented-300-e19ee57ce-20260920/results`.
The account is `coreai_dlalgo_nemorl`, partition `batch`, with one exclusive
four-GPU GB200 node per array task and a four-hour limit. Submission is not a
runtime or performance result; checkpoint closure and stage-2 resume remain
explicit gates.

### Stage-1 result and targeted recovery (September 21)

The three no-SpecDec controls completed their first 15-step segment with exit
zero and durable `step_15` checkpoints. Their correlated stage-2 allocations
7320392--7320394 then started from those result roots. The earlier standalone
20-step baselines 7287122--7287124 were not application failures: they reached
valid step-15 checkpoints and then hit the four-hour SLURM limit. Splitting the
baseline into 15-step segments removes that walltime failure mode without
changing the 300-step optimizer horizon.

The 12 SpecDec stage-1 tasks exposed a separate checkpoint-finalization defect.
They reached `step_20`, but `checkpointing.keep_top_k=1` pruned `step_10` before
the cadence runtime writer consumed its sealed decision-ledger prefix. Every arm
then failed closed with `FileNotFoundError` for
`checkpoints/step_10/draft-decision-ledger.jsonl`; this was not an OOM, CUDA
Graph, target-model, or drafter-weight failure. Source
`4fbfaa2d06fb71723850a7c2b34dbca1a088debc` retains the current and immediately
previous checkpoint with `keep_top_k=2`, while keeping storage bounded. It also
adds a SpecDec-only correlated retry mode that preserves original array indices
3--14. The relevant 32-test suite, formatting, shell syntax, bytecode compile,
and whitespace checks passed before submission.

The first retry array 7325027 never entered the workload because its submitted
full expected SHA was mistyped even though the first nine characters matched.
All 12 tasks failed in about 70 seconds at immutable-source checkout with
`fatal: reference is not a tree`; no training result is attributed to them.
The source bundle itself is valid and advertises the correct commit. A corrected
SpecDec-only chain was submitted after verifying the exact full commit, bundle
contents, and SHA256. Its stages are 7325243--7325257 under
`segmented-300-ledgerfix-4fbfaa2d0-corrected-20260921/results`; stage 1 is the
runtime gate for the ledger-retention fix. The submission boundary now has a
regression test that rejects a bundle unless its SHA256 matches and it advertises
the exact expected commit, preventing a repeat before any `sbatch` call.
