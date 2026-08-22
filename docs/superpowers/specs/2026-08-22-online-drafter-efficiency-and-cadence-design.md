# Online Drafter Efficiency and Adaptive Cadence Design

## Status

This is an architectural follow-up to the Qwen3-8B online-drafter PR stack.
The design is based on PR11 head
`3922322332967d9d4a5e2975b5238a934755f606` for source inspection only.
Implementation must start from the later PR11 head that has a terminal GREEN
full gate and terminal GREEN packed TP2 x CP2 DFlash-to-DSpark end-to-end gate.
Head drift invalidates the implementation base and requires re-review.

The work is split into two independently reviewable projects:

1. semantics-preserving removal of online-training and refit overhead;
2. opt-in fixed and adaptive draft-update scheduling, which intentionally
   changes how often the drafter learns and is refitted.

Project 1 ships first. Project 2 remains experimental until its 300-step and
1000-step evidence gates pass.

## Motivation and Evidence

The matched Qwen3-8B packing-disabled matrix used three same-node paired
replicates per shape, 30 steps per arm, and the closed measurement interval
5 through 29:

| Shape | Fixed E2E | Online E2E | Online overhead | Fixed policy | Online policy | Fixed refit | Online refit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GBS32, MBS1 | 19.703 s | 21.663 s | 10.066% | 3.263 s | 4.216 s | 3.796 s | 4.502 s |
| GBS64, MBS1 | 29.691 s | 31.987 s | 7.519% | 6.072 s | 8.021 s | 3.944 s | 3.998 s |
| GBS64, MBS2 | 27.391 s | 29.449 s | 7.539% | 3.432 s | 4.411 s | 3.936 s | 4.956 s |

GBS64 amortizes relative online overhead compared with GBS32. MBS2 materially
reduces absolute policy time at GBS64, but it does not reduce the relative
online overhead below about 7.5%. The policy path is the stable primary target;
refit measurements are noisier and require same-node paired confirmation.

The completed 1000-step packing-disabled online/fixed matrix found no
statistically consistent generation-throughput gain. DFlash K5 improved mean
acceptance by 2.17 percentage points, but its generation-TPS confidence interval
still crossed zero. The other DFlash/DSpark K values showed smaller or
inconsistent acceptance changes. This makes unconditional per-step draft work
an optimization target, but it does not justify changing the default cadence
without convergence evidence.

Prior source-proven optimizations exist on isolated branches:

- `15fc942fa62d18e6a0a013639ab2dbf9cbeaf882` removes a second TP
  metadata-agreement pass from each DFlash draft microbatch and, when the
  optional update probe is enabled, batches its scalar diagnostics into one
  device-to-host transfer;
- `f909e3d124bb663db4099e88f6846e55b0500912` defers split draft-loss scalar
  materialization and caches the finalized normalization scalar;
- `7169ab837` replaces four hook-time clones plus a later concatenation with
  one validated backing concatenation;
- `b010aefc4` buckets TP draft export by device and dtype so sharded parameters
  share one payload gather per bucket;
- `f9826b3b8` replaces gathered manifest digests with one fixed-size consensus
  all-reduce.

These changes are candidates to restack onto the final PR11 head. They are not
assumed compatible merely because their original branches passed. In
particular, PR11's CP-aware hidden capture already removed the old hook-time
clones, so `7169ab837` is a source of test ideas rather than an approved
cherry-pick. A fresh allocation/copy count must prove a remaining win before
that change is adapted.

The observed policy-time deltas scale approximately 1:1:2 with the inferred
draft microbatch counts for GBS32/MBS1, GBS64/MBS2, and GBS64/MBS1. This is
consistent with per-microbatch synchronization overhead, but it is not a
component attribution because the current heads have not yet been profiled.

## Research Context

There is no universal research convention of updating every 10 or 100 NeMo-RL
optimizer steps. The ICML 2024 Online Speculative Decoding implementation
triggers learning when a sample buffer reaches `online_update_interval`; the
published Spider example uses eight buffered samples. OnlineSPEC examples use
query chunks of 40 for EAGLE/EAGLE-3 and 80 for Hydra. A query count, a buffered
sample count, and a NeMo-RL optimizer step are not interchangeable.

The design therefore exposes explicit NeMo-RL step semantics and records the
actual update and refit counts. Adaptive scheduling is treated as a new NeMo-RL
experiment, not as a claimed reproduction of those systems.

Primary references:

- https://arxiv.org/abs/2310.07177
- https://github.com/LiuXiaoxuanPKU/OSD/tree/788a403d5495896b4fc5b7f56cfd41de5ae61967
- https://arxiv.org/abs/2603.12617
- https://github.com/ZinYY/OnlineSPEC/tree/e58f82eb3f3adca3a686211236bf4f6e9e7e3a2b

## Goals

### Project 1: semantics-preserving efficiency

- Preserve one successful draft optimizer update and one post-update refit on
  every online-training step.
- Preserve policy and draft losses, global normalization, gradients, optimizer
  state, checkpoint state, and generated-model weights.
- Remove redundant host synchronization, tensor copies, serialization, and
  collectives from the policy and refit critical paths.
- Keep the existing fixed-training path unchanged.
- Reduce the matched GBS64/MBS2 online policy-time delta by at least 20%.
- Reduce matched online E2E overhead from about 7.5% to at most 5% without a
  statistically supported generation-TPS or acceptance regression.

### Project 2: opt-in update cadence

- Support an unchanged `always` mode, fixed sparse update intervals, fixed
  refit-only intervals, and an adaptive sparse-update mode.
- Make a single controller-owned decision per policy step; workers never make
  rank-local cadence decisions.
- Persist and restore cadence state exactly.
- Bound staleness even when acceptance metrics are absent or stable.
- Measure whether reduced update/refit frequency preserves the acceptance and
  convergence benefit while lowering total step time.

## Non-goals

- Do not change speculative decoding acceptance semantics.
- Do not change policy optimizer cadence.
- Do not overlap training and generation in the first implementation.
- Do not retain hidden states, logits, or autograd graphs across policy steps.
- Do not make adaptive cadence the default in this PR series.
- Do not claim a throughput win from acceptance-rate improvement alone.
- Do not use a 30-step run to select a 40-step or 100-step cadence.

## Project 1 Architecture

### 1. Establish a final-head profile

The final PR11 head receives NVTX/timer coverage around:

- hidden-state capture and materialization;
- draft provider preparation and forward;
- draft-loss construction and backward;
- finish-time normalization and gradient correction;
- optimizer finalization;
- draft export, TP reconstruction, serialization, transfer, and vLLM apply.

Instrumentation must not call `.item()`, synchronize CUDA, or materialize
tensors when disabled. The profiling harness runs exact same-node fixed/online
pairs at GBS64/MBS2 with packing disabled and enabled. It reports time per
optimizer step, time per microbatch, collective counts, host scalar
materializations, and transferred bytes.

### 2. Restack proven bounded optimizations

Each prior optimization is reapplied independently with strict RED-to-GREEN
tests against the final PR11 source:

1. validate DFlash TP projected-loss metadata exactly once per microbatch,
   while retaining full validation for direct generic callers and fail-together
   behavior for asymmetric ranks;
2. when the optional update probe is enabled, transfer its scalar diagnostics
   once per step rather than synchronizing each statistic independently;
3. defer draft-loss metric scalar materialization;
4. cache one normalization scalar per finalized draft step;
5. gather TP export payload once per device/dtype bucket;
6. use one fixed-size manifest consensus collective.

The first change has a counted current-head source target: the redundant inner
validation issues one header `all_gather`, three broadcasts, three mismatch
`all_reduce` calls, three mismatch `.item()` synchronizations, and three
contiguous metadata clones per draft microbatch. The third and fourth changes
target approximately `3M + 2` forced scalar materializations for `M` split
draft microbatches. These are operation counts from source and regression
tests, not measured runtime attribution.

Hidden-state capture is profiled separately. It may adopt the one-backing-
allocation design only if a final-head dispatch test demonstrates fewer
allocations or bytes than the CP-aware implementation without retaining stale
forward tensors. `torch.cuda.empty_cache()` placement is also excluded from
the initial restack because both fixed and online paths execute it and changing
it requires an explicit peak-memory/OOM gate.

The commits remain separable so a reviewer can reject one optimization without
blocking the others. Full PR11 CP1/CP2/CP4 and packed E2E gates run after their
composition.

### 3. Evidence-driven follow-up

No additional hot-path rewrite is approved by this design without a profile
showing one of the following:

- repeated synchronization whose result is reused within one optimizer step;
- repeated tensor materialization with identical logical contents;
- per-parameter control or payload collectives that can be bucketed without
  changing reconstruction order;
- serialization or conversion of data already resident in the required format.

Every follow-up must name the removed operations, provide a counted regression
test, and report matched GPU timing. Kernel fusion, persistent cross-step
buffers, and asynchronous refit require a separate design amendment.

## Project 2 Configuration

Add a user-facing `DraftUpdateScheduleConfig` as a Pydantic discriminated union
using `mode` as the discriminator. This avoids a single model whose adaptive
defaults accidentally become populated in `always` mode. Every member uses
`extra="forbid"`; defaults live only on the member schema and are documented
in the exemplar YAML.

The union members are:

- `AlwaysDraftUpdateScheduleConfig`: `mode: Literal["always"] = "always"` and
  no cadence fields;
- `FixedDraftUpdateScheduleConfig`: `mode: Literal["fixed"]`, required
  `action: Literal["sparse_update", "refit_only"]`, and required positive
  `fixed_interval`;
- `AdaptiveDraftUpdateScheduleConfig`: `mode: Literal["adaptive"]`,
  `action: Literal["sparse_update"] = "sparse_update"`, positive
  `min_interval=10`, `max_interval=100`, `ewma_alpha=0.1`,
  `degradation_threshold=0.02`, `recovery_threshold=0.01`,
  `min_observations=20`, and `max_burst_updates=10`.

Adaptive validation requires `max_interval >= min_interval`,
`ewma_alpha` in `(0, 1]`, positive-integer `min_observations` and
`max_burst_updates`, and finite thresholds satisfying
`0 <= recovery_threshold < degradation_threshold <= 1`. The enclosing draft
config defaults to the `always` member. Incompatible or misspelled fields
therefore fail at config validation instead of being ignored. Fixed and
adaptive modes are opt-in and appear in separate experiment recipes, not the
default GRPO recipe.

## Project 2 Components

- `nemo_rl/models/policy/draft_config.py` owns the user-facing schedule schema
  and validation and nests it under the existing DFlash/DSpark draft configs.
- A new focused `nemo_rl/algorithms/draft_update_schedule.py` owns the internal
  state dataclass, immutable per-step decision, EWMA, hysteresis, counters, and
  serialization. It has no Ray, CUDA, Megatron, or vLLM dependency.
- `nemo_rl/algorithms/grpo_sync.py` and the supported single-controller loop
  consume the same shared scheduler API. Neither loop reimplements cadence
  arithmetic.
- `nemo_rl/models/policy/workers/megatron_policy_worker.py` consumes the
  immutable decision and gates hidden capture, draft provider/loss work, draft
  gradient update, and update evidence. It does not inspect acceptance metrics.
- The existing refit lifecycle consumes the controller decision and successful
  update result. Scheduler code does not call vLLM directly.
- Existing checkpoint metadata carries the scheduler's versioned serialized
  state; checkpoint readers validate config compatibility before restoration.

## Project 2 State and Decision Flow

An internal `DraftUpdateScheduleState` dataclass stores:

- the schedule-origin step;
- optional last successful draft-update and applied-refit steps;
- acceptance EWMA;
- frozen reference acceptance EWMA;
- valid-observation count;
- whether the hysteresis controller is in monitoring, training-burst, or
  awaiting-post-refit-observation state;
- successful updates in the current training burst;
- total attempted, successful, skipped, and forced updates/refits.

The GRPO controller owns this state. At the beginning of each policy step it
produces one immutable decision and passes it to the training workers.
For a fresh fixed or adaptive run, the starting global step initializes only
`schedule_origin_step`; the last-success fields remain `None` until real
events succeed. Update and refit ages are derived from the corresponding last
success or the origin, never stored as a second checkpoint source of truth.
The first scheduled update therefore occurs after the configured interval
rather than being silently forced at step one. `always` retains the existing
step-one update behavior. Tests cover interval one and a nonzero origin.

### Always

Draft hidden capture, loss, backward, optimizer update, and post-update refit
run every step exactly as before.

### Fixed sparse update

The draft path runs when `steps_since_update >= fixed_interval`. On skipped
steps, hidden capture and provider/loss construction are disabled, draft
parameters receive no gradients, and no draft refit occurs. Policy training is
unchanged.

### Fixed refit only

Draft training runs every step, but generation weights are refitted only when
`steps_since_refit >= fixed_interval`. This amortizes refit but deliberately
serves stale draft weights between refits. It does not remove policy-training
compute and is evaluated separately from sparse update.

### Adaptive sparse update

The controller begins in monitoring state. It collects `min_observations`
finite acceptance values to establish the reference EWMA. Missing, NaN, or
infinite observations do not increment this count. In monitoring, valid
observations update the current EWMA; once established, the reference may rise
to a better EWMA but never fall automatically. After at least `min_interval`
steps, the controller enters training-burst state when current acceptance is at
least `degradation_threshold` below the frozen reference.

Every step follows one causal order:

1. generation with the currently applied draft weights produces an optional
   acceptance observation;
2. the controller validates and consumes that observation, then produces the
   immutable training/refit decision for the same policy step;
3. a requested draft update and refit execute and succeed fail-together;
4. after a refit, the controller enters
   `awaiting_post_refit_observation`, so the next valid generation observation
   evaluates those weights before another adaptive update is allowed.

In `awaiting_post_refit_observation`, a missing observation pauses the burst;
it does not spend another update or increment the burst count. A valid
observation first updates the current EWMA, and recovery compares that updated
EWMA with the frozen reference. A gap at most `recovery_threshold` returns the
controller to monitoring. A larger gap resumes the burst and requests the next
update on that step. The reference remains frozen throughout the burst. This
hysteresis prevents update flapping and prevents a degraded run from silently
accepting a lower reference.

One maintenance update is forced when update age reaches `max_interval`, even
if acceptance is missing or stable. It is followed by the same awaiting state;
a later valid observation either returns to monitoring or begins a degradation
burst. If no valid observation arrives, no rapid sequence of blind updates is
issued. The refit from the `max_burst_updates`-th successful update is still
evaluated by its next valid post-refit observation. Only when that observation
updates the EWMA and still fails recovery does the controller fail the run with
the frozen reference, current EWMA, and decision history in its error. It never
rebaselines or silently returns to monitoring.

Only a successful draft optimizer update can set `last_update_step`, and only
a successful refit that applies those weights to the serving drafter can set
`last_refit_step`. A failed update or refit fails the step and is never counted
as a successful cadence event.

## Checkpoint and Resume Contract

Cadence state is stored in the training checkpoint beside the global training
step. Resume restores it before the first generation/refit decision. A legacy
checkpoint without cadence state may resume only in `always` mode. Fixed or
adaptive resume from such a checkpoint fails at startup with an actionable
message rather than silently resetting staleness.

The checkpoint contract records schedule configuration, state version, update
count, refit count, and last successful steps. A resumed run must produce the
same next decision as an uninterrupted run given the same acceptance sequence.
Version 1 requires exact equality of all resolved schedule fields on resume.
There is no implicit migration or threshold/interval override: a state-version
or config mismatch fails before workers start. A future migration requires an
explicit versioned converter and uninterrupted-versus-resumed sequence tests.

## Observability

Every step logs numeric W&B metrics without adding device synchronization to
the hot path:

- `train/draft_schedule/update_requested`;
- `train/draft_schedule/update_successful`;
- `train/draft_schedule/refit_requested`;
- `train/draft_schedule/refit_successful`;
- `train/draft_schedule/steps_since_update`;
- `train/draft_schedule/steps_since_refit`;
- `train/draft_schedule/acceptance_ewma`;
- `train/draft_schedule/reference_acceptance_ewma`;
- `timing/train/draft_policy_total`;
- `timing/train/draft_refit_total`.

Existing canonical generation throughput, generation time, acceptance rate,
accepted length, policy time, logprob time, refit time, and total step time
remain authoritative. Throughput is read from the logged canonical metric and
is never reconstructed from averaged times.

## Error Handling

- Invalid interval, threshold, or hysteresis configurations fail during config
  validation.
- A controller decision includes its global step and a monotonically
  increasing decision ID. Workers reject stale or mismatched decisions.
- Missing acceptance observations do not trigger rapid updates; `max_interval`
  provides one fail-safe update, followed by an observation wait.
- A skipped draft update must prove that draft gradients are absent and draft
  parameters are bitwise unchanged after the policy optimizer step.
- A post-update refit without at least one successful draft update since the
  previous applied refit is rejected. Startup refit remains a separate
  lifecycle event before the schedule origin. Any validation path that mutates
  the serving drafter must route through the same applied-refit accounting and
  cadence decision; otherwise cadence experiments disable that validation
  refit or use an isolated generation engine. An out-of-band serving refit is a
  fail-loud contract violation.
- Multi-rank update, refit, and failure outcomes use the existing fail-together
  contracts.

## Testing

### Project 1

- CPU-profiler tests count `_local_scalar_dense` operations.
- TP2 counting tests assert one projected-loss metadata agreement and preserve
  asymmetric-input fail-together behavior.
- Update-probe tests count one diagnostics device-to-host transfer when the
  probe is enabled; the default disabled path remains zero-cost.
- Hidden-capture dispatch/counting tests are required before, not after,
  accepting any one-backing-allocation adaptation.
- TP2 distributed tests assert one payload gather per device/dtype bucket and
  one manifest consensus collective.
- Loss and gradient parity cover empty draft ownership, CP1, CP2, and CP4.
- Exact final-head focused, full MCore, and packed DFlash-to-DSpark E2E gates
  must all be terminal GREEN.

### Project 2

- Table-driven unit tests cover `always`, fixed 10/40/100, adaptive trigger,
  multi-step burst, hysteresis recovery, burst cap, forced max interval, and
  missing observations.
- Resume tests compare uninterrupted and restored decision sequences.
- Worker tests prove a skipped sparse update performs no hidden capture, draft
  forward, draft backward, draft optimizer change, or refit.
- Refit-only tests prove per-step draft updates and exact periodic refits.
- Failure tests prove unsuccessful update/refit counters and state do not
  advance.
- Distributed tests prove all ranks receive the same decision and fail
  together on mismatch.

## Experiment Plan

### Semantics-preserving performance gate

Run three same-node sequential replicates per topology. Each replicate contains
three arms: fixed-drafter control, online final-PR11 baseline, and online
optimized. Rotate arm order across replicates to reduce warm-cache and temporal
bias. Each arm executes through logical training Step 30. The closed analysis
interval is the 25 canonical W&B training rows with `_step=5..29`; earlier rows
are warmup and Step 30 is a completion/flush guard. The report includes both
online-minus-fixed overheads and the optimized-minus-baseline online delta. A
source-diff guard must prove that the optimized product changes are outside the
fixed execution path; otherwise add a fixed baseline-head arm as a fourth
control.

Compare:

1. fixed versus online baseline versus online optimized, CP1, packing disabled;
2. fixed versus online baseline versus online optimized, CP1, packing enabled;
3. fixed versus online baseline versus online optimized, CP2, packing enabled;
4. CP4 packing-enabled only after CP2 passes and scheduling cost permits.

Hold model, draft model, revision, K, data order, seed, GBS, MBS, image, CUDA
Graph settings, and node constant within each replicate. Record structurally
unique W&B IDs, exact product/harness heads, resolved-config parity, and the
execution order for every arm.

### Cadence pilot

Use DFlash K5 first because it was the only 1000-step configuration with a
consistent acceptance improvement. Run matched fixed-control and `always`
controls beside each cadence candidate with the same model/draft revisions,
K, data order, seed, GBS, MBS, packing/SP topology, image, CUDA Graph settings,
and node. Record arm order and alternate it when a comparison is repeated.

Run a 300-step elimination pilot for:

- `always`;
- fixed sparse update at 10 and 40;
- fixed refit-only at 10 and 40;
- adaptive sparse update with min 10 and max 100.

Interval-100 sparse-update and refit-only candidates run 600 steps so the
analysis contains at least five scheduled events and their post-event
observations. The 300/600-step pilots
are elimination-only: a point estimate may reject a candidate, but it cannot
establish a production win. Run packing-enabled CP1 first, then repeat only
the surviving candidates at CP2. A 30-step cadence pilot is forbidden because
interval 40 and 100 would not produce representative update events.

### Long validation

Promote at most two cadence candidates. Compare them with `always` and fixed
drafter controls for 1000 steps using three matched replicates if the result is
intended to support a production claim. A fresh 1000-step run reports all ten
closed canonical W&B training-row windows `_step=1..100` through
`_step=901..1000`; a run resumed after completed Step 400 reports the
predeclared six windows `_step=401..500` through `_step=901..1000`. Final
system/flush rows such as `_step=1001` are excluded. The runner verifies this
mapping against checkpoint `training_info.current_step` and fails rather than
shifting a window if the logger convention changes. The primary aggregate is
the full common-step intersection of those predeclared windows, without
imputation. Fewer replicates may guide development but remain experimental.

## Analysis Contract

For each replicate and closed step window, let `T_fixed`, `T_always`, and
`T_candidate` be arithmetic means of the canonical logged total-step metric.
Define always-online overhead as `O_always = T_always - T_fixed`, candidate
overhead as `O_candidate = T_candidate - T_fixed`, and cadence overhead
reduction as `(O_always - O_candidate) / O_always`. The denominator must be
positive; otherwise the comparison is reported as invalid rather than coerced.

For Project 1, replace `T_candidate` with the online-optimized arm and also
report `(T_optimized - T_baseline) / T_baseline` for policy, refit, generation,
logprob, and total-step metrics. Per-replicate paired differences are the
sampling units. Report their mean, sample standard deviation, and 95% t
confidence interval; do not treat individual training steps as independent
replicates.

Generation non-inferiority requires the lower 95% bound for relative canonical
generation TPS to be above -2%, the lower bound for acceptance-rate difference
to be above -1 percentage point, and the lower bound for accepted-length
difference to be above -0.1 tokens. Cadence convergence additionally requires
finite loss/gradient metrics and no two consecutive 100-step windows in which
the candidate's draft-loss mean exceeds `always` by more than 20% with the
paired confidence interval wholly above zero. The canonical mean total reward
is also non-inferior only when its paired lower 95% bound is above -1
percentage point for the bounded math reward. The canonical `gen_kl_error`
must remain finite, and its paired upper 95% bound must not exceed the larger
of 10% above the `always` mean or an absolute increase of 0.01. The harness
resolves and records the exact W&B keys before submission and fails if a
required metric is absent. These margins and formulas are frozen before
submission and reported even when a candidate fails them.

## Success Criteria

Project 1 may be proposed as a performance change only if:

- all correctness and E2E gates pass on the exact performance head;
- policy-time delta falls by at least 20% on the primary GBS64/MBS2 pair;
- matched optimized-online E2E overhead over fixed is at most 5%;
- generation TPS, acceptance, and accepted length pass the predeclared
  non-inferiority margins, and loss/gradient parity remains exact within the
  existing correctness tolerances.

Project 2 remains experimental unless:

- the 1000-step cadence overhead-reduction estimand is at least 50% and its
  paired 95% confidence interval is wholly above zero;
- acceptance and accepted length pass the predeclared non-inferiority margins;
- no two consecutive 100-step windows meet the predeclared draft-loss
  divergence rule;
- total reward and `gen_kl_error` pass the predeclared policy-quality and
  stability gates;
- the 1000-step result preserves checkpoint, resume, and exact update/refit
  accounting.

If no cadence candidate satisfies these gates, only Project 1 ships and
`always` remains the sole supported production behavior.
