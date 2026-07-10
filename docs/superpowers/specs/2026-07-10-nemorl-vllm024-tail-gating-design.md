# NeMo-RL vLLM 0.24 Tail-Gated Speculative Decoding Design

## Objective

Measure and improve speculative-decoding performance in synchronous NeMo-RL
rollouts by disabling speculation in compute-bound, high-concurrency decode and
enabling it only in the low-concurrency tail. Compare three scheduling families:

1. stock vLLM 0.24 DynamicSD on Model Runner V1;
2. an EfficientRollout-style roofline gate on Model Runner V2; and
3. a FastRL-inspired threshold and consecutive-check gate on Model Runner V2.

The first target is Qwen3-32B because official vLLM 0.24 selects Model Runner V2
for this architecture and the existing NeMo-RL branch has matched Eagle-3 and
CUDA-graph dispatch instrumentation. Qwen3-30B-A3B follows after the Qwen3-32B
smoke and correctness gates pass. Qwen3-30B-A3B uses V1 by default; a V2 support
smoke must pass before any V2 production row is created. Otherwise its stock,
threshold, and roofline variants remain a matched V1 cohort. A 32K-output cohort
follows the unmodified performance-recipe cohort.

## Upstream Reference Points

- EfficientRollout is Apache-2.0 and vendors vLLM 0.11.2. Its `EngineCore`
  owns a monotone rollout-local toggle, a prefill ramp guard, previous-rollout
  acceptance feedback, and a roofline decision using active batch size,
  sequence length, draft length, and expected accepted length.
- FastRL is Apache-2.0 and uses SGLang. Its Eagle worker waits for a configured
  number of consecutive decode batches below a threshold, activates
  speculation monotonically, and resets at the next rollout. Its adaptive
  strategy layer selects among preconfigured draft strategies by batch bucket.
- Official vLLM 0.24 provides `num_speculative_tokens_per_batch_size`, but
  resolves K independently on each scheduler step and marks DynamicSD as
  unsupported by Model Runner V2. It therefore selects Model Runner V1.
- Official vLLM 0.24 Model Runner V2 disables PIECEWISE draft-decode CUDA
  graphs for autoregressive Eagle. `FULL_AND_PIECEWISE` is required to exercise
  the supported FULL draft-decode graph path.

The implementation ports scheduling ideas, not drafter algorithms. The target
and Eagle-3 checkpoints remain unchanged.

## Alternatives

### Stock DynamicSD Only

Use official vLLM 0.24 without runtime changes. This is the lowest-risk
benchmark and provides adaptive K, but it forces Model Runner V1. It cannot be
used as a direct performance comparison against a V2 baseline.

### Vendored EfficientRollout vLLM 0.11.2

Use EfficientRollout's vLLM fork directly. This provides the original toggle
implementation but downgrades vLLM, PyTorch, CUDA-graph behavior, and NeMo-RL
integration. It is a reference implementation, not an experiment arm.

### Native vLLM 0.24 V2 Tail Gate

Port the minimal scheduler and EngineCore state machine to the installed vLLM
0.24 runtime and teach Model Runner V2 to skip proposal work when runtime K is
zero. This preserves the production runner and is the recommended path.

## Cohorts and Comparison Rules

Runner versions are separate cohorts. Speedups must never cross a cohort.

### V2 Cohort

| Variant | Activation | K policy |
|---|---|---|
| `baseline_v2` | never | 0 |
| `always_on_v2_k5` | from first decode | fixed 5 |
| `fastrl_threshold_v2_k5` | threshold plus consecutive checks, advance-only while off | fixed 5 |
| `fastrl_rebuild_v2_k5` | threshold plus consecutive checks, rebuild drafter at activation | fixed 5 |
| `efficient_roofline_v2_k5` | predicted speedup exceeds margin | fixed 5 |

After these variants pass, add a FastRL-inspired adaptive-strategy cohort with
prevalidated K values `1`, `3`, and `5`. Runtime K must never exceed the maximum
captured K. This phase is reported separately because adding variable-K support
to Model Runner V2 changes more code than binary on/off gating.

The adaptive-strategy phase compares two policies:

- an offline predefined mapping that selects the calibrated fastest K for each
  active-batch bucket; and
- a bucketed epsilon-greedy selector derived from FastRL, trained in calibration
  runs using accepted target tokens per decode millisecond as reward and frozen
  with epsilon `0` for the final 20-step evaluation.

EfficientRollout's acceptance-aware gamma ladder is evaluated with the same
`1/3/5` candidates after variable-K V2 parity passes. Its final row remains
separate from the fixed-K roofline row.

### V1 Cohort

| Variant | Activation | K policy |
|---|---|---|
| `baseline_v1` | never | 0 |
| `always_on_v1_k5` | from first decode | fixed 5 |
| `stock_dynamic_v1` | reversible per scheduler step | `5/4/3/1/0` by batch bucket |

Every V1 arm explicitly sets `VLLM_USE_V2_MODEL_RUNNER=0` and uses
`cudagraph_mode=PIECEWISE`. This prevents baseline and fixed-K jobs from being
auto-promoted to V2 while DynamicSD is forced to V1.

The initial stock schedule is:

| Scheduled requests | K |
|---:|---:|
| 1-16 | 5 |
| 17-32 | 4 |
| 33-64 | 3 |
| 65-128 | 1 |
| 129-512 | 0 |

## V2 Runtime Design

### Configuration

Extend the vLLM 0.24 speculative configuration with experiment-scoped fields:

- `sd_tail_gate_mode`: `off`, `threshold`, or `roofline`;
- `sd_tail_gate_threshold`: positive active-request threshold;
- `sd_tail_gate_consecutive_checks`: positive decode-check count;
- `sd_tail_gate_margin`: predicted-speedup safety margin;
- `sd_tail_gate_config_path`: calibrated roofline JSON path.

Unknown modes, missing roofline configuration, invalid thresholds, and
non-finite model outputs fail closed during engine initialization. The runtime
must not silently fall back to always-on speculation.

### State Machine

Each rollout starts with speculation disabled when a tail gate is configured.
The state machine tracks:

- whether active decode batch has exceeded the ramp threshold;
- consecutive qualifying decode checks;
- whether the monotone transition has fired;
- activation tick, batch size, and mean sequence length; and
- expected acceptance length from the previous training rollout.

No toggle is allowed until the ramp guard has observed a batch above the
threshold. This prevents the small number of early prefill-complete requests
from triggering speculation. After activation, speculation remains enabled
until the rollout engine wakes for the next training rollout.

Threshold mode fires after `sd_tail_gate_consecutive_checks` decode checks at
or below the threshold. The default experiment value is `10`, matching the
FastRL implementation.

Roofline mode evaluates:

```text
predicted_speedup = L_accept / (K * T_D / T_T + T_V / T_T)
```

It fires only when predicted speedup is at least
`1 + sd_tail_gate_margin`. The default experiment margin is `0.05`.

### Scheduler Contract

The scheduler reserves lookahead capacity for the configured maximum K so the
transition cannot fail for lack of KV slots. While the gate is off it emits
runtime K `0`; after activation it emits the configured K. Previously generated
draft tokens drain before the first fully autoregressive step.

The first target decode after activation may produce drafts for the following
scheduler step. This one-step delay is recorded and is not treated as a
failure.

### Model Runner V2 Contract

Model Runner V2 must read runtime K from `SchedulerOutput`.

- At K `0` in advance-only mode, run ordinary target sampling and the minimum
  first drafter forward required to advance the external Eagle-3 KV state, but
  return no draft token IDs and execute none of the `K-1` serial draft-decode
  iterations.
- At configured fixed K, preserve official vLLM 0.24 behavior.
- Binary gate phase rejects intermediate K values rather than silently running
  the maximum K.

External Eagle-3 does not share target KV state. Completely skipping its
proposer while the gate is off makes later reactivation incorrect. A second
FastRL-style rebuild mode may skip all decode-time drafter work while off, but
must requeue or extend every still-active request to reconstruct drafter KV
before producing drafts. Advance-only and rebuild modes are separate result
rows; no silent fallback between them is allowed.

Variable K `1/3/5` is implemented only after the binary path passes parity and
performance gates. Every supported K needs an explicit graph-coverage test.

### CUDA Graph Contract

All V2 variants use:

- `enforce_eager=false`;
- `compilation_config.cudagraph_mode=FULL_AND_PIECEWISE`;
- identical capture sizes and maximum capture size; and
- scheduler token capacity sufficient for `active_requests * (K + 1)`.

Target, draft-prefill, and draft-decode graph dispatch are logged separately.
No result is accepted if the candidate changes graph mode relative to its
matched baseline or silently falls back because of an undersized capture.

## Roofline Calibration

Port the Apache-2.0 EfficientRollout `sd_toggle` model as an experiment-owned,
typed module with attribution. Do not import the external reference checkout at
runtime.

Calibrate separately for each model, target TP, draft TP, K, and GB200 cluster
runtime. Measure target decode, draft, and verification latency at:

- batch sizes `1, 2, 4, 8, 16, 32, 64, 128`;
- sequence lengths `2048, 4096, 8192, 16384, 32768`; and
- K values `1, 3, 5`.

The first roofline run uses K5 and the measured acceptance length from the
previous rollout. Cold start uses a conservative configured value and logs it.
The calibrated JSON records model, checkpoint, vLLM commit, GPU, TP, container,
and calibration timestamp.

## Metrics

Record per training step and aggregate over Steps 2-20:

- E2E step time and throughput;
- generation time and throughput;
- logprob and policy-training time and throughput;
- proposed and accepted tokens, acceptance rate, and mean accepted length;
- active batch-size and mean sequence-length distributions;
- gate mode, decision count, activation tick, batch, and sequence length;
- predicted speedup and the `T_D/T_T`, `T_V/T_T`, and `L_accept` inputs;
- fraction of generation ticks and generated tokens with SpecDec enabled;
- target, draft-prefill, and draft-decode CUDA-graph/eager dispatch ratios; and
- reward, response length, approximate KL, policy loss, and invalid-value count.

Every report row includes SLURM job ID, W&B URL, cluster, recipe, commit,
container, runner version, graph mode, gate configuration, and calibration
config hash.

## Experiment Sequence

1. Create branch and worktree `sna/nemorl-vllm024-tail-gating` from commit
   `74dd0ba5e55f5c48949337324fb3b89e8f76291f` plus this design commit.
2. Reinitialize recursive submodules in the new worktree.
3. Implement and locally test the V1 stock cohort launcher and collector.
4. Implement the V2 binary tail-gate contract test-first.
5. Run CPU unit tests and a small generation parity test.
6. Submit Qwen3-32B two-step smokes for all V1 and V2 arms on one GB200
   cluster and monitor through policy training.
7. Run matched Qwen3-32B 20-step jobs after every smoke passes.
8. Repeat the validated matrix for Qwen3-30B-A3B.
9. Implement and test variable-K V2 strategy selection using K `1/3/5`.
10. Run the 32K-output long-tail cohort without changing prompt count,
    generation count, or recipe-owned training settings.
11. Publish W&B links and matched result tables to the canonical NeMo-RL
    performance report.

## Validation Gates

### Unit and Integration

- Invalid gate configuration fails before model loading.
- Ramp guard blocks early low-batch activation.
- Threshold mode requires the exact configured consecutive-check count.
- Roofline mode respects the configured margin and rejects non-finite output.
- Activation is monotone within a rollout and resets at the next wake-up.
- V2 advance-only K0 performs exactly the state-advance drafter pass, performs
  no serial draft-decode iterations, and leaves no consumable draft tokens.
- V2 rebuild mode performs no drafter decode while off and cannot produce a
  draft until every active request has rebuilt drafter state.
- Fixed-K V2 behavior is unchanged when gate mode is `off`.
- V1 stock DynamicSD remains unmodified and uses official scheduler behavior.
- Metrics accurately distinguish disabled, not-yet-activated, and activated
  states.

### Accuracy

SpecDec continues to use standard rejection sampling. Target sampling remains
`temperature=1.0`, `top_p=1.0`. No gate decision may alter the target sampling
configuration.

Exact token identity is not required under stochastic sampling. A candidate is
blocked if it produces NaN or invalid rewards, or if mean reward, response
length, approximate KL, or policy loss differs by more than 10% relative to its
matched cohort baseline over Steps 2-20 without an explained stochastic
confidence interval.

### Performance

A row is final only after Step 20 and provenance validation. Report:

- baseline-relative generation-time and E2E step-time speedup;
- baseline-relative generation and E2E throughput speedup; and
- always-on-relative improvement for each tail gate.

The gate is beneficial only if it avoids the always-on regression and does not
reduce E2E throughput below its matched baseline. Results from V1 and V2 are
displayed separately.

## Failure Handling

- CUDA graph fallback or capture OOM: preserve logs and fix capture geometry;
  do not accept an eager fallback result.
- Roofline configuration mismatch: fail closed; do not use another model's
  calibration.
- Missing gate metrics: mark the run incomplete.
- Timeout: retain completed-step telemetry as partial evidence and retry with a
  longer walltime without changing runtime settings.
- V2 dynamic-K failure: retain the validated binary gate and report variable K
  as unsupported until its separate parity gate passes.

## Non-Goals

- Porting EfficientRollout's quantized self-drafter.
- Porting FastRL's online drafter training.
- Calling the FastRL-inspired linear-K policy an exact reproduction of its
  SGLang Eagle tree-strategy MAB.
- Comparing performance across runner versions, clusters, graph modes, or
  unmatched recipe settings.
