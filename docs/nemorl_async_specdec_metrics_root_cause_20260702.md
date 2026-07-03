# NeMo-RL Async-1off SpecDec Metrics Root Cause

Date: 2026-07-02

## Scope

This note explains why completed NeMo-RL async-1off SpecDec runs have valid
throughput and generic vLLM telemetry in W&B, but no speculative-decoding
acceptance metrics.

The primary evidence is the completed Qwen3-32B Eagle-3 K=7 run:

- SLURM job: `2259822`
- W&B run: `https://wandb.ai/nvidia/sna-nemorl-specdec-lyris/runs/ibtx92sp`
- Source revision: `1271b1530181a7378e40de40b4b46ad223e6596c`
- Recipe: `examples/configs/recipes/llm/performance/grpo-qwen3-32b-8n4g-async-1off.yaml`
- CUDA Graph: enabled (`enforce_eager=false`)
- vLLM metrics logger: enabled

## Observed Evidence

The run completed 20 steps and its W&B summary contains generic generation
telemetry, including `train/generation_logger_metrics`, in-flight request plots,
pending request plots, KV-cache use, and generation-token counters.

The same W&B summary contains zero keys matching `spec`, `accept`, or `draft`.
The driver log confirms that every async vLLM model-owner actor started the
`vLLM Metric Logger`, so the missing acceptance values are not caused by W&B
authentication or by the generic logger being disabled.

## Root Cause

Async-1off executes `async_grpo_train()` and delegates generation to
`AsyncTrajectoryCollector`. Each prompt group is generated in a background
thread by `_run_prompt_group_worker()` and only its regular rollout metrics are
stored in the replay buffer.

Unlike the regular GRPO loop and `SyncRolloutActor`, the async collector never
calls any of the SpecDec metric lifecycle methods:

- `policy_generation.snapshot_step_metrics()`
- `policy_generation.get_step_metrics()`
- `policy_generation.clear_logger_metrics()`
- `policy_generation.get_logger_metrics()` at a prompt-group or target-version
  boundary

The async training loop reads generic logger metrics before refit, but it does
not reset those metrics after each read. Consequently, the generic metric
arrays are cumulative, and no SpecDec counter delta is added to the W&B step
payload.

The existing `BaseVllmGenerationWorker._get_raw_spec_counters()` and
`VllmGeneration.get_step_metrics()` code is therefore unreachable from the
actual async-1off trajectory path. Adding more W&B flattening alone cannot fix
the omission.

## Concurrency Constraint

vLLM exposes SpecDec acceptance as engine-global monotonic counters. Async-1off
can have prompt groups from adjacent target weight versions in flight at the
same time. A global counter delta can accurately describe the engine interval,
but it cannot be attributed exactly to one replay-buffer training batch without
either pausing generation or adding per-request acceptance data to vLLM.

Pausing and draining the engine at every training step would change the
async-1off performance being measured. The low-overhead report should therefore
label these values as engine-interval metrics, not exact sampled-batch metrics.

## Minimal Instrumentation Design

The proposed implementation is metrics-only and does not alter logits,
sampling, scheduler decisions, model weights, or trajectory contents.

1. Extend the async vLLM worker logger result with current SpecDec counters and
   an interval baseline.
2. Aggregate counters across DP leaders in `VllmGeneration`, preserving the
   existing generic logger payload.
3. At each async refit boundary, read one engine-interval delta and immediately
   reset its baseline for the next interval.
4. Log derived acceptance rate, mean accepted length, reporting-worker count,
   and a `metrics_partial` flag to the existing `sna-nemorl-specdec-lyris`
   project.
5. Keep baseline runs free of SpecDec acceptance keys and retain all existing
   generic vLLM plots.

The existing local candidate `vllm_worker_async.py` must not be copied wholesale:
it is based on an older async worker and would remove newer socket locking,
compilation-config compatibility, and other upstream changes. Only the isolated
metric lifecycle should be ported into a fresh worktree.

## Verification Gates

Unit tests must cover:

- counter snapshot and delta calculation;
- aggregation across multiple DP leaders;
- reset behavior between training steps;
- missing and partial worker metrics;
- baseline behavior with no speculative configuration.

A Lyris max-steps=2 smoke must then prove:

- Eagle-3 reports nonzero draft and accepted-token counters in W&B;
- the second step is an interval delta rather than a cumulative total;
- baseline has no fabricated acceptance values;
- reward, generated token count, logprob-error metrics, and output lengths stay
  consistent with an uninstrumented control;
- generation and E2E throughput do not regress beyond run-to-run noise.

## Separate Qwen3-235B Issue

This logging omission is independent of the Qwen3-235B exact async recipe
all-gather timeout and independent of the vLLM V1 target-TP/draft-TP mismatch
guard. Those execution issues require separate matched experiments and must not
be hidden by the metric patch.
