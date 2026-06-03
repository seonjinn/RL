# Async GRPO Deep Dive

This guide explains how async GRPO works in NeMo RL from the beginning to the
end of a training step. It complements the shorter
[`async-grpo.md`](async-grpo.md) setup guide by focusing on implementation
behavior, replay-buffer semantics, refit, off-policy correction, and common
debugging questions.

The main files to follow are:

- `examples/run_grpo.py`
- `nemo_rl/algorithms/grpo.py`
- `nemo_rl/algorithms/async_utils.py`
- `nemo_rl/experience/rollouts.py`
- `nemo_rl/models/generation/vllm/vllm_generation.py`
- `nemo_rl/models/policy/lm_policy.py`
- `nemo_rl/algorithms/loss_functions.py`
- `nemo_rl/algorithms/advantage_estimator.py`

## Mental Model

Synchronous GRPO alternates between generation and training:

```text
generate rollouts -> compute rewards/logprobs/advantages -> train -> refit -> repeat
```

Async GRPO overlaps generation and training:

```text
background collector -> replay buffer -> trainer samples current target -> train
          ^                                                     |
          |                                                     v
          +------------------- refit generation weights <--------+
```

The async path is not a separate RL objective. It is the same GRPO-style policy
gradient update, but the rollout source can lag the training policy by one or
more weight versions. That makes the rollouts off-policy relative to the policy
that the training step treats as "previous", so the loss must include an
importance-sampling correction.

## Required Configuration

Async GRPO requires all of the following:

```yaml
policy:
  generation:
    backend: vllm
    colocated:
      enabled: false
      resources:
        num_nodes: 1
        gpus_per_node: 1
    vllm_cfg:
      async_engine: true

grpo:
  async_grpo:
    enabled: true
    max_trajectory_age_steps: 1
    in_flight_weight_updates: false
    recompute_kv_cache_after_weight_updates: false

loss_fn:
  use_importance_sampling_correction: true
```

`async_grpo_train()` asserts these important properties:

- vLLM backend with `vllm_cfg.async_engine: true`
- non-colocated generation
- `loss_fn.use_importance_sampling_correction: true`

`examples/run_grpo.py` also rejects several DAPO-style features for async mode,
including dynamic sampling, reward scaling, and reward shaping.

## Components

### Driver and Setup

`examples/run_grpo.py` loads the config, initializes Ray, creates the tokenizer,
builds datasets and environments, calls `setup()`, then chooses between
`grpo_train()` and `async_grpo_train()`.

`nemo_rl/algorithms/grpo.py::setup()` builds the shared GRPO runtime:

- logger
- checkpoint manager
- train dataloader
- validation dataloader
- `ClippedPGLossFn`
- Ray virtual clusters
- training `Policy`
- vLLM `VllmGeneration`
- optional reward-model or NeMo Gym resources

For async GRPO, the train and inference resources are separate because
colocated inference is not supported.

### Replay Buffer

`nemo_rl/algorithms/async_utils.py::ReplayBuffer` is a Ray actor that stores
per-prompt trajectory groups. Internally it is list-backed:

```python
self.trajectories = []
self.trajectory_versions = []
self.target_weight_versions = []
```

The lists are protected by a thread lock. A single buffer entry corresponds to:

```text
1 prompt x grpo.num_generations_per_prompt completions
```

For example, if `num_generations_per_prompt: 16`, all 16 generated responses for
one prompt are stored together in one replay-buffer entry. This grouping is
intentional because GRPO computes advantages across the responses for the same
prompt.

Each buffer entry stores:

- `batch`: a CPU `BatchedDataDict` containing message logs, generated assistant
  messages, token ids, generation logprobs, rewards, loss multipliers, truncation
  flags, and task metadata
- `rollout_metrics`: generation length, reward, turn count, duration, and other
  rollout diagnostics
- `timestamp`
- `trajectory_version`: the model weight version used to generate the trajectory
- `target_weight_version`: the training weight version where this entry is meant
  to be consumed

The buffer lives in RAM inside the Ray actor. It is written to disk only when
checkpointing saves `replay_buffer.pt`.

### Async Trajectory Collector

`nemo_rl/algorithms/async_utils.py::AsyncTrajectoryCollector` is another Ray
actor. It owns the background collection loop and the generation-side scheduling
logic.

Its job is to:

- iterate the train dataloader
- choose which target training step needs trajectories
- slice a prompt from the dataloader batch
- repeat it `num_generations_per_prompt` times
- run the async rollout path
- push the completed prompt group into `ReplayBuffer`

The collector limits concurrent prompt-group workers with a semaphore sized
roughly as:

```text
num_prompts_per_step * max_trajectory_age_steps
```

### Rollout Path

For normal environments, the collector calls:

```text
run_async_multi_turn_rollout()
```

from `nemo_rl/experience/rollouts.py`.

That path calls:

```text
generate_responses_async() -> policy_generation.generate_async()
```

`VllmGeneration.generate_async()` yields per-sample outputs as soon as vLLM
finishes them. The rollout code then appends assistant messages, calculates
environment rewards, tracks termination/truncation, and returns a final batch
plus rollout metrics.

For NeMo Gym, the collector calls `run_async_nemo_gym_rollout()` instead. NeMo
Gym is optional. Async GRPO does not require NeMo Gym.

## One Training Step

At a high level, one iteration of `async_grpo_train()` does this:

1. Sample `num_prompts_per_step` prompt groups from `ReplayBuffer`.
2. Concatenate those prompt groups into one train batch.
3. Extract the buffered `total_reward`.
4. Convert message logs into flat token tensors.
5. Recompute `prev_logprobs` with the training policy.
6. Optionally compute reference-policy logprobs.
7. Compute GRPO or Reinforce++ advantages.
8. Train the policy with `ClippedPGLossFn`.
9. Pause new generation starts.
10. Refit vLLM generation weights from the policy.
11. Increment `weight_version`.
12. Notify the collector and resume generation.
13. Log metrics, validate if needed, checkpoint if needed, and increment `step`.

## Refit

Refit is the process of updating the generation engine with the latest training
policy weights.

In async GRPO, refit normally happens after every successful `policy.train()`
step. The driver calls:

```text
trajectory_collector.prepare_for_refit()
refit_policy_generation(policy, policy_generation, colocated_inference)
weight_version += 1
trajectory_collector.set_weight_version(weight_version)
trajectory_collector.resume_after_refit()
```

There is also an initial refit before training if the generation engine is
stale.

For async GRPO, inference is non-colocated. The refit path uses collective
communication:

```text
policy.offload_before_refit()
policy.broadcast_weights_for_collective()
policy_generation.update_weights_from_collective()
policy.prepare_for_training()
```

`prepare_for_refit()` pauses new generation starts. If
`in_flight_weight_updates: false`, it waits for pending generations to complete
before refit. If `in_flight_weight_updates: true`, it allows ongoing generations
to finish while the generation engine receives updated weights. New generations
after refit use the updated weights.

If `recompute_kv_cache_after_weight_updates: true`, the collector asks the
generation engine to invalidate vLLM prefix/KV caches after the weight update.

## Weight Versions

Async GRPO tracks two versions for every replay-buffer entry:

- `trajectory_version`: the weight version used by vLLM to generate the tokens
- `target_weight_version`: the training weight version that should consume the
  trajectory

The trainer's `weight_version` starts at the restored training step. After every
refit it increments by one.

The buffer only samples entries intended for the current training version:

```python
valid_indices = [
    i for i, gen_version in enumerate(trajectory_versions)
    if current_weight_version - max_age <= gen_version <= current_weight_version
]

intended_indices = [
    i for i in valid_indices
    if target_weight_versions[i] == current_weight_version
]

selected = intended_indices[:num_prompt_groups]
```

This is FIFO among entries that match the current target. The implementation
does not randomly sample and does not use LIFO.

## Replay Window and Cleanup

Replay is bounded by `grpo.async_grpo.max_trajectory_age_steps`.

Sampling accepts only trajectories whose generation version is in:

```text
[current_weight_version - max_trajectory_age_steps, current_weight_version]
```

The buffer capacity is computed as:

```text
num_prompts_per_step * max_trajectory_age_steps * 2
```

The trailing factor is slack for late arrivals.

When entries are sampled, they are removed from the lists immediately. The
sampled indices are deleted in reverse index order to avoid list index shifts
during deletion. That reverse order is a deletion detail, not a sampling policy.

The buffer also evicts entries that are older than the age window during
sampling.

## FIFO, LIFO, and Long-Generation Bias

FIFO is deterministic and avoids preferentially taking the latest-arriving
trajectories. That matters because latest-arriving does not necessarily mean
newer policy weights. It often means longer generations, slower environments, or
multi-turn interactions.

However, long-generation bias is a real concern in async GRPO. A long trajectory
can arrive after its target step has already been consumed, or after its
generation version falls outside the age window. In that case it can be wasted
or evicted as stale.

The current implementation mitigates this with:

- target-specific sampling via `target_weight_version`
- stalling if the current target does not have enough prompt groups
- advancing `last_target_weight_already_generated` only when a full batch is
  consumed
- buffer starvation diagnostics such as `trajectory_duration_s`,
  `max_gen_tokens_per_turn`, and `turns_per_sample`
- `max_trajectory_age_steps` to allow older trajectories to remain valid
- `in_flight_weight_updates` to reduce refit stalls

It does not fully remove length bias. If long rollouts are common, watch p95/max
trajectory duration, stale eviction, buffer starvation, and average trajectory
age. Consider reducing max generation length, increasing generation capacity,
enabling in-flight updates, or increasing `max_trajectory_age_steps` if training
is frequently waiting on late trajectories.

## Logprobs

Async GRPO uses three different policy distributions during a step:

```text
pi_gen   = policy that generated the rollout
pi_prev  = training policy before the current optimizer update
pi_curr  = differentiable policy being optimized in this train forward pass
```

The corresponding tensors are:

- `generation_logprobs`: logprobs recorded by vLLM during rollout generation.
  These come from `pi_gen`.
- `prev_logprobs`: logprobs recomputed by the training policy before this
  optimizer update. These come from `pi_prev`.
- `curr_logprobs`: logprobs computed inside `ClippedPGLossFn` from the current
  forward pass. These come from `pi_curr`.

In synchronous GRPO, `pi_gen` and `pi_prev` should be nearly identical. That is
why token multiplicative probability error, or TMPE, is mostly a parity check
between the generation and training engines.

In async GRPO, `pi_gen` may be older than `pi_prev`. TMPE still exists, but it
means something different: it measures generator/trainer distribution mismatch
for the current sampled batch. Low TMPE means the async data is close to
on-policy. High TMPE means the batch is more off-policy and the
importance-sampling correction is doing more work.

The loss computes TMPE from:

```text
abs(generation_logprobs - prev_logprobs)
```

The async functional test checks `train/token_mult_prob_error`, so this metric is
part of the async path too.

## Why Importance Sampling Is Needed

Normal GRPO/PPO uses the ratio:

```text
ratio = pi_curr / pi_prev
```

This assumes the samples were drawn from `pi_prev`. That assumption is valid in
sync GRPO, up to numerical and engine-parity differences.

In async GRPO, samples are drawn from `pi_gen`, which may be one or more weight
versions behind. If the loss only used `pi_curr / pi_prev`, the expectation would
be taken over samples from the wrong behavior policy.

The async loss therefore multiplies the clipped GRPO objective by:

```text
is_weight = pi_prev / pi_gen
```

In logprob form:

```text
is_weight = exp(prev_logprobs - generation_logprobs)
```

For the unclipped term, the product simplifies algebraically:

```text
exp(curr - prev) * exp(prev - gen) = exp(curr - gen)
```

But the implemented loss is not equivalent to simply replacing the denominator
with `pi_gen`, because the normal GRPO ratio is clipped first:

```text
clip_loss = max(
  -advantage * exp(curr - prev),
  -advantage * clip(exp(curr - prev), 1 - eps_low, 1 + eps_high)
)

actor_loss = mean(is_weight * clip_loss)
```

`pi_prev` still matters because it defines the trust region for the current
optimizer step. `pi_gen` corrects for the data-collection distribution.

## Token-Level and Sequence-Level IS

By default, importance sampling is token-level:

```text
exp(prev_logprobs_t - generation_logprobs_t)
```

The loss also supports sequence-level importance ratios via:

```yaml
loss_fn:
  sequence_level_importance_ratios: true
  token_level_loss: false
```

In that mode, the code computes a sequence-level weight and broadcasts it back
over tokens. Sequence-level IS is associated with GSPO-style objectives and is
not the default async GRPO configuration.

## Truncated IS, Filtering, and Rejection

Large IS weights can destabilize training. The default template leaves
truncation off:

```yaml
loss_fn:
  truncated_importance_sampling_ratio: null
```

But the code supports several stabilization modes:

- `tis`: clamp IS weights to `[min, max]`
- `icepop`: zero out tokens whose IS weights fall outside `[min, max]`
- `seq-mask-tis`: mask entire sequences using a geometric-mean IS ratio

Example:

```yaml
loss_fn:
  use_importance_sampling_correction: true
  truncated_importance_sampling_ratio: 5
  truncated_importance_sampling_ratio_min: 0.2
  truncated_importance_sampling_type: tis
```

Several larger async recipes in `examples/configs/` use
`truncated_importance_sampling_ratio: 5` with `truncated_importance_sampling_type:
tis`. Some also set:

```yaml
grpo:
  seq_logprob_error_threshold: 2
```

That threshold is not IS clipping. It is a separate sequence-level masking
mechanism based on multiplicative logprob error. It can reject or mask sequences
whose generation/training mismatch is too large.

## Utilization Tuning

There is no universal train-to-generation GPU ratio. The right allocation
depends on:

- model size
- tensor/pipeline/expert parallelism
- prompt length
- generation length
- number of generations per prompt
- environment latency
- reward model latency
- refit time
- validation frequency

The goal is for the generation side to produce about
`num_prompts_per_step` prompt groups per training step while keeping average
trajectory age low.

Useful metrics:

- `idle/buffer_starvation`: trainer is waiting for replay entries
- `idle/generation_limit_pause`: collector has filled all allowed targets and is
  waiting for training/refit to advance the version window
- `idle/refit_bubble`: exposed pause around refit
- `weight_sync`: actual weight-transfer time
- `buffer_size`
- `avg_trajectory_age`
- `trajectory_duration_s/max`
- `trajectory_duration_s/p95`
- `max_gen_tokens_per_turn/*`
- `turns_per_sample/*`
- `train/token_mult_prob_error`
- `train/gen_kl_error`
- `performance/valid_tokens_per_sec_per_gpu`

If `idle/buffer_starvation` is high, generation is not keeping up. Consider more
generation GPUs, lower max generation length, faster environments, fewer
generations per prompt, in-flight weight updates, or a larger age window.

If `idle/generation_limit_pause` is high, generation is ahead of training.
Consider fewer generation GPUs, larger train capacity, reducing refit overhead,
or increasing the allowed target window only if higher off-policy age is
acceptable.

If `weight_sync` or `idle/refit_bubble` dominates, optimize refit before adding
more rollout capacity.

## Dataloaders

There is one train dataloader for prompts. In async GRPO, the collector owns and
iterates that dataloader.

The trainer does not iterate a second dataloader for training samples. It samples
from `ReplayBuffer`.

There can also be a validation dataloader, which is separate and used by
`validate()`.

## When Generation Can Be Idle

The generation cluster is intended to stay busy, but it can still idle in these
cases:

- refit is in progress and new generation starts are paused
- pending generations are being waited on because `in_flight_weight_updates` is
  false
- validation pauses the collector
- all target versions in the current age window are already complete or in
  progress
- replay buffer is full
- the in-flight worker semaphore is saturated
- dataloader exhaustion or slow dataloader workers
- slow or blocked environments

Async GRPO reduces idle time compared with sync GRPO, but it does not eliminate
all scheduling bubbles.

## Checkpointing and `replay_buffer.pt`

The replay buffer is stored in memory during training. During checkpointing, the
driver saves:

```text
replay_buffer.pt
```

This file can be large because each replay entry may contain:

- long message logs
- generated token ids
- generation logprobs
- reward and environment metadata
- multiple completions per prompt
- multi-turn histories

The buffer is bounded by the configured capacity, but a single entry can still
be heavy for long rollouts or large `num_generations_per_prompt`.

## Common Questions

### Does refit happen after every weight update?

Yes. In the normal async path, after each `policy.train()` call the driver
refits the generation engine, increments `weight_version`, and tells the
collector to resume with the new version.

### Does async GRPO require NeMo Gym?

No. NeMo Gym is one optional rollout backend. Normal async rollouts use
`run_async_multi_turn_rollout()`.

### Does replay randomly sample?

No. It filters to valid entries for the current training version, then selects
FIFO among those entries.

### Why not LIFO?

LIFO would prefer latest-arriving entries, but latest arrival is often a signal
of long generation or slow environment, not newer weights. That could introduce
length or environment-latency bias. The implementation instead uses version
targeting and an age window to control off-policy staleness.

### What happens after a replay entry is used?

It is removed from the lists immediately. Used prompt groups are not reused.

### How far back can replay go?

At most `max_trajectory_age_steps` training versions, subject to buffer capacity
and target-version matching.

### Is IS done in addition to the normal GRPO ratio?

Yes. The normal clipped GRPO ratio is `pi_curr / pi_prev`. The async
importance-sampling correction is `pi_prev / pi_gen`. The correction multiplies
the clipped GRPO loss.

### Is IS clamped by default?

Not in the base template. Large async recipes often enable TIS or sequence
masking. For unstable runs, consider enabling truncation and monitoring
`is_oob_ratio`, TMPE, and sequence logprob-error metrics.

### Is generation different between sync and async GRPO?

The vLLM decoding behavior should be the same for the same generation config.
The scheduling is different. Sync GRPO generates inline, then trains. Async GRPO
keeps a background collector running, stores rollouts in replay, and refits vLLM
when new weights are ready.

### What should I inspect first when debugging?

Start with:

- `train/token_mult_prob_error`
- `train/gen_kl_error`
- `train/avg_trajectory_age`
- `train/buffer_size`
- `timing/train/idle/buffer_starvation`
- `timing/train/idle/generation_limit_pause`
- `timing/train/weight_sync`
- rollout duration and generation length metrics

Then inspect whether the issue is generation throughput, training throughput,
off-policy drift, refit overhead, or environment latency.
