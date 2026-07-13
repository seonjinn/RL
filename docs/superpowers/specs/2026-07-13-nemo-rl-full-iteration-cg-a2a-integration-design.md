# NeMo-RL Full-Iteration CUDA Graph and A2A Integration Design

**Date:** 2026-07-13
**Base:** `97fdbdaac99787ba36eec79eab8869b28d138483`
**Implementation branch:** `sna/nemo-2606-full-cg-a2a-integration-20260713`

## Objective

Integrate the existing fixed-shape full-iteration CUDA Graph implementation into the current NeMo-RL CuTeDSL and expert-parallel A2A branch without regressing either feature. Validate the graph first around synchronous PolicyTraining, then in a non-colocated GRPO topology whose generation and weight synchronization preserve captured policy storage. Keep true colocated generation/refit outside the first implementation because its CPU offload lifecycle reallocates graph-captured model, gradient, and optimizer storage on every RL update.

The implementation must produce independently attributable measurements for:

- CuTeDSL only;
- CuTeDSL plus A2A overlap;
- CuTeDSL plus full-iteration CUDA Graph; and
- CuTeDSL plus full-iteration CUDA Graph plus A2A overlap.

## Current State

The current branch already propagates these A2A settings into Megatron Core:

- `overlap_moe_expert_parallel_comm=true`;
- `high_priority_a2a_comm_stream=true`; and
- `delay_wgrad_compute=true`.

It also provides the `return_schedule_plan` adapter required by Megatron Core's combined-1F1B scheduler. The official Qwen3-30B-A3B A2A OFF/ON performance cohort is running separately at the immutable base SHA, so this implementation branch must not advance or rewrite the source branch used by those jobs.

The existing full-CG implementation is available in commits `5ee358abb` and `690fc74da`. It captures the Megatron forward/backward schedule for synchronous PolicyTraining. It does not capture optimizer or scheduler steps, and it explicitly rejects forward-only Logprob, evaluation, split/async training, colocated refit/offload, dynamic batching, sequence packing, and context parallelism greater than one.

## Architecture

### Stage 1: PolicyTraining-only graph integration

Semantically integrate the full-CG implementation instead of accepting a mechanical cherry-pick. The integration must retain the current A2A schedule-plan path in `train.py` and the current worker lifecycle telemetry.

The captured unit is the fixed-shape Megatron forward/backward schedule. NeMo-RL continues to execute gradient zeroing, optimizer step, scheduler step, and metric materialization eagerly outside the graph. Reports must describe this boundary explicitly; they must not call the implementation an optimizer-inclusive iteration graph.

Static graph inputs use stable device buffers. Before replay, NeMo-RL verifies:

- microbatch count, sequence length, microbatch size, and supported loss configuration;
- input tensor shapes, dtypes, and devices;
- parameter, gradient, and optimizer tensor storage addresses; and
- a fixed policy-training operation mode.

The implementation records capture, warm-up, and replay counters so remote validation can prove that replay happened rather than merely observing a generic CUDA graph kernel.

### Stage 2: Full-CG and A2A composition

When A2A is enabled, the graph-wrapped forward/backward function must preserve `return_schedule_plan=True` and build Megatron Core's schedule plan from the same graph-stable inputs and normalizers. A dedicated unit test must fail if either integration silently replaces the other.

The first GPU validation order is:

1. CuTeDSL plus full-CG, A2A disabled;
2. CuTeDSL plus full-CG plus A2A; and
3. eager equivalents for parity and timing comparison.

The MoE routing distribution may change between updates. Therefore changing router decisions must be exercised during capture and at least two replays. A successful static-input test alone is insufficient evidence for the combined MoE path.

### Stage 3: Non-colocated E2E GRPO

After isolated PolicyTraining replay succeeds, run GRPO with dedicated vLLM generation GPUs and resident policy state:

- `policy.generation.colocated.enabled=false`;
- vLLM generation rather than Megatron generation;
- `force_on_policy_ratio=true` to skip previous-policy Logprob;
- `grpo.skip_reference_policy_logprobs_calculation=true`;
- evaluation and KV-calibration paths disabled; and
- fixed shapes, sequence packing disabled, and context parallelism equal to one.

The non-colocated weight synchronizer broadcasts resident policy weights without policy storage offload. The graph remains limited to PolicyTraining; generation, weight transfer, reward computation, and outer-loop orchestration remain eager. This stage measures both PolicyTraining improvement and its effect on E2E step time, but it produces no Logprob speedup result because Logprob is intentionally skipped.

### Deferred: True colocated lifecycle

True colocated support is a separate design and implementation. Current IPC and HTTP synchronizers invoke policy offload around refit, and Logprob preparation moves gradient buffers to CPU. These operations invalidate captured addresses. Resetting and warming a graph every outer step is not acceptable because the target workload has one policy global batch per RL step and uses three graph warm-up iterations; it would never reach useful replay.

Colocated support therefore requires a graph-resident synchronization/offload protocol or a memory-proven resident-state mode. It must not be approximated by removing fail-closed guards.

## Code Boundaries

### Full-CG adapter

`nemo_rl/models/megatron/full_cuda_graph.py` owns:

- static microbatch storage;
- call and storage signatures;
- graph wrapper construction;
- capture/warm-up/replay counters;
- operation support validation; and
- explicit reset/invalidation interfaces for future lifecycle work.

### Megatron training integration

`nemo_rl/models/megatron/train.py` owns:

- graph-stable normalizer extraction;
- forward/backward function injection;
- preservation of `_build_post_processing_fn`; and
- composition with `return_schedule_plan` for A2A.

### Policy worker lifecycle

`nemo_rl/models/policy/workers/megatron_policy_worker.py` owns:

- graph construction for synchronous PolicyTraining;
- eager optimizer and scheduler execution;
- graph-safe tensor metric materialization;
- operation guards; and
- pointer validation before replay.

### Typed configuration

`nemo_rl/models/policy/__init__.py` and `nemo_rl/models/megatron/setup.py` expose and propagate:

- `cuda_graph_impl=full_iteration`;
- `cuda_graph_warmup_steps`;
- `cuda_graph_use_single_mempool`; and
- optional paged-stash controls already supported by the pinned Megatron Core.

A2A fields remain independent typed settings and must preserve upstream defaults when absent.

### Graph-safe losses

`nemo_rl/algorithms/loss/interfaces.py` and `nemo_rl/algorithms/loss/loss_functions.py` keep changing normalization values in graph-stable tensor inputs and materialize Python metrics only after replay. The first supported losses are exactly `ClippedPGLossFn` and `NLLLossFn`.

### Experiment harness

The existing colocated factorial launcher continues to reject full-CG contexts. A separate policy-only and non-colocated launcher records the different resource topology and operation exclusions. It must never relabel a colocated job as full-CG capable.

## Failure Handling

Configuration fails before GPU allocation when any of these conditions hold:

- dynamic batching or sequence packing is enabled;
- context parallel size is not one;
- an unsupported loss or operation is requested;
- graph input or storage signatures change;
- colocated generation/refit is requested by a Stage 1-3 launcher; or
- A2A and graph schedule-plan composition is unavailable.

Remote jobs retain bounded diagnostics and immutable source, image, cache, and resolved-config evidence. A configuration flag or `cudaGraphLaunch` alone is not accepted as proof of replay.

## Verification

### Unit and integration tests

The TDD suite must cover:

- all existing static-buffer, signature, loss, and setup contracts from the full-CG branch;
- eager and graph loss/update parity over multiple optimizer updates;
- capture once followed by at least two replays;
- parameter, gradient, and optimizer storage mismatch rejection;
- A2A `return_schedule_plan` composition with graph wrapping;
- graph-safe metric and auxiliary-loss scale handling;
- explicit rejection of Logprob, eval, async/split, packing, dynamic batching, CP greater than one, and colocated lifecycle operations; and
- resolved configs for isolated policy and non-colocated GRPO launchers.

### GB200 validation

Each GPU gate records exact source and dependency revisions, resolved configuration, graph counters, representative pointer signatures, Nsight evidence, and component timings.

Acceptance requires:

1. identical eager and replayed update outputs within the configured numerical tolerance;
2. one capture and at least two confirmed replays;
3. `cudaGraphLaunch` correlated with the PolicyTraining range;
4. CuTeDSL fused GLU, quantization, dgrad, and wgrad kernels inside the captured/replayed range;
5. NCCL A2A kernels temporally overlapping expert compute for the combined cell;
6. no graph recapture between resident-state PolicyTraining updates; and
7. three performance replicas with warm-up excluded and identical workload controls.

Performance reports separate E2E, generation, refit/weight transfer, policy/reference Logprob, and PolicyTraining timings. Unsupported or skipped components are marked not applicable rather than assigned a zero speedup.

## Implementation Strategy

Use TDD and semantic integration:

1. transplant the new full-CG adapter and its tests;
2. reproduce failures against the current branch;
3. manually compose overlapping `train.py`, setup, worker, loss, and config changes;
4. add A2A composition and pointer-lifecycle tests;
5. run focused and broader NeMo-RL suites;
6. obtain independent task review;
7. push only the dedicated feature branch; and
8. run isolated PolicyTraining, combined A2A, and non-colocated GRPO gates in that order.

Do not merge or advance the immutable branch used by the active Pre-Tyche A2A cohort.
