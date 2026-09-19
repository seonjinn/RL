# Refit-aware vLLM sleep for colocated training

## Status

Proposed for a standalone NeMo-RL pull request based on `main`.

## Problem

Sync colocated RL alternates the same GPUs between vLLM generation and policy
training. NeMo-RL currently ends each generation phase with vLLM sleep level 1.
That mode copies every vLLM weight to host memory before releasing its GPU
allocation. A full refit then wakes empty GPU allocations and immediately
overwrites those weights with the latest policy weights.

The host copy is therefore unnecessary when the next refit can reconstruct every
runtime weight. It also scales poorly. In a 64-GPU Qwen3-235B run, each vLLM rank
backed up about 66 GiB while colocated training also moved optimizer and gradient
state to host memory. Host pressure produced long-tail
`prepare_for_generation` stalls even though the measured transfer itself stayed
near nine seconds.

This is a lifecycle problem, not an MXFP8 conversion or CUDA IPC bandwidth
problem. BF16 and MXFP8 rollouts use the same sleep path, although large MXFP8
runs exposed the tail more often.

## Goals

- Avoid host backup of stale vLLM weights before a guaranteed full refit.
- Preserve the current level-1 behavior whenever complete reconstruction is not
  proven.
- Keep the decision automatic. Do not add a recipe flag that users must set
  correctly.
- Fail the run immediately if refit fails after weights were discarded.
- Support synchronous and asynchronous vLLM engine wrappers for the colocated
  Sync RL lifecycle.
- Measure correctness, host memory, refit latency, and end-to-end performance on
  GB200.

## Non-goals

- Changing non-colocated Async-1off or NCCL Reshard lifecycle behavior.
- Replacing or combining the existing optimizer CPU-buffer reuse experiment.
- Making sparse, delta, or partial refit eligible for weight discard.
- Reworking vLLM's allocator or sleep implementation.
- Adding model-name-specific rules.

## Existing lifecycle

The current colocated IPC path is:

1. `finish_generation()` resets caches and calls vLLM sleep level 1.
2. vLLM copies weight allocations to host memory and frees GPU allocations.
3. `IPCWeightSynchronizer.sync_weights()` offloads policy state.
4. Generation wakes the `weights` allocation.
5. Policy weights replace the generation weights through CUDA IPC/ZMQ.
6. Policy state is restored and generation wakes the KV cache.

Step 2 preserves stale values that step 5 replaces. The optimization removes
only that redundant preservation.

## Design

### 1. Represent reconstruction as a capability

Add a read-only capability to the weight-synchronizer boundary that answers one
question: can the next successful sync reconstruct every runtime weight that
would be discarded?

The default is `False`. The standard colocated IPC synchronizer may return
`True` only after one real refit has completed and every generation worker has
attested that the update reconstructed its complete runtime parameter set.
Metadata initialization alone is not proof: it describes the source stream but
does not show which destination parameters vLLM actually loaded. Other
synchronizers keep the default unless they gain equivalent proof in a later
change. The first refit therefore always follows the existing level-1 path.

Each worker builds its attestation from three independent checks:

1. `_IPCWeightManifest.require_complete()` proves that every tensor declared by
   the policy-side refit metadata arrived.
2. The vLLM loader reports the destination names that it actually consumed.
   NeMo-RL preserves this return value through both the normal and FP8 loaders
   instead of discarding it.
3. Native reload finalization completes successfully. Parameters owned by a
   quantization method that rebuilds derived runtime storage are counted only
   after that finalizer succeeds.

The worker canonicalizes loader-reported names through the same realized
loader/refit ownership mapping used by the update, then compares the loaded or
finalized destination names with its `named_parameters()` set. Missing, empty,
or unreported loader results fail closed. The synchronizer enables discard only
when every live model owner returns an explicit `True` result. Empty RPC result
lists, `None`, and partial worker responses are failures, not successful
vacuous checks.

Coverage is based on realized runtime parameters, not model names or global
precision. Ignored BF16 layers inside an MXFP8 model remain eligible when their
logical weights are present in the full refit map. The following cases are not
eligible in this change:

- sparse or delta refit;
- any realized MTP module, including a co-trained one;
- any external speculative drafter;
- any missing, ambiguous, or unsupported runtime parameter;
- refit metadata that has not completed initialization.

This initial change preserves weights whenever a realized MTP or external
drafter exists. A later change may make a co-trained drafter eligible after its
runtime ownership is included in the same coverage proof. Unknown cases choose
preservation, never discard.

The attestation is invalidated when refit metadata or worker membership changes,
when a new sync starts, when workers restart, or when any update/finalization
step raises. A later successful complete refit may establish it again.

### 2. Pass semantic intent to vLLM sleep

Reconstruction capability alone is insufficient. `finish_generation()` is also
used after initialization, validation, and checkpoint-related phases where a
refit is not guaranteed before the next wake. Discarding in those calls would
lose the only valid copy of the rollout weights.

The caller therefore passes a semantic next-phase intent. The generation
interface exposes a backend-neutral helper whose default delegates to the
existing `finish_generation()` method. vLLM overrides the helper to select its
sleep behavior; Megatron generation does not receive a vLLM-specific keyword.
Discard is selected only when both conditions hold:

1. the caller states that policy training followed by a full refit is the only
   path back to generation; and
2. the synchronizer reports complete reconstruction capability.

The default intent is preserve. Initialization, validation, checkpoint, and
unknown callers require no changes and remain on level 1. The backend-neutral
generation API receives a semantic enum, not a raw vLLM sleep-level integer:
`PRESERVE` or `TRAIN_THEN_FULL_REFIT`. After vLLM combines that intent with the
coverage capability, its private worker RPC receives only the resulting
`discard_weights` boolean.

- Preserve: call vLLM sleep level 1, matching current behavior.
- Discard before full refit: call vLLM sleep level 2.

Both paths reset prefix and multimodal caches exactly as they do today. Both the
synchronous and asynchronous vLLM worker wrappers implement the same mapping.
Non-colocated generation does not sleep and is unchanged.

The Sync GRPO training-rollout boundary supplies the destructive intent only
when the current rollout is guaranteed to proceed to policy training. Dynamic
sampling can request another rollout before training, so it conservatively uses
preserve in this change. The validation and setup boundaries also preserve.

The public `GenerationInterface` remains backend-neutral. Backends other than
vLLM use the default helper and keep their current behavior.

### 3. Treat discarded weights as a transaction

Once weights have been discarded, generation is unusable until a complete refit
succeeds. The synchronizer records that actual state before dispatching the
destructive sleep RPC; it does not infer it later from requested intent. IPC
refit then enforces this sequence:

1. Mark the generation weights stale when a sync starts and invalidate the old
   reconstruction attestation.
2. Require explicit success from weight allocation and every worker update.
3. Restore the policy side in bounded cleanup regardless of success.
4. Wake the generation KV cache only after transfer, native finalization, and
   the new coverage attestation all succeed.
5. Clear the discarded/stale state only after that KV wake also succeeds.
6. Raise on a false, empty, partial, or exceptional worker result. Do not
   convert the failure to a log message plus `False` that callers may ignore.

This change does not attempt in-process recovery from a failed destructive
refit. The safe response is to terminate the run so an orchestrator can restart
from a known checkpoint.

### 4. Keep optimizer buffer reuse separate

Reusable policy optimizer offload buffers reduce repeated host allocation and
copy overhead, but they do not remove vLLM's model-sized host backup. That work
stays in a separate PR so each mechanism can be reviewed and measured alone.

The benchmark integration branch will test:

- current level-1 sleep;
- discard-aware sleep only;
- discard-aware sleep plus optimizer-buffer reuse.

## Code boundaries

Expected production changes are limited to:

- `nemo_rl/models/generation/interfaces.py`: define the backend-neutral
  next-phase intent and finish helper;
- `nemo_rl/weight_sync/interfaces.py`: conservative reconstruction capability;
- `nemo_rl/weight_sync/ipc_weight_synchronizer.py`: full-refit capability and
  transactional failure handling;
- `nemo_rl/models/generation/vllm/vllm_generation.py`: select semantic sleep
  intent;
- `nemo_rl/models/generation/vllm/vllm_backend.py` and
  `nemo_rl/models/generation/vllm/quantization/fp8.py`: preserve loader results
  and produce the post-refit runtime-coverage attestation;
- `nemo_rl/models/generation/vllm/vllm_worker.py`: map intent to vLLM sleep
  level for the synchronous engine;
- `nemo_rl/models/generation/vllm/vllm_worker_async.py`: the same mapping for
  the asynchronous engine;
- `nemo_rl/algorithms/grpo.py`: mark the standard Sync GRPO training-rollout
  boundary when no intermediate rollout can occur;
- `nemo_rl/algorithms/grpo_sync.py` and
  `nemo_rl/experience/sync_rollout_actor.py`: carry the same intent through the
  TQ rollout actor;
- focused unit tests under `tests/unit/weight_sync/` and
  `tests/unit/models/generation/`, plus GRPO and rollout-actor tests.

If runtime coverage cannot be proven from existing refit metadata without a
large refactor, this PR will initially enable discard only for the standard full
IPC refit contract and explicitly reject known partial/drafter cases. It will
not guess from parameter-name patterns.

## Tests

### Unit tests

- Default and incomplete coverage select level 1.
- A first refit preserves weights and establishes coverage only after success.
- A later verified full IPC refit selects level 2.
- Full coverage without explicit next-phase intent remains on level 1.
- Explicit intent without full coverage remains on level 1.
- Synchronous and asynchronous worker wrappers map the semantic intent to the
  same vLLM levels.
- BF16 rollout, MXFP8 rollout, and mixed BF16/MXFP8 layers do not affect the
  decision when coverage is complete.
- Static MTP, external drafter, sparse refit, missing metadata, and partial
  coverage fall back to level 1.
- Co-trained MTP also falls back to level 1 in this initial change.
- Empty, `None`, and partial RPC results cannot establish coverage.
- Metadata replacement, sync entry, worker restart, and failed finalization
  invalidate previously established coverage.
- A failed weight wake or transfer raises, restores policy state, does not wake
  the KV cache, and leaves generation stale.
- A successful transfer restores policy state and wakes the KV cache once.
- Setup, validation, checkpoint, and dynamic-sampling boundaries preserve
  weights; a standard training-rollout boundary requests discard.

### Blackwell integration correctness

On GB200, extend the existing real MoE Megatron/vLLM integration to refit
distinctive weights A to B and then B to C. After each refit:

- compare the final packed runtime weights with a freshly initialized model at
  the same B or C state where the backend exposes them;
- compare fixed-input logits with the fresh model using a backend-appropriate
  numerical tolerance;
- verify that A differs from B and B differs from C, so the test cannot pass by
  retaining stale weights;
- cover BF16 FlashInfer TRTLLM and MXFP8 routed-expert paths;
- include one mixed-precision model with BF16 first/last layers.

The integration must run on Blackwell because the packed FlashInfer TRTLLM
layout is the behavior under test.

## Performance validation

Use the current GB200 performance recipes unchanged except for rollout
precision and the candidate lifecycle change. Run 20 optimizer steps and report
the mean and median of steps 2 through 19.

### Qwen3-235B Sync

Compare:

1. BF16 training + BF16 rollout, current level-1 sleep;
2. BF16 training + MXFP8 rollout, current level-1 sleep;
3. BF16 training + MXFP8 rollout, discard-aware sleep;
4. configuration 3 plus the separate optimizer-buffer reuse change.

### Nemotron3 Super Sync

Run the same four arms with the established Sync performance recipe. If a
level-1 arm cannot complete because of host memory pressure, record the failure,
peak host memory, and last completed phase rather than changing global batch
size or parallelism only for that arm.

For every completed arm, report:

- E2E, generation, policy training, logprob, and refit time;
- E2E and generation tokens/s/GPU;
- per-node peak host RSS;
- refit mean, median, p95, and maximum;
- `gen_kl_error`, reward, and entropy over the same steady-state window;
- W&B run link and exact commit SHA.

## Acceptance criteria

- Eligible full-refit runs make no vLLM model-parameter backup during sleep.
  vLLM may still preserve non-parameter named buffers required by level 2.
- No path selects discard unless both full-refit coverage and explicit
  next-phase intent are present.
- Ineligible configurations retain level-1 behavior without user action.
- Refit failure after discard terminates promptly and cannot resume generation.
- A-to-B-to-C integration results match freshly initialized B/C models within
  the declared backend tolerance.
- Qwen3-235B completes 20 steps without the prior host-memory refit tail.
- Nemotron3 Super either completes the unchanged Sync recipe or produces a
  bounded, attributable non-lifecycle failure.
- No regression appears in BF16/MXFP8 KL, reward, or entropy relative to the
  matching level-1 arm.

## Rollout plan

The feature is automatic but conservative. It ships enabled only for verified
full colocated IPC refit. Logs state whether weights were preserved or discarded
and why. Any unsupported or uncertain configuration uses level 1, making the
failure mode lower performance rather than lost weights.
