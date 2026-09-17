# SpecDec-Aware Refit Memory Lifecycle

## Objective

Eliminate the Qwen3-235B colocated-refit host-memory pressure and long-tail
`prepare_for_generation` stalls without changing the accuracy, execution
semantics, or performance of runs that do not use speculative decoding.

The immediate target is NeMo-RL with the pinned vLLM 0.25.1 environment used by
the existing Qwen3-235B performance recipe. The design must also support frozen
DFlash and DSpark drafters and provide an explicit extension point for online
drafter refit.

## Observed Failure

The current vLLM worker always calls `sleep(level=1)` after generation. Level 1
backs model weights in host memory and discards the KV cache. On Qwen3-235B,
the colocated policy actors, vLLM target weights, speculative drafter, and CUDA
Graph allocations together push host memory to the Ray threshold. Raising the
threshold from 95% to 98% delays the failure but can end in a Slurm cgroup OOM.

The measured DFlash K7 long-tail step spent 1,766.96 seconds in the aggregate
`prepare_for_generation` timer, while target-weight transfer consumed only
3.54 seconds. The missing time is therefore in the surrounding policy offload
or vLLM wake operations, not in the ZMQ weight stream.

The current timer hierarchy cannot identify the exact subphase. In addition,
vLLM level-2 sleep cannot be enabled blindly: it discards both target and draft
weights, while the existing target refit stream reloads only target weights.

## Non-Goals

- Changing speculative decoding algorithms or acceptance logic.
- Changing the no-SpecDec baseline recipe, scheduling, parallelism, or CUDA
  Graph policy.
- Changing policy optimization, rewards, KL computation, or sampled outputs.
- Making level-2 sleep the global default.
- Hiding memory pressure by increasing Ray's memory-kill threshold.

## Compatibility Contract

The existing lifecycle remains the default. A run enters the new lifecycle
only when all of the following are true:

1. vLLM generation is colocated.
2. speculative decoding is configured.
3. the new refit memory-lifecycle mode is explicitly enabled.
4. the drafter implementation advertises a supported restore or refit path.

A no-SpecDec baseline must not invoke draft snapshot, draft restore, or any new
remote procedure call. Its sleep level, wake order, target transfer, generated
tokens, policy/logprob calculations, and recipe defaults remain unchanged.

Unsupported combinations fail during setup with an actionable error. They must
not fall back silently to an uninitialized or stale drafter.

## Alternatives

### Keep level-1 sleep and tune memory

Reducing CUDA Graph buckets, `max_num_seqs`, or GPU memory utilization can
lower pressure but does not remove the target-weight CPU backup. It also risks
changing the benchmark workload. This remains a diagnostic tool, not the
primary fix.

### Use non-colocated generation

Separating policy and generation workers avoids the sleep/wake lifecycle and is
a reliable operational fallback. It consumes additional GPUs and changes the
official recipe topology, so it is not the primary performance comparison.

### SpecDec-aware deep refit

Discard the stale target weights, preserve or explicitly update the much
smaller drafter, stream the new target weights, and then restore the KV cache.
This directly removes the large host backup while preserving the colocated
recipe. This is the selected design.

## Configuration

Add a typed refit memory-lifecycle block under the existing `refit_cfg` schema.
The schema owns all defaults; call sites must not invent fallback values.

Conceptual shape:

```yaml
policy:
  generation:
    refit_cfg:
      memory_lifecycle:
        mode: legacy_level1
```

Supported modes:

- `legacy_level1`: current behavior and default.
- `specdec_deep_refit`: target weights and KV cache are discarded; draft
  weights are restored or updated explicitly before generation resumes.

The Qwen3-235B SpecDec experiment recipe opts into `specdec_deep_refit`.
No-SpecDec recipes keep the default and require no edits.

## Runtime Lifecycle

### Initialization

1. Detect whether speculative decoding is configured.
2. Resolve the drafter restore capability.
3. For a frozen drafter, create one CPU-resident snapshot of the drafter state.
4. For an online drafter, register the existing drafter-refit provider instead
   of creating a frozen snapshot.
5. Validate that every draft parameter and required buffer has an owner.

### Refit transaction

1. Quiesce generation and reset weight-dependent caches.
2. Offload the policy optimizer as required by the existing policy backend.
3. Put vLLM into level-2 sleep for the opt-in deep-refit path.
4. Wake only the weight allocations.
5. Stream and install the updated target-policy weights.
6. Restore frozen drafter weights from the CPU snapshot, or install online
   drafter weights through the registered refit provider.
7. Validate draft state completeness.
8. Complete the existing policy offload transition.
9. Wake the KV cache.
10. Resume generation only after all participants acknowledge completion.

Any failure leaves generation unavailable and raises an error. The engine must
never serve with missing, stale, or uninitialized drafter weights.

### Legacy transaction

The legacy and no-SpecDec paths retain the current sequence:

1. Policy offload before refit.
2. vLLM level-1 wake for weights.
3. Target-policy weight transfer.
4. Policy offload after refit.
5. vLLM KV-cache wake.

No draft-specific work is added to this branch.

## Timing and Memory Diagnostics

Split the aggregate refit timer into stable child metrics:

- `prepare_for_generation/policy_offload_before_refit`
- `prepare_for_generation/vllm_wake_weights`
- `prepare_for_generation/transfer_and_update_weights`
- `prepare_for_generation/drafter_restore_or_refit`
- `prepare_for_generation/policy_offload_after_refit`
- `prepare_for_generation/vllm_wake_kv_cache`

Record process RSS, node-available memory, and CUDA allocated/reserved bytes at
the boundaries of these phases. Memory diagnostics must be bounded and must not
scan the node or scheduler.

The existing parent timer remains for dashboard compatibility.

## Policy-Offload Cleanup

`offload_after_refit()` currently calls `offload_before_refit()` again after
moving the model to CPU. The first implementation does not remove this behavior
without evidence. The new timers determine whether the second invocation is
material. If it is redundant, a separate change makes the operation idempotent
and tests optimizer/model placement before and after each transition.

## Accuracy and Correctness Gates

### Unit tests

- The default configuration resolves to `legacy_level1`.
- A no-SpecDec worker still calls level-1 sleep and makes no draft RPC.
- Deep refit fails at setup when no supported drafter restore provider exists.
- Frozen-drafter deep refit follows the exact order: sleep, wake weights,
  target refit, draft restore, wake KV.
- Online-drafter deep refit uses the online provider and never restores the
  frozen snapshot.
- A partial draft restore prevents generation from resuming.
- Timer scopes cover each operation and retain the existing parent metric.

### GPU gate

Run an isolated three-step Qwen3-235B DFlash K7 job before any 20-step run.
The gate requires:

- no Ray memory kill or Slurm cgroup OOM;
- stable host-memory usage across repeated refit cycles;
- nonzero acceptance after every deep-refit wake;
- generated output and target logprobs consistent with a cold-loaded engine;
- no NaN or discontinuity in reward, `policy_kl_error`, `gen_kl_error`, or
  approximate entropy;
- complete child-timer coverage of the parent refit duration.

### Baseline non-regression gate

Run the no-SpecDec baseline before and after the patch with the same image,
recipe, seed, data, and allocation. The baseline must remain on the legacy
path. Generated token counts and accuracy metrics must retain their expected
run-to-run behavior, and no new draft-related log or RPC may appear.

Performance is evaluated with time-weighted step totals over the matched
window. The patch is accepted only if any baseline E2E difference is within
normal rerun variance and the code path adds no remote operation. A regression
outside the observed baseline variance blocks the SpecDec performance run.

## Performance Gates

After the three-step correctness gate passes:

1. Run a 20-step no-SpecDec baseline.
2. Run DFlash K5/K7 and DSpark K5/K7 with the same target recipe.
3. Compare Steps 3-20 using summed wall times.
4. Report generation speedup, refit subphase times, E2E speedup, host-memory
   peak, acceptance length/rate, reward, entropy, and KL metrics.

The fix succeeds when refit no longer has memory-pressure long tails, all
quality gates pass, and generation savings translate into an E2E improvement
or a clearly accounted-for residual bottleneck.

## Rollout and Fallback

The new mode is opt-in and isolated in a dedicated worktree. If the pinned vLLM
version cannot safely expose or restore DFlash/DSpark draft state, fail the GPU
gate and use non-colocated generation as the temporary benchmark fallback.
Do not switch the global default or publish performance claims from a run whose
drafter state was not verified after every wake.

