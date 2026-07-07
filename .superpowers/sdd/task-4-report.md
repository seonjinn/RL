# Task 4: NeMo-RL Evaluation Fast Path

## Result

- Base commit: `69c0a7110eb7d71fd57f8eca179f1a51e521c2c2`
- Branch: `sna/sft-validation-opt-20260706`
- Added `policy.megatron_cfg.eval_mode_fast_path`, with absent or `false`
  preserving the legacy path. The Super V3 prepacked exemplar sets it to
  `false`.
- Preserved the public `Policy.train(...)` signature. Timed Megatron evaluation
  adds an internal worker keyword only; training and DTensor worker calls retain
  their previous kwargs.

## Exact Fast-Path Skips

For normal (non-MXFP8-shared-buffer) forward-only evaluation, the enabled fast
path skips:

- `model.zero_grad_buffer()`
- `optimizer.zero_grad()`
- `_copy_main_params_to_param_buffer()`
- the forced parameter sync caused by `disable_forward_pre_hook()`
- `optimizer.step()`
- the model-parallel update-success `logical_and` collective
- the model-parallel grad-norm max reduction
- the model-parallel zero-count max reduction
- `scheduler.step()`

The optimizer and scheduler steps were already guarded by `eval_mode`; the new
path removes the remaining setup and statistic work.

## Correctness-Preserving Work

- The existing `total_dataset_size` data-parallel all-reduce is unchanged.
- Forward execution, pipeline loss broadcast, context/data-parallel loss
  aggregation, and global loss calculation are unchanged.
- Normal overlap parameter-gather hooks stay enabled. MCore's forward pre-hook
  finishes pending parameter synchronization when each module consumes its
  parameters, avoiding the eager forced sync without using stale parameters.
- MXFP8 overlap with a shared parameter/gradient buffer is an explicit
  exception. It retains `disable_forward_pre_hook()`, the FP32-main-parameter
  copy, shared-buffer zeroing, and the existing one-training-step hook
  transition because MCore does not otherwise copy updated FP32 shards into
  that buffer. It still skips optimizer zeroing/stepping and statistic
  collectives during evaluation.
- FP8 extra state and the model's pre-evaluation train/eval mode are restored on
  the fast path. Tests compare parameters, extra state, grad buffers, optimizer
  state, scheduler state, hook state, and the next training call with a direct
  training control.
- The review follow-up adds a source-isolated behavioral harness that extracts
  the real worker `train`, hook-transition, buffer-copy, MXFP8 predicate,
  extra-state, and `param_sync_func` helper method bodies into one executable
  class. Stateful minimal DDP and optimizer objects replace only the external
  MCore dependencies; none of the worker helpers under review are replaced by
  lambdas.
- Both normal and MXFP8 shared-buffer tests compare `train -> train` with
  `train -> fast eval -> train`. They compare model weights, parameter and
  gradient buffers, optimizer and scheduler state, hook state, first-step
  disable state, saved `param_sync_func`, and restored `param_sync_func`.
  MXFP8 additionally proves copy-plus-zero precedes forced sync, the next
  training forward runs with hooks and `param_sync_func` disabled, and both are
  restored after the optimizer step. The tests did not expose a production
  divergence, so the follow-up requires no production-code change.

## Optional Timing

When validation comparison instrumentation supplies the existing `Timer`, the
Megatron worker reports:

- `worker_state_transition_s`
- `forward_s`
- `metric_reduction_s`
- `state_restore_s`

These use `time.perf_counter()` only. They do not call CUDA synchronization or
add a distributed collective. `Policy.train` takes the maximum observed value
from already-returned worker results on the driver. Validation also records
local `data_fetch_s` and `data_processing_s`. No timing fields or timer kwargs
are added by the default validation path.

## Verification

RED evidence before implementation:

- The source-level API check failed because the worker lacked
  `collect_eval_timing` and validation lacked
  `comparison_instrumentation_enabled`.
- `uv run pytest ...` could not start because the lockfile supports Linux
  `x86_64`/`aarch64`, not the local macOS platform.
- `python3 -m pytest ...` could not collect because local Python lacks `ray`.

Passing local checks after implementation:

- Source-isolated execution of the actual worker `train` method: fast-path
  skips/state restore, legacy fallback, next-training equivalence, timing
  fields, and MXFP8 required sync all passed.
- Source-isolated execution of validation instrumentation: opt-in worker/data
  timings and default-off behavior passed.
- Source-isolated execution of the actual `Policy.train` method: Megatron-only
  timing routing, DTensor isolation, driver max aggregation, and unchanged
  public signature passed.
- `python3 tests/unit/models/policy/test_megatron_worker_eval_state.py` executes
  the actual extracted worker helper bodies and passes both normal-buffer and
  MXFP8 shared-buffer state-equivalence tests.
- Standalone Pyright reports zero errors and zero warnings for the new
  dependency-free state-equivalence test module.
- `python3 -m py_compile` passed for all changed Python and test files.
- `ruff format --check` and `ruff check` passed for all changed Python and test
  files.
- `git diff --check` passed.
- Standalone Pyright cannot resolve the Linux project dependencies and exits
  with 128 import/pre-existing diagnostics. A JSON changed-line audit found no
  diagnostics on added lines.

## Remaining Concerns

- The real Ray/Megatron/PyTorch test modules were not runnable on this macOS
  host. Linux CI or a project container must run the focused repository tests.
- The local worker timings intentionally avoid CUDA synchronization, so they are
  host-observed section timings rather than exact GPU kernel completion times.
- Full zero/copy/hook skipping is intentionally unavailable for MXFP8 overlap
  with `reuse_grad_buf_for_mxfp8_param_ag`; removing that exception would risk
  evaluating stale parameters.
- This exact base does not contain the separate common-comparison-metrics task.
  Its integration can enable the new validation instrumentation argument; the
  default callers remain uninstrumented here.
