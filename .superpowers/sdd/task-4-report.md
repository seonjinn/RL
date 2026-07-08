# Task 4: Code-Level Correctness Audit and Review Gate

## Status

`DONE_WITH_CONCERNS`

- Approved Tasks 1-3 head: `e8e13f5a9e0694adb1a574fd4b7e35507ab3ca9b`
- Task 4 implementation commit: `2c8bfdc5d63d8db05da34258148aea640222e311`
- Independent code-review gate: pass, with no findings
- Supported-Linux execution gate: pending because the local macOS environment
  does not provide Ray, Torch, or Megatron

The correctness audit is opt-in and disabled by default. The disabled path does
not construct the auditor and adds no worker RPC, tensor reduction,
synchronization, RNG read, or audit timing call.

## Changed Files

Implementation commit `2c8bfdc5d63d8db05da34258148aea640222e311`
changes exactly these files:

- `examples/configs/sft.yaml`
- `examples/configs/sft_superv3_prepacked.yaml`
- `nemo_rl/algorithms/sft.py`
- `nemo_rl/algorithms/sft_correctness_audit.py`
- `nemo_rl/models/policy/lm_policy.py`
- `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- `pyrefly.toml`
- `tests/unit/algorithms/test_sft.py`
- `tests/unit/algorithms/test_sft_correctness_audit.py`
- `tests/unit/algorithms/test_sft_validation_artifact.py`
- `tests/unit/reference_configs/sft.yaml`

The documentation follow-up changes:

- `.superpowers/sdd/task-4-report.md`
- `docs/superpowers/reviews/2026-07-07-precomputed-validation-correctness.md`

## Implementation

- Added `CorrectnessAuditConfig(enabled=False)` and explicit disabled defaults
  to the SFT reference and example configurations.
- Added driver snapshots for Python, NumPy, Torch CPU and initialized driver
  CUDA RNG, explicit generator state, train-loader state, validation payload,
  ordered sample identity, and exact token counts.
- Records validation-boundary snapshots and digests the next train batch only
  after the existing training loop naturally yields it. The audit never
  advances or prefetches the train loader.
- Added the concrete `Policy.get_correctness_state_fingerprint()` method only.
  `PolicyInterface` and unrelated backends remain unchanged.
- Added per-worker fingerprints using direct named parameter, buffer, module
  mode, optimizer-state, and exact optimizer-step inspection. Returned values
  are small Python records; tensor moments and fixed samples are evidence, not
  cryptographic equality proof.
- Worker RNG inspection calls only MCore `get_all_rng_states()` plus Torch's
  read-only CUDA RNG state API. It does not initialize trackers or advance RNG.
- Audit code does not call state-dict/load/checkpoint paths, change model mode,
  synchronize parameters, seed or set RNG, run forwards, or step optimizers.
- Audit timing has separate metrics and is excluded from reported step, loop,
  and end-to-end performance windows.
- Default per-batch validation, CPU-cache validation, and precomputed-event
  paths retain their existing behavior when the audit is disabled.

## TDD Evidence

The initial source-level tests failed before implementation because the audit
module, concrete Policy method, worker method, and config block did not exist.
Representative RED results were:

```text
AssertionError: missing required audit module: nemo_rl/algorithms/sft_correctness_audit.py
AssertionError: nemo_rl/models/policy/lm_policy.py: missing Policy.get_correctness_state_fingerprint
AssertionError: examples/configs/sft.yaml
```

The repository unit suite could not collect in the local environment even at
RED because `tests/unit/conftest.py` imports Ray.

## Exact Verification

Focused Task 4 unit command:

```bash
/opt/homebrew/bin/pytest -q tests/unit/algorithms/test_sft_validation_artifact.py tests/unit/algorithms/test_sft_correctness_audit.py tests/unit/algorithms/test_sft.py
```

Result: blocked during collection, exit 4.

```text
ImportError while loading conftest 'tests/unit/conftest.py'
ModuleNotFoundError: No module named 'ray'
```

Source-isolated regression command:

```bash
/opt/homebrew/bin/pytest -q tests/source_isolated/test_sft_event_batch_source.py
```

Result: `16 passed in 0.19s`.

Ruff format command:

```bash
ruff format --check nemo_rl/algorithms/sft_validation_artifact.py nemo_rl/algorithms/sft_correctness_audit.py nemo_rl/algorithms/sft.py nemo_rl/models/policy/lm_policy.py nemo_rl/models/policy/workers/megatron_policy_worker.py examples/prepare_sft_validation_event.py examples/run_sft.py tests/unit/algorithms/test_sft_validation_artifact.py tests/unit/algorithms/test_sft_correctness_audit.py tests/unit/algorithms/test_sft.py
```

Result: `10 files already formatted`, exit 0.

Ruff lint command:

```bash
ruff check nemo_rl/algorithms/sft_validation_artifact.py nemo_rl/algorithms/sft_correctness_audit.py nemo_rl/algorithms/sft.py nemo_rl/models/policy/lm_policy.py nemo_rl/models/policy/workers/megatron_policy_worker.py examples/prepare_sft_validation_event.py examples/run_sft.py tests/unit/algorithms/test_sft_validation_artifact.py tests/unit/algorithms/test_sft_correctness_audit.py tests/unit/algorithms/test_sft.py
```

Result: `Ruff: No issues found`, exit 0.

Compilation command:

```bash
/opt/homebrew/bin/python3 -m py_compile nemo_rl/algorithms/sft_validation_artifact.py nemo_rl/algorithms/sft_correctness_audit.py nemo_rl/algorithms/sft.py nemo_rl/models/policy/lm_policy.py nemo_rl/models/policy/workers/megatron_policy_worker.py examples/prepare_sft_validation_event.py examples/run_sft.py tests/unit/algorithms/test_sft_validation_artifact.py tests/unit/algorithms/test_sft_correctness_audit.py tests/unit/algorithms/test_sft.py
```

Result: exit 0 with no output.

Requested Pyright command:

```bash
pyright nemo_rl/algorithms/sft_validation_artifact.py nemo_rl/algorithms/sft_correctness_audit.py nemo_rl/algorithms/sft.py nemo_rl/models/policy/lm_policy.py nemo_rl/models/policy/workers/megatron_policy_worker.py examples/prepare_sft_validation_event.py examples/run_sft.py
```

Result: exit 1 with `162 errors, 1 warning, 0 informations`. The diagnostics
are dominated by missing local Torch, Ray, Megatron, Pydantic, TorchData,
Transformers, OmegaConf, and Safetensors dependencies plus existing repository
diagnostics. A focused check of the new audit module and tests left only the two
unresolved local Torch imports; the worker helper's relevant lines left only
unresolved Torch and `megatron.core.tensor_parallel.random` imports.

Diff validation command:

```bash
git diff --check
```

Result: exit 0 with no output.

Tasks 1-3 previously passed their combined supported-Linux gate at the approved
head: CW job `13559835`, `177 passed, 3 warnings in 43.82s`. That result does
not replace a Linux execution of the new Task 4 tests.

Not run for Task 4:

- The focused unit command on a supported Linux environment with Ray, Torch,
  and Megatron installed.
- A Task 4 GPU/Ray/Megatron integration run.
- The full repository test suite.

## Self-Review

- Re-read the brief and concrete worker/RNG API research against the final
  implementation.
- Inspected the disabled branch to confirm no auditor construction, worker RPC,
  RNG read, tensor reduction, synchronization, or audit timing occurs.
- Checked that the next train-batch digest is attached only to a naturally
  yielded batch and that no audit-only loader iteration exists.
- Checked that worker collection covers every worker rank, sorts records, and
  fails on duplicate ranks or uninitialized MCore tracker state.
- Checked that all digest inputs are canonical and mutation-sensitive while
  worker tensor summaries remain explicitly characterized as fingerprints.
- Checked timing placement and subtraction so audit work cannot contaminate
  reported performance windows.
- Checked scope: no `PolicyInterface` expansion, no unrelated backend change,
  and no changes to default per-batch, CPU-cache, or precomputed behavior.
- Checked the implementation diff for forbidden APIs and unrelated changes.
- Independent read-only Codex review session
  `019f40c6-3d07-7b20-84c5-80e0a1e30c06` reviewed commit
  `2c8bfdc5d63d8db05da34258148aea640222e311` and returned `Findings: None`
  and `Verdict: LGTM`.

Detailed invariant evidence is recorded in
`docs/superpowers/reviews/2026-07-07-precomputed-validation-correctness.md`.

## Remaining Concerns

- The focused Ray-dependent Task 4 unit suite has not run on supported Linux.
  It is the remaining execution gate before merge.
- Local Pyright cannot produce a clean project result without the Linux project
  dependencies. The focused audit diagnostics were reduced to unresolved
  dependency imports.
- Worker moments and deterministic samples can miss a localized tensor change;
  they are bounded diagnostic evidence, not equality proofs.
- Audit mode intentionally performs worker RPCs, local GPU reductions, and
  host-visible state reads. It must remain disabled for timed performance runs.
- The persisted precomputed artifact omits producer-only `idx` metadata, so
  ordered `input_ids` are the persisted sample-identity evidence.

## Review Fixes

This section records the follow-up for the six Important findings and
supersedes the earlier pre-fix review conclusion for commit
`2c8bfdc5d63d8db05da34258148aea640222e311`.

### Changes

1. Worker optimizer fingerprints now include every tensor referenced by every
   optimizer parameter group, including distinct MCore FP32/main local shards.
   Each record carries the stable `(optimizer_index, group_index, param_index)`
   owner tuple and the same shape, dtype, moments, finite counts, and fixed
   samples used for model and optimizer-state tensors.
2. A successful validation record remains pending until the normal training
   iterator yields the next batch. The resulting single finalized
   `CorrectnessAuditRecord` contains that batch evidence. The explicit
   `compare_next_train_batch_to_control()` gate compares it with a separately
   and naturally consumed no-validation control; missing natural-batch evidence
   fails finalization instead of producing a passing log-only record.
3. Audit-enabled validation hashes the actual canonical runtime payload,
   ordered `idx` plus `input_ids` identity, and exact recomputed valid-token
   counts immediately before and after every real policy submission. This
   covers per-batch, CPU-cache event, uncached event, and precomputed paths.
   Missing mappings, `input_ids`, masks, counts, or evidence fail explicitly.
4. Worker RNG collection now rejects both empty and non-mapping
   `get_all_rng_states()` results. It still imports and calls that API directly
   and never calls `get_cuda_rng_tracker()`.
5. Production-capture tests now mutate real Python, NumPy, Torch CPU, explicit
   generator, loader, model parameter, optimizer main-shard parameter,
   optimizer step, and module training-mode state and assert the production
   gate identifies each family.
6. The restart test saves a real `StatefulDataLoader` and generator state,
   recreates and restores them at the same boundary, executes the production
   SFT validation function with exact audit evidence, then compares the next
   naturally consumed batch with an independent no-validation control loader.

Audit evidence hashing and exact-token reductions execute only when
`collect_correctness_audit_evidence` is true. Their measured duration is
removed from validation and step performance windows and added to the separate
correctness-audit timing. The default audit-disabled worker and validation
paths do not call these helpers.

### Changed Files

- `nemo_rl/algorithms/sft.py`
- `nemo_rl/algorithms/sft_correctness_audit.py`
- `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- `tests/source_isolated/test_sft_event_batch_source.py`
- `tests/unit/algorithms/test_sft.py`
- `tests/unit/algorithms/test_sft_correctness_audit.py`
- `.superpowers/sdd/task-4-report.md`

`tests/unit/algorithms/test_sft_validation_artifact.py` was rerun as requested
but did not require a source change.

### Exact Verification

Requested focused unit command:

```bash
/opt/homebrew/bin/pytest -q tests/unit/algorithms/test_sft_correctness_audit.py tests/unit/algorithms/test_sft.py tests/unit/algorithms/test_sft_validation_artifact.py
```

Result: exit 4 during collection; no unit tests ran.

```text
ImportError while loading conftest 'tests/unit/conftest.py'
ModuleNotFoundError: No module named 'ray'
```

Requested source-isolated command:

```bash
/opt/homebrew/bin/pytest -q tests/source_isolated/test_sft_event_batch_source.py
```

Result: `18 passed in 0.20s`, exit 0.

Ruff lint command:

```bash
ruff check nemo_rl/algorithms/sft.py nemo_rl/algorithms/sft_correctness_audit.py nemo_rl/models/policy/workers/megatron_policy_worker.py tests/source_isolated/test_sft_event_batch_source.py tests/unit/algorithms/test_sft.py tests/unit/algorithms/test_sft_correctness_audit.py
```

Result: `Ruff: No issues found`, exit 0.

Ruff format command:

```bash
ruff format --check nemo_rl/algorithms/sft.py nemo_rl/algorithms/sft_correctness_audit.py nemo_rl/models/policy/workers/megatron_policy_worker.py tests/source_isolated/test_sft_event_batch_source.py tests/unit/algorithms/test_sft.py tests/unit/algorithms/test_sft_correctness_audit.py
```

Result: `6 files already formatted`, exit 0.

Compilation command:

```bash
/opt/homebrew/bin/python3 -m py_compile nemo_rl/algorithms/sft.py nemo_rl/algorithms/sft_correctness_audit.py nemo_rl/models/policy/workers/megatron_policy_worker.py tests/source_isolated/test_sft_event_batch_source.py tests/unit/algorithms/test_sft.py tests/unit/algorithms/test_sft_correctness_audit.py tests/unit/algorithms/test_sft_validation_artifact.py
```

Result: exit 0 with no output.

Focused Pyright command:

```bash
pyright nemo_rl/algorithms/sft_correctness_audit.py tests/unit/algorithms/test_sft_correctness_audit.py
```

Result: exit 1 with three missing-import diagnostics only: Torch in the module
and test, plus TorchData in the test. There were no additional type
diagnostics.

Diff validation command:

```bash
git diff --check
```

Result: exit 0 with no output.

Not run locally:

- The focused unit tests on supported Linux with Ray, Torch, TorchData, and
  Megatron installed.
- A GPU/Ray/Megatron integration run.
- The full repository test suite.
- The controller's CW Linux gate and post-fix re-review.

### Self-Review

- Confirmed optimizer parameter records are independent of optimizer-state
  presence and cover chained optimizer children with stable owner tuples.
- Confirmed invalid tracker returns fail before `.items()` and no code path
  imports or calls `get_cuda_rng_tracker()`.
- Confirmed successful boundary records are not emitted until natural next
  train-batch evidence is attached, and pending records fail closed at flush.
- Confirmed the no-validation control uses a separate restored-equivalent
  loader and does not pre-consume or rewind the audited loader.
- Confirmed live validation evidence comes from the actual submitted or
  canonical payload, not config metadata, batch dimensions, or dataset names.
- Confirmed per-batch and event token counts are computed from the exact
  `sample_mask * token_mask` payload and precomputed counts come from the
  verified artifact contract.
- Confirmed before/after runtime evidence is included in the same audit record
  and payload mutation causes the gate to reject.
- Confirmed all production mutation tests call the production capture and gate
  functions rather than only replacing synthetic snapshot fields.
- Confirmed loader/generator restart coverage uses `state_dict()`, recreates
  the loader and generator, restores with `load_state_dict()`/`set_state()`,
  runs production validation, and compares the natural next batch to control.
- Confirmed audit timing is excluded from validation, step, end-to-end, and
  loop-interval performance windows, while disabled mode makes no audit RPC,
  RNG read, tensor reduction, synchronization, or evidence-hash call.
- Confirmed the concrete `Policy` method remains unchanged, `PolicyInterface`
  is not expanded, and no unrelated backend or default validation path changed.

## Second Review Fixes

This section records the follow-up for the second Task 4 review's two
Important findings and stale-documentation Minor. It supplements the prior
review-fix section above.

### Changes

1. Validation evidence now belongs to a
   `_ValidationCorrectnessEvidenceCollector` created outside the validation
   return path. Each audited policy submission captures after-evidence in a
   `finally`, and the model-training restoration boundary finalizes the
   collector in a second `finally`. `SFTCorrectnessAuditor.audit_validation()`
   consumes the collector on both success and exception, so failed submission,
   invalid loss, and restoration failure are checked against actual runtime
   digests rather than placeholder `None` values.
2. Production-capture mutation coverage now includes controlled driver Torch
   CUDA RNG API side effects, controlled worker Torch CUDA and MCore RNG API
   side effects, a real optimizer `exp_avg` tensor, independent runtime `idx`
   mutation, and independent mask-derived exact-token-count mutation. These
   tests invoke the production capture and comparison gate.
3. `docs/superpowers/reviews/2026-07-07-precomputed-validation-correctness.md`
   now covers the full current Task 4 commit range, both review rounds and
   their fixes, current invariant evidence, the actual 19-test source-isolated
   count, and the pending controller decision. It does not claim that the CW
   Linux Task 4 gate passed.
4. Self-review found and removed audit scaffolding from disabled execution:
   policy submission and training-mode restoration keep direct calls when no
   collector is present. The new `try`/`finally`, evidence hashing, RNG reads,
   reductions, and audit timing are confined to enabled mode.

### Changed Files

- `nemo_rl/algorithms/sft.py`
- `nemo_rl/algorithms/sft_correctness_audit.py`
- `tests/source_isolated/test_sft_event_batch_source.py`
- `tests/unit/algorithms/test_sft.py`
- `tests/unit/algorithms/test_sft_correctness_audit.py`
- `docs/superpowers/reviews/2026-07-07-precomputed-validation-correctness.md`
- `.superpowers/sdd/task-4-report.md`

### TDD Evidence

The source-isolated contract test was added before the collector implementation:

```bash
/opt/homebrew/bin/pytest -q tests/source_isolated/test_sft_event_batch_source.py -k correctness_evidence_survives_validation_exceptions
```

Initial result: exit 1, `1 failed`; the failure was
`AssertionError: assert '_ValidationCorrectnessEvidenceCollector' in class_names`.
After implementing the external collector and finally boundaries, the focused
test passed. The complete source-isolated file result is recorded below.

### Exact Verification

Requested focused unit command:

```bash
/opt/homebrew/bin/pytest -q tests/unit/algorithms/test_sft_correctness_audit.py tests/unit/algorithms/test_sft.py tests/unit/algorithms/test_sft_validation_artifact.py
```

Result: exit 4 during collection; no unit tests ran.

```text
ImportError while loading conftest 'tests/unit/conftest.py'
tests/unit/conftest.py:24: in <module>
    import ray
E   ModuleNotFoundError: No module named 'ray'
```

Requested source-isolated command:

```bash
/opt/homebrew/bin/pytest -q tests/source_isolated/test_sft_event_batch_source.py
```

Result: `19 passed in 0.24s`, exit 0.

Ruff lint command:

```bash
ruff check nemo_rl/algorithms/sft.py nemo_rl/algorithms/sft_correctness_audit.py tests/source_isolated/test_sft_event_batch_source.py tests/unit/algorithms/test_sft.py tests/unit/algorithms/test_sft_correctness_audit.py
```

Result: `Ruff: No issues found`, exit 0.

Ruff format command:

```bash
ruff format --check nemo_rl/algorithms/sft.py nemo_rl/algorithms/sft_correctness_audit.py tests/source_isolated/test_sft_event_batch_source.py tests/unit/algorithms/test_sft.py tests/unit/algorithms/test_sft_correctness_audit.py
```

Result: `5 files already formatted`, exit 0.

Compilation command:

```bash
/opt/homebrew/bin/python3 -m py_compile nemo_rl/algorithms/sft.py nemo_rl/algorithms/sft_correctness_audit.py tests/source_isolated/test_sft_event_batch_source.py tests/unit/algorithms/test_sft.py tests/unit/algorithms/test_sft_correctness_audit.py tests/unit/algorithms/test_sft_validation_artifact.py
```

Result: exit 0 with no output.

Diff validation command:

```bash
git diff --check
```

Result: exit 0 with no output after the report update.

Not run locally:

- The three requested unit files, because collection requires the unavailable
  `ray` dependency.
- A supported Linux run with Ray, Torch, TorchData, Megatron, and CUDA.
- GPU/Ray/Megatron integration tests and the full repository suite.
- The controller's CW Linux Task 4 gate and second post-fix review.

### Self-Review

- Confirmed the evidence provider is independent of `_SFTValidationResult` and
  is consumed in the auditor's exception-safe `finally` path.
- Confirmed every audited submission captures after-evidence even when policy
  submission or loss validation raises, and restoration finalization runs even
  when `model.train()` restoration raises.
- Confirmed incomplete evidence raises `CorrectnessAuditError`; the gate cannot
  compare placeholder `None` validation digests.
- Confirmed exception-path tests mutate the actual submitted payload and assert
  the production gate reports the corresponding digest mismatch.
- Confirmed CUDA and MCore RNG tests use controlled side effects on the APIs
  called by production capture without initializing trackers or advancing RNG.
- Confirmed identity and exact-token tests mutate `idx` and mask data
  independently, then invoke production runtime capture and comparison.
- Confirmed disabled mode retains direct policy and restoration calls and adds
  no audit RPC, RNG read, tensor reduction, synchronization, evidence hash, or
  audit-timing work.
- Confirmed no `PolicyInterface`, backend, checkpoint/state-load, model-mode,
  parameter-sync, or unrelated source changes were introduced.
