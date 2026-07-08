# Task 3 Report: Runtime Precomputed Validation Mode

## Status

Implemented opt-in runtime consumption of verified precomputed SFT validation
events. The default dataloader path and the reviewed runtime CPU-cache path
remain enabled by their existing configuration and execution branches.

## Scope

- Added shared provenance construction in
  `nemo_rl/algorithms/sft_validation_provenance.py` and changed the producer to
  import the moved functions without changing its public call sites.
- Added typed precomputed-mode, manifest, and trusted external digest fields to
  `SFTConfig` and the SFT exemplar/reference configs.
- Added fail-closed precomputed cross-field validation.
- Added runtime artifact verification before Ray, tokenizer, data, model, or
  seed initialization.
- Added explicit event propagation through `sft_train()`, `validate()`, and the
  internal loss-availability path.
- Added focused unit behavior tests and source-isolated ordering/regression
  guards.

## Design

- Runtime preprocessing provenance is derived from the active resolved
  `MasterConfig`. Source provenance is derived from the active repository and
  recursive submodules after clean-tree checks. Only dataset, tokenizer, and
  container SHA-256 values come from trusted config fields.
- The expected `ValidationArtifactFingerprint` is built before calling
  `load_validation_event()`. No expected field is read from the manifest.
- Precomputed mode requires `event_batch`, a manifest, trusted lowercase
  SHA-256 inputs, fixed `4/64/1` validation shape, and runtime CPU cache off.
- Validation data loading is disabled with `setup_data(...,
  load_validation=False)`, so `setup()` receives no validation dataset and
  creates no validation dataloader.
- Every validation call clones `PrecomputedValidationEvent.data` immediately
  before the single event submission. The canonical loaded event is never
  submitted.
- Precomputed and live event validation both use `_event_validation_losses()`
  and token-count weighting.
- Training-mode restoration occurs at the original successful point inside the
  validation timer. A `finally` restores it after every exceptional exit once
  validation preparation has begun. CPU-cache publication remains after a
  successful restore.

## RED Evidence

### Config Contract

Added unit contract tests first. The normal unit command could not collect in
the local macOS environment because `tests/unit/conftest.py` imports Ray:

```text
pytest -q tests/unit/algorithms/test_sft.py -k 'precomputed_mode or sft_validation_execution_mode_defaults'
ModuleNotFoundError: No module named 'ray'
Pytest: No tests collected
```

Added an executable source-isolated contract test and observed the intended
failure: precomputed mode reached the live event payload-budget check instead
of accepting artifact-managed memory validation.

```text
FAILED test_precomputed_event_contract_fails_closed_without_runtime_dependencies
ValueError: event_batch validation requires an explicit positive payload byte budget
```

### Shared Provenance

Added the producer/shared-module structural guard before the refactor:

```text
FAILED test_validation_provenance_is_shared_with_the_producer
StopIteration: no sft_validation_provenance import
```

### Runtime Event Execution

Added behavior tests for loader avoidance, clone isolation, loss parity, and
error-path restoration before implementation. The source-isolated API/lifecycle
guard failed on the missing explicit event parameter:

```text
FAILED test_precomputed_event_is_explicit_cloned_and_restored_in_finally
AssertionError: precomputed_validation_event missing from keyword-only arguments
```

Self-review then added two regression guards before their fixes:

```text
FAILED: cache publication was not present after the restoration finally block
FAILED: successful restore_training_mode call was absent from the timed body
```

### Runtime Startup

Added startup-order and fingerprint-construction guards before runner changes:

```text
FAILED test_runtime_loads_verified_event_before_distributed_or_data_setup
AssertionError: missing _load_precomputed_validation_event
```

### Config Files

Added exemplar-config checks before adding the fields:

```text
FAILED test_sft_exemplar_configs_document_precomputed_inputs[sft.yaml]
AssertionError: validation_input_mode: dataloader was absent
```

## GREEN Evidence

Locally available final checks:

```text
/opt/homebrew/bin/pytest -q tests/source_isolated/test_sft_event_batch_source.py
16 passed in 0.20s

ruff check <changed Python files>
Ruff: No issues found

ruff format --check <changed Python files>
6 files already formatted

.venv/bin/python -m py_compile <changed Python files>
exit 0

pyright nemo_rl/algorithms/sft_validation_provenance.py
0 errors, 0 warnings, 0 informations

git diff --check
exit 0
```

The requested combined Pyright command runs locally but exits with existing
missing third-party imports and pre-existing `sft.py`/`run_sft.py` TypedDict
diagnostics in this sparse interpreter. After fixing the one introduced
optional-dataloader diagnostic, it reports 42 remaining errors outside the new
runtime interfaces. The new shared provenance module is clean independently.

## Self-Review

- Artifact validation occurs after full config resolution and before
  `init_ray()`, tokenizer construction, `setup_data()`, `setup()`, and
  `set_seed()`.
- Expected fingerprint construction uses active config/source plus trusted
  config digests and never reads the artifact manifest.
- Precomputed mode does not iterate a validation loader or instantiate the
  process-lifetime CPU cache.
- The canonical event is cloned for every call.
- Live `per_batch` still enumerates its loader and retains the existing policy
  call shape. Live event and CPU-cache capacity, clone, loss, and publication
  code remains in its existing branch.
- Training mode is restored after submission, reduction, timing, or cache
  failures, and cache publication waits for successful restoration.

## Remaining Concern

The Ray/Torch-dependent focused unit suites cannot run in this macOS worktree:
the sparse local environments lack Ray and the repository lock targets Linux.
Run these commands in the supported Linux test environment before merge:

```bash
pytest -q tests/unit/algorithms/test_sft.py tests/unit/algorithms/test_sft_validation_artifact.py
pytest -q tests/source_isolated/test_sft_event_batch_source.py
pyright nemo_rl/algorithms/sft.py examples/run_sft.py
```

## Review Fixes: Executable Startup Coverage And Pyrefly

### Behavioral Startup Coverage

Added three executable `examples.run_sft.main()` tests to the Linux-capable
`tests/unit/algorithms/test_sft_validation_artifact.py` suite. The tests execute
the real runner orchestration and patch only CLI/config loading and external
runtime side effects.

- Artifact fingerprint/load failure propagates before `init_ray()`, tokenizer
  construction, data setup, model setup, or training.
- Successful precomputed startup loads the artifact exactly once before Ray,
  tokenizer, data, and model setup; calls `setup_data(...,
  load_validation=False)`; passes `val_dataset=None` to `setup()`; and forwards
  the exact loaded event by keyword to `sft_train()`.
- Default dataloader mode does not call the artifact loader, calls
  `setup_data(tokenizer, config.data)` without suppressing validation, and
  passes the returned validation dataset into `setup()`.

No production change was required. The existing runner behavior satisfies the
new executable contract.

The tests were added before metadata changes. Local execution attempts record
the platform dependency gap explicitly:

```text
/opt/homebrew/bin/pytest -q tests/unit/algorithms/test_sft_validation_artifact.py -k 'run_sft_main' -vv
ImportError from tests/unit/conftest.py: ModuleNotFoundError: No module named 'ray'

/opt/homebrew/bin/pytest -q -o addopts='' --confcutdir=tests/unit/algorithms \
  tests/unit/algorithms/test_sft_validation_artifact.py -k 'run_sft_main' -vv
Collection error: ModuleNotFoundError: No module named 'torch'
```

The controller will execute these behavioral tests with the full focused Linux
unit suites.

### Pyrefly Coverage

Added `nemo_rl/algorithms/sft_validation_provenance.py` to
`pyrefly.toml` `project-includes`. Standalone execution is available through
`uvx` on this host:

```text
uvx pyrefly check nemo_rl/algorithms/sft_validation_provenance.py
INFO 0 errors
```

The full configured project invocation was also attempted:

```text
uvx pyrefly check
INFO 167 errors
```

Those project-wide diagnostics are missing local dependencies such as Torch,
OmegaConf, Safetensors, Pydantic, and YAML plus pre-existing diagnostics in
other allow-listed files. No diagnostic was reported for
`sft_validation_provenance.py` in the focused run.

Additional local checks for the executable test file passed:

```text
ruff check tests/unit/algorithms/test_sft_validation_artifact.py
Ruff: No issues found

pyright tests/unit/algorithms/test_sft_validation_artifact.py
0 errors, 0 warnings, 0 informations

.venv/bin/python -m py_compile tests/unit/algorithms/test_sft_validation_artifact.py
exit 0
```

## CW Job 13558843 Test-Helper Fix

### Linux RED Evidence

The controller ran the combined Task 3 suite on CW Linux. Pytest collected 177
tests and failed first at:

```text
tests/unit/algorithms/test_sft.py::test_precomputed_mode_requires_event_batch_and_manifest[overrides1]
TypeError: _validation_config() got multiple values for keyword argument
'validation_precomputed_manifest'
```

Full log:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/RL_worktrees/sft-validation-precomputed-20260707/logs/validation-artifact-tests/20260708-000720-task3/sna-val-runtime-task3_13558843.out
```

The failure occurred in test setup before production validation executed.

### Root Cause

`_precomputed_validation_config()` passed six defaults as explicit keywords and
then expanded caller `**overrides` in the same call. Python rejects any override
of those defaults as a duplicate call keyword. The failing manifest case was
the first collected example; dataset, tokenizer, container, input-mode, and
cache-mode overrides had the same helper-level risk.

The exact helper body was executed in isolation before the fix and reproduced:

```text
TypeError: _validation_config() got multiple values for keyword argument
'validation_precomputed_manifest'
```

### Fix And Coverage

The helper now builds one typed defaults mapping, updates it with caller
overrides, and expands the merged mapping once into `_validation_config()`.
Production code is unchanged.

An isolated execution of the fixed helper checked every defaulted field:

```text
6 override fields accepted
```

Locally available checks after the fix:

```text
Source-isolated pytest: 16 passed in 0.33s
Ruff: no issues
Ruff format: already formatted
Python py_compile: passed
git diff --check: passed
```

Direct Pyright remains unavailable as a clean project gate in the sparse local
environment. It reports unresolved Torch, Pydantic, and TorchData imports plus
pre-existing diagnostics elsewhere in `test_sft.py`; no diagnostic points to
the corrected helper. The controller will rerun the complete CW Linux suite.

## CW Job 13559485 CPU-Cache Test-Helper Fix

### Linux RED Evidence

The controller reran the combined Task 3 suite on CW Linux. The job passed 45
tests and then failed at:

```text
tests/unit/algorithms/test_sft.py::test_validation_cpu_cache_rejects_non_cpu_tensor_before_train
ValueError: sft.validation_event_cache_mode=cpu requires a process-lifetime
validation event cache
```

The test expected the later `CPU cache.*meta` payload rejection, but
`_run_validation()` passed `validation_event_cache=None`.

Full log:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/RL_worktrees/sft-validation-precomputed-20260707/logs/validation-artifact-tests/20260708-001543-task3-r2/sna-val-runtime-task3-r2_13559485.out
```

### Root Cause And Baseline Comparison

This is an older test-helper omission, not a Task 3 production regression. At
the approved Task 2 baseline `fb9ca3d52`, production validation already rejected
CPU-cache mode when `validation_event_cache` was absent, and the same test's
`_run_validation()` helper already omitted that argument.

An AST caller audit found exactly one `_run_validation()` caller configured for
CPU cache: the failing non-CPU-tensor test. Other CPU-cache tests call
`validate()` directly and already pass an explicit shared `_ValidationEventCache`.

Focused pre-fix helper execution reproduced:

```text
TypeError: _run_validation() got an unexpected keyword argument
'validation_event_cache'
```

### Fix And Regression Evidence

`_run_validation()` now accepts a keyword-only optional cache and forwards it
unchanged to `validate()`. The affected CPU-mode test explicitly supplies a new
`_ValidationEventCache()` so it reaches the intended meta-tensor rejection.
The helper does not create caches implicitly, preserving caller ownership of
the process-lifetime cache. Production and precomputed paths are unchanged.

Focused post-fix helper execution verified object identity through the call:

```text
explicit cache forwarded
```

Locally available checks after the fix:

```text
Source-isolated pytest: 16 passed in 0.20s
Ruff: no issues
Ruff format: already formatted
Python py_compile: passed
git diff --check: passed
```

The controller will rerun the complete CW Linux suite.
