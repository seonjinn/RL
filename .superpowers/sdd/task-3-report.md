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
