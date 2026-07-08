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
