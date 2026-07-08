# Precomputed Validation Correctness Review

## Scope

- Full feature range: `c1a415dae1e0bc909eb1891b0d78be92da35e50f^..HEAD`
- Approved Tasks 1-3 head: `e8e13f5a9e0694adb1a574fd4b7e35507ab3ca9b`
- Initial Task 4 implementation: `2c8bfdc5d63d8db05da34258148aea640222e311`
- Initial review documentation: `f9361d3dd35b6e44d38c9786282cef60a73bf751`
- First-review fix: `b104c8e894507644400eec38e6b8fea75196d5c6`
- Second-review fix: `8f1ca28d09ff9823f1932c0b3d21a796edb8edbc`
- CW test-contract follow-up: the signed commit containing this document
- Requirements: `.superpowers/sdd/task-4-brief.md`
- Worker/RNG research: `.superpowers/sdd/task-4-api-research.md`

The full range covers validation artifact production, publication, loading,
runtime consumption, CPU caching, exact loss weighting, opt-in correctness
auditing, worker-local fingerprints, natural next-batch evidence, exception
lifecycle handling, timing isolation, and the merge review gate.

## Current Decision

**Code review gate: READY FOR CONTROLLER RE-REVIEW.** The two Important findings
from the second review, its stale-document Minor, and the subsequent CW
test-contract mismatch are addressed in the current range. This document does
not claim that the controller has approved the follow-up.

**Supported-Linux execution gate: PENDING FULL RERUN.** CW job `13566467`
reached `105 passed, 1 failed` before `maxfail=1` stopped the suite. The only
failure was a stale test expectation: the CUDA RNG case expected
`torch_cuda_rng_digests`, while the comparator correctly returned the stronger
device-specific path `torch_cuda_rng_digests.0`. The current follow-up updates
only that parameterized contract. The controller must rerun the full focused
suite on CW Linux.

The historical CW job `13559835` reported `177 passed, 3 warnings in 43.82s`
at the approved Tasks 1-3 head `e8e13f5a9`; it does not cover either Task 4
implementation or either Task 4 review-fix commit.

Worker scalar moments and deterministic samples remain bounded fingerprints.
They are mutation evidence, not cryptographic tensor-equality proofs.

## Review History

### Initial Review

The initial review of `2c8bfdc5d` did not identify the later coverage and
lifecycle gaps. Its prior PASS statement is superseded by the two subsequent
review rounds.

### First Review: Six Important Findings

Commit `b104c8e894507644400eec38e6b8fea75196d5c6` addressed:

1. Missing optimizer param-group/main-shard fingerprints.
2. Detached, log-only next-train-batch evidence.
3. Config metadata substituted for live/CPU-cache validation payload evidence.
4. Empty or non-mapping MCore RNG tracker returns accepted as valid.
5. Synthetic-only state mutation tests.
6. A restart test that did not restore a real loader and generator around real
   validation.

The fix fingerprints every optimizer param-group tensor with a stable owner
tuple, finalizes one comparable audit record after a naturally consumed batch,
captures exact runtime payload/identity/token evidence, rejects invalid tracker
maps, adds production mutation tests, and restores a real StatefulDataLoader
and generator before running production SFT validation against a control.

### Second Review: Two Important Findings and One Minor

The current signed follow-up addresses:

1. Validation evidence was owned by the validation return value and disappeared
   when submission, loss validation, or training-mode restoration raised.
2. Production mutation coverage omitted driver CUDA RNG, worker Torch/MCore RNG,
   optimizer state tensors, independent sample identity, and independent exact
   token-count changes.
3. This review document described only the initial implementation and reported
   a stale source-isolated count and decision.

The runtime now creates one external evidence collector per audited validation.
The opt-in submission branches capture before evidence, call the policy inside
`try`, and capture after evidence in `finally`. The validation wrapper finalizes
the collector in a `finally` after the training-mode restoration attempt. The
auditor reads the finalized collector on both success and exception paths, so
it never substitutes `None` payload, sample, or token digests. Collector
capture failures also fail closed with an explicit audit error.

The audit-disabled branches retain their direct policy submission and direct
restoration call shapes. They do not construct a collector or execute its
timers, hashes, reductions, or exception wrappers.

### CW Linux Contract Follow-Up

CW job `13566467` exposed one test-only mismatch after 105 tests passed. The
parameterized driver-state test used each dataclass field name as the expected
difference, but CUDA RNG digests are a sequence and the production comparator
recurses into sequences. A mutation of device 0 therefore intentionally
reports `torch_cuda_rng_digests.0`. The test now carries an explicit expected
path per parameter: CUDA expects the indexed path, while all scalar families
retain their existing top-level expectation. Production comparison behavior is
unchanged.

## Invariant Matrix

| Invariant | Current implementation evidence | Current test/review evidence | Status |
|---|---|---|---|
| Dataset and preprocessing identity are pinned | Manifest provenance derives from active dataset, tokenizer, preprocessing, and container fingerprints | Tasks 1-3 provenance/startup tests; historical Linux gate through `e8e13f5a9` | Pass through Tasks 1-3 |
| Artifact publication is atomic and fail closed | Strict schema and hashes, memory preflight, locked atomic publication | Corrupt, partial, interrupted, and concurrent publication tests | Pass through Tasks 1-3 |
| Canonical precomputed and CPU-cache payloads remain owned and immutable | Loads own CPU tensors; submissions clone canonical CPU/precomputed payloads | Clone, mutation, failed-submission, cache lifecycle tests | Covered; current Linux rerun pending |
| Live and precomputed loss weighting use exact valid tokens | Validation weights losses with exact `sample_mask * token_mask` counts and artifact counts | Parity and invalid-loss-shape tests | Covered; current Linux rerun pending |
| Driver Python, NumPy, Torch CPU, and Torch CUDA RNG are exact | Production snapshots hash complete states; initialized CUDA devices use `get_rng_state_all()` | Real mutation matrix plus controlled driver CUDA API side effects | Covered; current Linux rerun pending |
| Explicit generator and train-loader position are exact | `Generator.get_state()` and loader `state_dict()` are hashed without advancing iteration | Real mutation tests and restored StatefulDataLoader restart/control test | Covered; current Linux rerun pending |
| Runtime validation payload survives exceptions | External collector exists independently of result and finalizes after restoration attempt | Mutating submission-failure, invalid-loss, and restore-failure production tests | Covered; current Linux rerun pending |
| Ordered runtime sample identity is exact | Evidence independently hashes ordered `idx` and `input_ids` | Production path mutates only `idx` and asserts sample-identity rejection | Covered; current Linux rerun pending |
| Runtime exact token counts are independent evidence | Counts are recomputed from current `sample_mask * token_mask` before and after submission | Production path mutates only a mask and asserts token-count rejection | Covered; current Linux rerun pending |
| Next train batch is natural and comparable | Successful records remain pending until the existing training iterator yields a batch; explicit control comparison API gates it | Natural-batch, no-validation control, and restart/resume tests | Covered; current Linux rerun pending |
| Every worker rank is represented | Concrete Policy RPC calls every worker, validates ranks, and sorts records | Multi-rank routing, duplicate-rank, and order tests | Covered; current Linux rerun pending |
| Worker Torch CUDA RNG is read-only and mutation-sensitive | Worker calls `torch.cuda.get_rng_state(current_device)` | Controlled API side effects pass through production worker capture and gate | Covered; current Linux rerun pending |
| MCore RNG is read without initialization | Worker calls only direct `get_all_rng_states()` and rejects assertion, empty, and non-mapping results | Controlled tracker mutation, uninitialized, empty, non-mapping, and source-guard tests | Covered; current Linux rerun pending |
| Model parameters, buffers, and training mode are stable | Direct named traversal records local fingerprints and every module mode | Production model and independent mode mutation tests | Covered; current Linux rerun pending |
| Optimizer main shards, state tensors, and steps are stable | Every param-group tensor and direct state tensor has owner, shape/dtype, moments, samples; exact steps are separate | Main-shard, `exp_avg`, and step mutation tests | Covered; current Linux rerun pending |
| Forbidden worker paths remain absent | Fingerprint uses no state/load/checkpoint, mode change, parameter sync, seed/setter, forward, or optimizer step | Source-isolated and unit AST guards | Covered locally by source guard |
| Failed validation cannot publish cache state | Cache publication occurs only after validation and restoration return successfully | Existing failed-submission, invalid-loss, restore-failure, and atomic-cache tests | Covered; current Linux rerun pending |
| Audit overhead is isolated | Collector and worker audit timing are separate and removed from validation/step/loop windows | Fixed-timer, separate logging, and source path checks | Covered; current Linux rerun pending |
| Disabled mode remains unchanged | No auditor/collector construction and direct submission/restoration branches | Disabled-path RPC/read assertions and source inspection | Covered; current Linux rerun pending |
| Interfaces and unrelated backends remain scoped | Only concrete Policy method exists; PolicyInterface and unrelated backends are unchanged | Source inspection | Pass |

## Verification Evidence

Passed locally on the current second-review follow-up:

- `/opt/homebrew/bin/pytest -q tests/source_isolated/test_sft_event_batch_source.py`
  returned `19 passed`.
- Ruff lint passed on all changed Python files.
- Ruff format check passed on all changed Python files.
- Python compilation passed for all changed Python files and requested focused
  test modules.
- `git diff --check` passed.
- Focused Pyright checks reported only missing local dependency imports plus
  existing repository diagnostics; the new audit module/test check had only
  unresolved Torch imports.

Blocked locally:

- `/opt/homebrew/bin/pytest -q tests/unit/algorithms/test_sft_correctness_audit.py tests/unit/algorithms/test_sft.py tests/unit/algorithms/test_sft_validation_artifact.py`
  stops in `tests/unit/conftest.py` with
  `ModuleNotFoundError: No module named 'ray'`; no focused unit test runs.
- No current Task 4 GPU/Ray/Megatron integration or full repository suite ran
  locally.
- The controller's CW Linux gate and post-fix review are pending.

CW Linux partial result:

- Job `13566467`: `105 passed, 1 failed`, then stopped at `maxfail=1`.
- Failure:
  `tests/unit/algorithms/test_sft_correctness_audit.py::test_correctness_gate_rejects_each_driver_state_family[cuda-rng]`.
- Expected: `("torch_cuda_rng_digests",)`.
- Actual: `("torch_cuda_rng_digests.0",)`.
- A complete rerun after the test-contract correction remains pending; this
  document does not claim that the supported-Linux gate passed.

## Residual Limits

- Fixed samples and moments can miss a localized tensor mutation. Increase the
  deterministic sample count when stronger fingerprint evidence is required.
- Audit mode intentionally performs worker RPCs, local tensor reductions, and
  host-visible state reads. It must remain disabled in timed performance runs.
- Precomputed artifacts omit producer-only `idx`; ordered persisted
  `input_ids` remain their sample-identity evidence. Live and CPU-cache runtime
  paths independently include `idx` when present.
- Merge readiness still requires controller approval and a current supported-
  Linux execution of the focused Task 4 unit suite.
