# Precomputed Validation Correctness Review

## Scope

- Full feature range: `c1a415dae1e0bc909eb1891b0d78be92da35e50f^..HEAD`
- Approved Tasks 1-3 head: `e8e13f5a9e0694adb1a574fd4b7e35507ab3ca9b`
- Initial Task 4 implementation: `2c8bfdc5d63d8db05da34258148aea640222e311`
- Initial review documentation: `f9361d3dd35b6e44d38c9786282cef60a73bf751`
- First-review fix: `b104c8e894507644400eec38e6b8fea75196d5c6`
- Second-review fix: `8f1ca28d09ff9823f1932c0b3d21a796edb8edbc`
- CW test-contract follow-up: `175e39c9250af597e8c860d846f9be44dd39799c`
- CW read-only capture follow-up:
  `45ca880cbaf32ec742a66934b36d3fb41420ee0f`
- Loader identity fail-closed follow-up:
  `2a4c1ca8f025dc1099e72b148dd2ac8a9b4779a2`
- TorchData source-provenance follow-up:
  `61d7ff5d9ee57e077ed5ccde2536bac037b9b32b`
- Static typing follow-up: the signed commit containing the final review section
- Requirements: `.superpowers/sdd/task-4-brief.md`
- Worker/RNG research: `.superpowers/sdd/task-4-api-research.md`

The full range covers validation artifact production, publication, loading,
runtime consumption, CPU caching, exact loss weighting, opt-in correctness
auditing, worker-local fingerprints, natural next-batch evidence, exception
lifecycle handling, timing isolation, and the merge review gate.

## Current Decision

**Code review gate: READY FOR CONTROLLER RE-REVIEW.** The prior review findings,
the CUDA test-contract mismatch, the restored-loader read-only failure, and the
subsequent loader-adapter identity and source-provenance findings are addressed
in the current range. This document does not claim that the controller has
approved the follow-up.

**Supported-Linux functional gate: PASS AT `61d7ff5d9`.** CW job `13568701`
completed with `269 passed, 47 warnings in 47.25s`, using isolated
`RAY_TMPDIR`/`TMPDIR` and an unset `RAY_ADDRESS`.

**Supported-Linux static gate: FOLLOW-UP PENDING.** CW job `13568879` passed
Ruff, then the project-standard Pyrefly run reported 109 diagnostics. Most are
pre-existing repository debt. The Tasks 1-4-owned diagnostics are corrected by
the current type-only follow-up; the controller must rerun standard Pyrefly.

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
reports `torch_cuda_rng_digests.0`. The test now expects the indexed path for
CUDA, while all scalar families retain their existing top-level expectation.
Production comparison behavior is unchanged.

### CW Linux Read-Only Capture Follow-Up

CW job `13566796` exposed a production read-only violation after 160 tests
passed. TorchData 0.11.0 leaves a restored loader with `_iterator is None` and
its checkpoint in `next_iter_state`. Calling `StatefulDataLoader.state_dict()`
at that boundary creates the iterator. PyTorch iterator construction consumes
the loader's shared explicit generator for its base seed, while TorchData moves
the loader from the pending boundary to a live iterator. The audit therefore
changed both states during observation and later rejected validation on
`explicit_generator_digest`.

Production capture now reads and hashes the pending checkpoint directly,
including an explicit `pending` boundary marker, without calling
`state_dict()`. It similarly represents a not-started loader without creating
an iterator. Active and opaque loaders retain their existing state-dict path.
The fix does not reorder generator and loader digests, restore RNG after a
mutation, or suppress `explicit_generator_digest` comparison.

The regression first demonstrates the locked TorchData mutation, then calls
production `capture_correctness_snapshot()` and verifies that the generator,
pending loader state, iterator boundary, and next naturally consumed batch all
remain identical to an equivalent no-capture control.

### Loader Identity Fail-Closed Follow-Up

Review of `45ca880cb` found that private attribute names alone were not a safe
TorchData adapter boundary. A custom protocol loader could be misclassified,
while a recognized TorchData loader with missing or renamed fields could fall
back to the mutating `state_dict()` path. The adapter also defaulted a missing
boundary flag to false and accepted impossible field combinations.

The adapter now recognizes the exact imported
`torchdata.stateful_dataloader.StatefulDataLoader` class and verifies the
installed top-level package owner, distribution name, and exact locked version
`0.11.0` through `importlib.metadata`. A recognized subclass, unknown package
identity/version or runtime class layout, missing field, wrong field type, empty
pending state, or impossible boundary raises `CorrectnessAuditError` before
`state_dict()`. Private runtime-class lookup occurs only inside enabled audit
capture, after package/version validation.
Pending and not-started evidence includes the exact loader class, package, and
package version. Active state calls `state_dict()` only after the complete
layout and boundary are validated. Non-TorchData protocol loaders always use
their own `state_dict()` even when private names collide.

### TorchData Source-Provenance Follow-Up

Review of `2a4c1ca8f` found that installed distribution metadata and matching
class names did not prove that the imported runtime module came from the
distribution. A shadow source earlier on `sys.path` could retain matching class
metadata while executing different code.

Enabled audit capture now finds exactly one
`torchdata/stateful_dataloader/stateful_dataloader.py` entry in
`distribution.files`, requires an available SHA-256 RECORD hash, and decodes
its URL-safe base64 value using the RECORD no-padding convention. It resolves
the PackagePath location, `runtime_module.__spec__.origin`, and
`runtime_module.__file__` with `strict=True` and requires all three canonical
paths to be identical. It then hashes the actual imported source bytes and
requires an exact RECORD digest match before class/layout inspection or any
loader `state_dict()` call. The canonical source origin and hexadecimal
SHA-256 are included in valid TorchData fingerprint evidence.

Missing/duplicate file records, unavailable or unsupported hashes, malformed
RECORD encodings, unresolvable or mismatched origins, content mismatches, and
public/private class module mismatches all fail closed. Non-TorchData loaders
still bypass TorchData metadata and use their protocol `state_dict()` directly.

## Invariant Matrix

| Invariant | Current implementation evidence | Current test/review evidence | Status |
|---|---|---|---|
| Dataset and preprocessing identity are pinned | Manifest provenance derives from active dataset, tokenizer, preprocessing, and container fingerprints | Tasks 1-3 provenance/startup tests; historical Linux gate through `e8e13f5a9` | Pass through Tasks 1-3 |
| Artifact publication is atomic and fail closed | Strict schema and hashes, memory preflight, locked atomic publication | Corrupt, partial, interrupted, and concurrent publication tests | Pass through Tasks 1-3 |
| Canonical precomputed and CPU-cache payloads remain owned and immutable | Loads own CPU tensors; submissions clone canonical CPU/precomputed payloads | Clone, mutation, failed-submission, cache lifecycle tests | Pass in CW job `13568701` |
| Live and precomputed loss weighting use exact valid tokens | Validation weights losses with exact `sample_mask * token_mask` counts and artifact counts | Parity and invalid-loss-shape tests | Pass in CW job `13568701` |
| Driver Python, NumPy, Torch CPU, and Torch CUDA RNG are exact | Production snapshots hash complete states; initialized CUDA devices use `get_rng_state_all()` | Real mutation matrix plus controlled driver CUDA API side effects | Pass in CW job `13568701` |
| Explicit generator and train-loader position are exact | `Generator.get_state()` is hashed directly; exact TorchData 0.11 package, source origin, RECORD SHA-256, class, and layout are verified; pending/not-started state is hashed without creating an iterator; active state is read only after validation | Real restart/control test plus package/source provenance, identity, version, missing-field, bad-type, impossible-boundary, and colliding-loader tests | Pass in CW job `13568701` |
| Runtime validation payload survives exceptions | External collector exists independently of result and finalizes after restoration attempt | Mutating submission-failure, invalid-loss, and restore-failure production tests | Pass in CW job `13568701` |
| Ordered runtime sample identity is exact | Evidence independently hashes ordered `idx` and `input_ids` | Production path mutates only `idx` and asserts sample-identity rejection | Pass in CW job `13568701` |
| Runtime exact token counts are independent evidence | Counts are recomputed from current `sample_mask * token_mask` before and after submission | Production path mutates only a mask and asserts token-count rejection | Pass in CW job `13568701` |
| Next train batch is natural and comparable | Successful records remain pending until the existing training iterator yields a batch; explicit control comparison API gates it | Natural-batch, no-validation control, and restart/resume tests | Pass in CW job `13568701` |
| Every worker rank is represented | Concrete Policy RPC calls every worker, validates ranks, and sorts records | Multi-rank routing, duplicate-rank, and order tests | Pass in CW job `13568701` |
| Worker Torch CUDA RNG is read-only and mutation-sensitive | Worker calls `torch.cuda.get_rng_state(current_device)` | Controlled API side effects pass through production worker capture and gate | Pass in CW job `13568701` |
| MCore RNG is read without initialization | Worker calls only direct `get_all_rng_states()` and rejects assertion, empty, and non-mapping results | Controlled tracker mutation, uninitialized, empty, non-mapping, and source-guard tests | Pass in CW job `13568701` |
| Model parameters, buffers, and training mode are stable | Direct named traversal records local fingerprints and every module mode | Production model and independent mode mutation tests | Pass in CW job `13568701` |
| Optimizer main shards, state tensors, and steps are stable | Every param-group tensor and direct state tensor has owner, shape/dtype, moments, samples; exact steps are separate | Main-shard, `exp_avg`, and step mutation tests | Pass in CW job `13568701` |
| Forbidden worker paths remain absent | Fingerprint uses no state/load/checkpoint, mode change, parameter sync, seed/setter, forward, or optimizer step | Source-isolated and unit AST guards | Covered locally by source guard |
| Failed validation cannot publish cache state | Cache publication occurs only after validation and restoration return successfully | Existing failed-submission, invalid-loss, restore-failure, and atomic-cache tests | Pass in CW job `13568701` |
| Audit overhead is isolated | Collector and worker audit timing are separate and removed from validation/step/loop windows | Fixed-timer, separate logging, and source path checks | Pass in CW job `13568701` |
| Disabled mode remains unchanged | No auditor/collector construction and direct submission/restoration branches | Disabled-path RPC/read assertions and source inspection | Pass in CW job `13568701`; unchanged by typing fix |
| Interfaces and unrelated backends remain scoped | Only concrete Policy method exists; PolicyInterface and unrelated backends are unchanged | Source inspection | Pass |

## Verification Evidence

Passed locally on the current static typing follow-up:

- `rtk proxy /opt/homebrew/bin/pytest -q --disable-warnings tests/source_isolated/test_sft_event_batch_source.py`
  returned `45 passed in 0.98s` on the final rerun.
- Ruff lint passed on all changed Python files.
- Ruff format check passed on all changed Python files.
- Python compilation passed for all changed Python files and requested focused
  test modules.
- `git diff --check` passed.
- Pinned standalone Pyrefly reports no Task-owned diagnostic on a changed line.
  Its 12 residual focused diagnostics are eight missing local dependency
  imports plus four pre-existing `sft.py` diagnostics at lines 126, 418, and
  1256 (two diagnostics).

Blocked locally:

- `/opt/homebrew/bin/pytest -q tests/unit/algorithms/test_sft_correctness_audit.py tests/unit/algorithms/test_sft.py tests/unit/algorithms/test_sft_validation_artifact.py`
  stops in `tests/unit/conftest.py` with
  `ModuleNotFoundError: No module named 'ray'`; no focused unit test runs.
- No current Task 4 GPU/Ray/Megatron integration or full repository suite ran
  locally.
- The project-standard CW Pyrefly rerun and post-fix review are pending.

CW Linux partial result:

- Job `13566467`: `105 passed, 1 failed`, then stopped at `maxfail=1`.
- Failure:
  `tests/unit/algorithms/test_sft_correctness_audit.py::test_correctness_gate_rejects_each_driver_state_family[cuda-rng]`.
- Expected: `("torch_cuda_rng_digests",)`.
- Actual: `("torch_cuda_rng_digests.0",)`.
- At that point, a complete rerun after the test-contract correction remained
  pending; later job `13568701` is the passing functional gate.

- Job `13566796`: `160 passed, 1 failed`, then stopped at `maxfail=1`.
- Failure:
  `tests/unit/algorithms/test_sft.py::test_restart_restores_loader_and_generator_then_runs_real_validation`.
- Gate difference: `explicit_generator_digest`.
- At that point, a complete rerun after the read-only capture correction
  remained pending; later job `13568701` is the passing functional gate.

CW Linux final functional result:

- Job `13568701` at `61d7ff5d9`: `269 passed, 47 warnings in 47.25s`, with
  isolated `RAY_TMPDIR`/`TMPDIR` and unset `RAY_ADDRESS`.
- Job `13568879`: Ruff passed; full project-standard Pyrefly reported 109
  diagnostics. The current follow-up fixes only the Tasks 1-4-owned
  diagnostics. A standard Pyrefly rerun is pending.

## Static Typing Follow-Up

The current follow-up adds explicit post-check narrowing for TorchData package
metadata, tensors, event payload evidence, and scalar timing. It moves the
successful audit and artifact-publication returns after their existing
`finally` blocks so Pyrefly sees complete return paths without changing cleanup,
evidence, or exception precedence. The Hydra-resolved producer mapping is
typed as dynamic only after its existing runtime mapping check.

The audit-only event payload binding and assertion remain inside the enabled
collector branch. The disabled path gains no worker RPC, state read, reduction,
synchronization, RNG access, or timing call. A source regression verifies both
successful returns remain after their corresponding `finally` blocks.

## Residual Limits

- Fixed samples and moments can miss a localized tensor mutation. Increase the
  deterministic sample count when stronger fingerprint evidence is required.
- Audit mode intentionally performs worker RPCs, local tensor reductions, and
  host-visible state reads. It must remain disabled in timed performance runs.
- Precomputed artifacts omit producer-only `idx`; ordered persisted
  `input_ids` remain their sample-identity evidence. Live and CPU-cache runtime
  paths independently include `idx` when present.
- Merge readiness still requires controller approval and a project-standard CW
  Pyrefly rerun of the current static typing follow-up.
