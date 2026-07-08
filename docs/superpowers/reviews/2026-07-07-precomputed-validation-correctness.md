# Precomputed Validation Correctness Review

## Scope

- Feature base: `c1a415dae1e0bc909eb1891b0d78be92da35e50f`
- Approved Tasks 1-3 head: `e8e13f5a9e0694adb1a574fd4b7e35507ab3ca9b`
- Task 4 implementation: `2c8bfdc5d63d8db05da34258148aea640222e311`
- Inputs: `.superpowers/sdd/task-4-brief.md` and
  `.superpowers/sdd/task-4-api-research.md`

The review covers dataset identity, artifact ownership, validation loss
weighting, driver and worker RNG isolation, model and optimizer state,
training-mode restoration, checkpoint continuation, memory accounting,
disabled-path compatibility, and audit timing isolation.

## Decision

**Code review gate: PASS.** No Critical or Important finding remains.

**Supported-Linux execution gate: PENDING.** The local macOS host cannot collect
the Ray-dependent unit suites because `ray` is not installed. The Task 4 unit
suite must run in the reviewed Linux NeMo-RL environment before merge. Tasks
1-3 passed their combined Linux gate at the approved head: 177 tests passed in
CW job `13559835`.

Worker tensor moments and deterministic samples are fingerprints. They are
evidence of unchanged local state, not a cryptographic equality proof.

## Independent Review

Independent read-only Codex review session
`019f40c6-3d07-7b20-84c5-80e0a1e30c06` inspected Task 4 commit
`2c8bfdc5d63d8db05da34258148aea640222e311` against its parent and returned:

- Findings: none
- Verdict: LGTM

The CLI needed a resumed final-response request after its initial diff output,
but the successful review did not modify the worktree.

## Invariant Matrix

| Invariant | Implementation evidence | Test/review evidence | Result |
|---|---|---|---|
| Dataset identity is externally pinned | Runtime derives expected provenance from active config/source and trusted dataset, tokenizer, and container SHA-256 values | Task 3 startup/mismatch tests and approved Linux gate | Pass |
| Producer supports deterministic prepacked data only | Eligibility rejects raw online packing, stochastic preprocessing, dynamic batching, multimodal data, and train-derived validation | Producer eligibility tests | Pass |
| Canonical artifact remains CPU-owned and immutable | Load returns owning CPU tensors; each submission clones canonical event data | Round-trip, CPU-only, repeated-clone, mutation, and failed-submission tests | Pass |
| Corrupt or partial artifacts fail closed | Strict manifest/tensor schema, content hashes, memory preflight, and atomic publication | Corrupt-byte, malformed-manifest, interrupted-publish, and writer-serialization tests | Pass |
| Live and precomputed loss weighting match | Both paths use ordered global-batch losses and exact valid-token counts | Parity and invalid-loss-shape tests | Pass |
| Python RNG is unchanged | Full Python state is canonically digested at both boundaries | Driver mutation matrix and read-only capture test | Pass |
| NumPy RNG is unchanged | Full NumPy state array and metadata are digested | Driver mutation matrix and artifact-load isolation test | Pass |
| Torch CPU/CUDA RNG is unchanged | CPU state and every initialized driver CUDA state are digested | Driver mutation matrix and read-only capture test | Pass |
| Explicit generator is unchanged | `Generator.get_state()` is digested when the loader exposes one | Read-only capture and resume/restart tests | Pass |
| Train-loader position is unchanged | Train-loader `state_dict()` is digested at both boundaries | Loader mutation and resume/restart tests | Pass |
| Validation payload, sample identity, and token counts are stable | Precomputed tensors, ordered `input_ids`, and exact per-batch counts are digested at both boundaries | Mutable-payload and driver mutation tests | Pass |
| Next train batch is not prefetched | Digesting occurs only after the existing loop naturally yields the batch | Natural-batch auditor and SFT wiring tests | Pass |
| Every worker rank is represented | Concrete Policy method calls every worker and rank records are sorted; duplicate ranks fail | Multi-rank routing and order tests | Pass |
| Torch CUDA RNG is read without advancing it | Worker uses `torch.cuda.get_rng_state(current_device)` | Read-only worker test and source guard | Pass |
| MCore tracker is read without initialization | Worker calls only `get_all_rng_states()` and fails on uninitialized state | Tracker-failure test and source guard | Pass |
| Model state remains unchanged | Direct parameter/buffer traversal records local shape, dtype, moments, and fixed sample hashes | Worker read-only and mutation tests | Pass |
| Optimizer state remains unchanged | Direct wrapper-aware traversal records tensor moments and exact step counters | Worker read-only and mutation tests | Pass |
| Training mode remains unchanged | Every named module training flag is recorded; audit never changes mode | Worker read-only, mutation, and source tests | Pass |
| Forbidden state paths are absent | Audit avoids state/load/checkpoint, parameter sync, mode, seed, RNG setter, forward, and optimizer-step APIs | Transitive AST guard and independent review | Pass |
| Failed validation cannot publish mutable cache state | Cache publication follows loss validation and successful mode restore | Invalid-loss, restore-failure, and atomic-cache tests | Pass |
| Checkpoint/restart evidence is stable | Resumed generator and loader states reproduce the same snapshot digest | Resume/restart test | Pass |
| Memory checks remain fail closed | Artifact load/submission retain copy-count, payload, host, and verified Ray headroom checks | Memory-budget and deep-payload tests | Pass |
| Disabled mode has no audit work | Auditor construction is inside the enabled branch; all defaults are false | Disabled-path test and source inspection | Pass |
| Audit overhead is excluded | Boundary audit time is outside validation timing and subtracted from step/loop windows; next-batch time has its own prefix | Fixed-timer and separate-logging tests | Pass |
| Unrelated interfaces/backends are unchanged | `PolicyInterface` is unchanged; non-Megatron enablement fails before RPC | Source inspection and independent review | Pass |

## Verification Evidence

Passed locally:

- `pytest -q tests/source_isolated/test_sft_event_batch_source.py`: 16 passed
- Ruff check and format check on the requested source set and Task 4 tests
- Python compilation for all changed Python and test files
- Dependency-light execution of the real snapshot ordering/mutation gate
- Dependency-light execution of the real boundary/natural-batch flow
- Transitive worker source guard for forbidden state, mode, sync, and RNG APIs
- `git diff --check`

Environment-blocked locally:

- Focused unit collection stops in `tests/unit/conftest.py` with
  `ModuleNotFoundError: No module named 'ray'`.
- The requested Pyright command reports missing local Torch, Ray, Megatron,
  Pydantic, TorchData, Transformers, OmegaConf, and Safetensors imports plus
  pre-existing repository diagnostics. Focused cleanup left the new audit
  module with only its unresolved local Torch import, and the new worker helper
  line range with only unresolved Torch/Megatron imports.

## Residual Limits

- Worker moments and fixed samples can miss localized changes; increase the
  deterministic sample count when stronger evidence is required.
- Audit mode intentionally performs local GPU reductions and host-visible
  state reads. It is for correctness gates, not timed performance runs.
- The artifact omits producer-only `idx` metadata. Ordered `input_ids` provide
  persisted precomputed sample identity.
- Linux Ray/Megatron execution remains the final merge prerequisite.
