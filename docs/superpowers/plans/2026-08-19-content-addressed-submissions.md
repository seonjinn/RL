# Content-Addressed Campaign Submissions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make MCore and Bridge dry-runs leave zero durable entries and make actual submissions reuse one verified content-addressed source snapshot while transactionally owning their unique intent.

**Architecture:** Add a stdlib-only submission lifecycle module that owns snapshot publication, intent publication, scheduler invocation, and rollback in one Python process. Both shell submitters retain their provenance/profile/row validation but delegate artifact and scheduler lifecycle to this shared implementation.

**Tech Stack:** Python 3.13 dataclasses/enums/pathlib/subprocess, Bash, Git archive, pytest.

**Spec:** `docs/superpowers/specs/2026-08-19-campaign-inode-lifecycle-design.md`

## Global Constraints

- Preserve exact source SHA, directory digest, profile, runtime, container, typed-row, and attestation checks.
- `TEST_ONLY=1` contacts no scheduler and leaves zero entries under durable `RUN_LOG_ROOT`.
- `SBATCH_TEST_ONLY=1` contacts the scheduler exactly once with literal `--test-only` and leaves zero entries under durable `RUN_LOG_ROOT`.
- Actual submissions use `RUN_LOG_ROOT/source-snapshots/<kind>/<commit>/<snapshot_sha256>/` and never create a second verified snapshot for identical content.
- Scheduler rejection removes the unique intent; shared verified snapshots are preserved for safe reuse.
- Publication is same-filesystem, atomic, no-clobber, and fail-closed on mismatched or unsafe existing content.
- No destructor-driven cleanup, raw command payload, push, SLURM job, or remote deletion is allowed in this plan.

---

## File Structure

- Create `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/submission_lifecycle.py`: typed artifact lifecycle and scheduler executor.
- Modify `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/run_mcore_training.py`: re-export snapshot/intent verification interfaces for worker compatibility.
- Modify `experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_mcore_matrix.sh`: construct typed requests and delegate lifecycle.
- Modify `experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_bridge_matrix.sh`: adopt identical modes and lifecycle.
- Create `tests/unit/experiments/test_campaign_submission_lifecycle.py`: pure lifecycle/concurrency tests.
- Modify `tests/unit/experiments/test_mcore_standalone_driver.py`: real MCore launcher contract tests.
- Modify `tests/unit/experiments/test_matrix_submitters.py`: MCore/Bridge parity and reserved-environment tests.

### Task 1: Content-addressed snapshot cache

**Files:**
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/submission_lifecycle.py`
- Create: `tests/unit/experiments/test_campaign_submission_lifecycle.py`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/run_mcore_training.py:34-270`
- Modify: `tests/unit/experiments/test_mcore_standalone_driver.py:450-550`

**Interfaces:**
- Produces: `SubmissionMode`, `ArchiveSource`, `SubmissionArtifacts`, `SubmissionTransaction`, `remove_owned_intent()`, `prepare_candidate_submission()`, `verify_source_snapshot()`, `load_submission_intent()`.
- Consumes: existing `_directory_sha256()`, Git repositories, full commit/digest validation rules.

- [ ] **Step 1: Write the failing cache-reuse and safety tests**

```python
def test_identical_candidate_reuses_one_content_addressed_snapshot(tmp_path: Path) -> None:
    repository, commit = git_repository(tmp_path / "candidate")
    first = prepare_actual(repository, commit, tmp_path / "logs")
    before = tree_identity(first.snapshot_root)
    second = prepare_actual(repository, commit, tmp_path / "logs")
    assert second.snapshot_root == first.snapshot_root
    assert second.snapshot_sha256 == first.snapshot_sha256
    assert tree_identity(second.snapshot_root) == before
    assert second.intent_path != first.intent_path

@pytest.mark.parametrize("mutation", ("writable", "digest", "symlink"))
def test_existing_unsafe_snapshot_fails_closed(tmp_path: Path, mutation: str) -> None:
    artifact = prepare_fixture(tmp_path)
    mutate_snapshot(artifact.snapshot_root, mutation)
    with pytest.raises(ValueError, match="snapshot"):
        prepare_fixture(tmp_path)
```

Also add archive-failure, intent-failure, escaping destination, reserved marker, unsupported file type, and temporary/claim residue assertions. Replace the old assertion that two preparations must have different snapshot roots.

- [ ] **Step 2: Run the focused tests and record RED**

Run: `.venv/bin/pytest -q --confcutdir=tests/unit/experiments tests/unit/experiments/test_campaign_submission_lifecycle.py tests/unit/experiments/test_mcore_standalone_driver.py -k 'submission_preparation or content_addressed or unsafe_snapshot'`

Expected: FAIL because `submission_lifecycle.py` and content-addressed reuse do not exist and the old implementation returns UUID paths.

- [ ] **Step 3: Implement the typed lifecycle primitives**

```python
class SubmissionMode(StrEnum):
    TEST_ONLY = "test_only"
    SBATCH_TEST_ONLY = "sbatch_test_only"
    ACTUAL = "actual"

@dataclass(frozen=True)
class ArchiveSource:
    repository: Path
    commit: str
    relative_destination: Path

@dataclass(frozen=True)
class SubmissionArtifacts:
    snapshot_root: Path
    snapshot_sha256: str
    intent_path: Path
    intent_sha256: str

@dataclass
class SubmissionTransaction:
    artifacts: SubmissionArtifacts
    artifact_root: Path
    mode: SubmissionMode
    snapshot_created: bool
    _scheduler_accepted: bool = False
    _closed: bool = False

    def commit_scheduler_acceptance(self) -> None:
        if self._closed:
            raise RuntimeError("submission transaction is closed")
        self._scheduler_accepted = True

    def close(self) -> None:
        if self._closed:
            return
        if not self._scheduler_accepted:
            remove_owned_intent(self.artifacts.intent_path)
        self._closed = True

    def __enter__(self) -> SubmissionTransaction:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
```

Implement `prepare_candidate_submission(*, archive_sources: tuple[ArchiveSource, ...], artifact_root: Path, mode: SubmissionMode, candidate_kind: Literal["mcore", "bridge"], candidate_sha: str, intent_payload: Mapping[str, object]) -> SubmissionTransaction`. Build into a private sibling, compute the digest before choosing the final path, use an `O_CREAT|O_EXCL` candidate claim, verify any concurrent winner, fsync files/directories, and remove only owned unpublished state. Do not implement `__del__`.

Implement `remove_owned_intent(path)` to require the exact transaction-created regular non-symlink path beneath `artifact_root/submission-intents`, unlink the leaf only, fsync its parent, and reject path escapes or a changed inode.

- [ ] **Step 4: Add concurrent publisher RED/GREEN coverage**

```python
def test_concurrent_identical_publishers_converge(tmp_path: Path) -> None:
    with ThreadPoolExecutor(max_workers=2) as executor:
        transactions = tuple(executor.map(lambda _: prepare_fixture(tmp_path), range(2)))
    assert len({item.snapshot_root for item in transactions}) == 1
    assert sum(item.snapshot_created for item in transactions) <= 1
    assert not tuple((tmp_path / "logs").rglob("*.tmp"))
    assert not tuple((tmp_path / "logs").rglob("*.claim"))
```

Run the focused test from Step 2 until GREEN.

- [ ] **Step 5: Preserve worker-facing imports and commit**

Re-export `verify_source_snapshot`, `load_submission_intent`, and compatibility artifact fields from `run_mcore_training.py`. Run `py_compile` on both modules and commit only Task 1 files with `git commit -s -m 'feat: cache typed candidate snapshots'`.

### Task 2: Transaction-owned scheduler execution

**Files:**
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/submission_lifecycle.py`
- Modify: `tests/unit/experiments/test_campaign_submission_lifecycle.py`

**Interfaces:**
- Consumes: `SubmissionTransaction` from Task 1.
- Produces: `TypedSchedulerRequest`, `TypedSchedulerResult`, `run_one_typed_submission()`, `run_typed_matrix_submissions()`.

- [ ] **Step 1: Write scheduler lifecycle RED tests**

```python
@pytest.mark.parametrize("mode", (SubmissionMode.TEST_ONLY, SubmissionMode.SBATCH_TEST_ONLY))
def test_dry_modes_leave_zero_durable_entries(tmp_path: Path, mode: SubmissionMode) -> None:
    before = inode_paths(tmp_path / "durable")
    result = run_fixture_submission(tmp_path, mode=mode)
    assert inode_paths(tmp_path / "durable") == before
    assert result.scheduler_contact_count == (0 if mode is SubmissionMode.TEST_ONLY else 1)

def test_scheduler_rejection_rolls_back_unique_intent(tmp_path: Path) -> None:
    result = run_fixture_submission(tmp_path, scheduler_exit=2)
    assert result.returncode == 2
    assert not tuple((tmp_path / "durable" / "submission-intents").rglob("*.json"))
```

Add accepted, repeated-actual, exact `--test-only`, in-call digest inspection, explicit-empty mode, signal cleanup, and multi-row partial-acceptance tests.

- [ ] **Step 2: Run focused tests and record RED**

Run: `.venv/bin/pytest -q --confcutdir=tests/unit/experiments tests/unit/experiments/test_campaign_submission_lifecycle.py -k scheduler`

Expected: FAIL because the scheduler request/result API is absent.

- [ ] **Step 3: Implement scheduler request ownership**

```python
@dataclass(frozen=True)
class TypedSchedulerRequest:
    row_id: str
    argv_template: tuple[str, ...]
    intent_payload: Mapping[str, object]

@dataclass(frozen=True)
class TypedSchedulerResult:
    row_id: str
    returncode: int
    stdout: str
    stderr: str

def run_typed_matrix_submissions(
    *,
    requests: tuple[TypedSchedulerRequest, ...],
    archive_sources: tuple[ArchiveSource, ...],
    durable_root: Path,
    mode: SubmissionMode,
    candidate_kind: Literal["mcore", "bridge"],
    candidate_sha: str,
) -> tuple[TypedSchedulerResult, ...]:
    return tuple(
        run_one_typed_submission(
            request=request,
            archive_sources=archive_sources,
            durable_root=durable_root,
            mode=mode,
            candidate_kind=candidate_kind,
            candidate_sha=candidate_sha,
        )
        for request in requests
    )
```

The function must allocate a private temporary root for both dry modes, replace only typed `{snapshot_root}`, `{snapshot_sha256}`, `{intent_path}`, and `{intent_sha256}` argv fields, scrub inherited `SBATCH_*`, invoke argv without a shell, commit only after scheduler acceptance, close on every failure/signal, and preserve earlier accepted rows if a later row fails.

`run_one_typed_submission()` owns the context-manager scope for exactly one row; `run_typed_matrix_submissions()` is only the ordered tuple comprehension shown above. A nonzero scheduler result is returned after `close()` rolls back its intent, while an accepted result calls `commit_scheduler_acceptance()` before scope exit.

- [ ] **Step 4: Run scheduler and full lifecycle tests GREEN**

Run: `.venv/bin/pytest -q --confcutdir=tests/unit/experiments tests/unit/experiments/test_campaign_submission_lifecycle.py`

Expected: all tests pass; durable inode delta is exactly zero for both dry modes.

- [ ] **Step 5: Commit**

Run static checks, then commit Task 2 files with `git commit -s -m 'feat: transact typed scheduler submissions'`.

### Task 3: MCore submitter integration

**Files:**
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_mcore_matrix.sh:174-258`
- Modify: `tests/unit/experiments/test_mcore_standalone_driver.py:1180-1430`

**Interfaces:**
- Consumes: `run_typed_matrix_submissions()` from Task 2.
- Produces: unchanged `ROW`, `SBATCH`, and `SBATCH_OUTPUT` user-visible contracts.

- [ ] **Step 1: Add real-shell RED tests**

Extend `MCoreSubmitterHarness` so fake `sbatch` validates artifacts during invocation and records durable paths rather than reading deleted dry-run paths afterward. Assert TEST_ONLY `0` contacts/`0` durable delta, SBATCH_TEST_ONLY `1` contact/`0` durable delta, rejected actual no intent, accepted repeated actual one snapshot/two intents, and identical normalized command builders.

- [ ] **Step 2: Run RED**

Run: `.venv/bin/pytest -q --confcutdir=tests/unit/experiments tests/unit/experiments/test_mcore_standalone_driver.py -k mcore_submitter`

Expected: SBATCH_TEST_ONLY creates a durable snapshot and repeated actual creates two snapshot directories.

- [ ] **Step 3: Delegate the MCore lifecycle to Python**

Replace the prepare-only heredoc and shell `sbatch` loop with one heredoc that builds typed row requests and calls `run_typed_matrix_submissions()`. Keep all existing branch, remote SHA, clean-tree, profile, runtime-feature, allocation, segment, and raw-command rejection gates before delegation. Render results using the existing output labels.

- [ ] **Step 4: Run MCore tests GREEN and commit**

Run the focused command, `bash -n submit_mcore_matrix.sh`, and the entire standalone driver module. Commit with `git commit -s -m 'fix: bound MCore submission artifacts'`.

### Task 4: Bridge submitter integration

**Files:**
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_bridge_matrix.sh:1-160`
- Modify: `tests/unit/experiments/test_matrix_submitters.py`

**Interfaces:**
- Consumes: Task 2 lifecycle, including nested `ArchiveSource` entries.
- Produces: Bridge mode validation and zero-durable dry-run behavior identical to MCore.

- [ ] **Step 1: Build a real Bridge harness and record RED**

Create a nested Bridge/MCore fixture and parameterize MCore/Bridge over TEST_ONLY, SBATCH_TEST_ONLY, rejected actual, accepted actual, and repeated actual. Explicitly assert that Bridge rejects unset-invalid/empty/mutually-exclusive modes before external work.

- [ ] **Step 2: Run RED**

Run: `.venv/bin/pytest -q --confcutdir=tests/unit/experiments tests/unit/experiments/test_matrix_submitters.py -k 'bridge and (test_only or submission)'`

Expected: Bridge currently writes durable artifacts for scheduler dry-run and lacks equivalent TEST_ONLY lifecycle.

- [ ] **Step 3: Migrate Bridge to the shared executor**

Use two `ArchiveSource` values, for Bridge at `.` and nested MCore at `3rdparty/Megatron-LM`. Add exact unset-only mode defaults and validation before external work. Preserve provenance, runtime, allocation, segment, GRES, output, and reserved-environment behavior.

- [ ] **Step 4: Run both submitter suites GREEN and commit**

Run both test modules and `bash -n` on both submitters. Commit with `git commit -s -m 'fix: bound Bridge submission artifacts'`.

### Task 5: Regression and independent review gate

**Files:**
- Modify only files required by failures proven in this task.

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces: reviewed submission-lifecycle commit range ready for the rank-evidence plan.

- [ ] **Step 1: Run the complete compatible regression set**

Run the three experiment test modules, `bash -n` for both submitters and `run_mcore_scope.sub`, Ruff on changed Python files, `py_compile`, and `git diff --check`.

- [ ] **Step 2: Run adversarial inode accounting**

Execute one filesystem-backed cycle of TEST_ONLY, SBATCH_TEST_ONLY, actual, and repeated actual. Record before/after path sets and assert `0`, `0`, bounded intent/log growth, and no second snapshot respectively.

- [ ] **Step 3: Request independent review**

Require review of atomic no-clobber races, signal cleanup, scheduler rejection, Bridge parity, and retained provenance checks. Resolve all blocker/high/medium findings through RED/GREEN fix rounds.

- [ ] **Step 4: Record completion**

Append exact commands/results/commit range to this plan's SDD ledger. Do not push or submit a job.
