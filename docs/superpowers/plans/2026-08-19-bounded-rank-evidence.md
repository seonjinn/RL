# Bounded Rank Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace persistent per-rank exchange trees with one deterministic verified archive and one small index for every successful, failed, or catchably signaled distributed run.

**Architecture:** Keep the shared-filesystem JSON exchange during execution, then make the outer batch shell—not rank 0—the sole finalization owner. A stdlib finalizer freezes the exact run tree, creates and verifies a deterministic tar plus canonical index, atomically publishes them, and only then removes transient files; publication failures move the frozen tree into quarantine.

**Tech Stack:** Python 3.13 tarfile/hashlib/json/pathlib, Bash signal handling, pytest.

**Spec:** `docs/superpowers/specs/2026-08-19-campaign-inode-lifecycle-design.md`

## Global Constraints

- Preserve all rank identity, topology, pytest-node, capability, and final result validation.
- Bind evidence to `(scheduler_job_id, restart_count, submission_intent_sha256)`.
- Success and failure both publish bounded evidence; missing ranks are explicit.
- Delete a transient tree only after archive and index atomically publish and independently re-verify.
- Finalizer/publication uncertainty quarantines the frozen tree and fails closed.
- `HUP`, `INT`, and `TERM` finalize once and preserve exits `129`, `130`, and `143`; SIGKILL/node loss remain retention inputs.
- No push, SLURM job, remote deletion, or CUDA-capture validation change is allowed.

---

## File Structure

- Create `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/finalize_mcore_rank_evidence.py`: deterministic bundle/index finalizer CLI.
- Modify `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/run_mcore_training.py`: safe path helpers shared with finalizer.
- Modify `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/run_mcore_scope.sub`: worker status/signal ownership and finalizer invocation.
- Modify `experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_mcore_matrix.sh`: export exact finalizer path.
- Modify `experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_bridge_matrix.sh`: export exact finalizer path inherited by Bridge scope.
- Create `tests/unit/experiments/test_mcore_rank_evidence.py`: pure finalizer tests.
- Modify `tests/unit/experiments/test_mcore_standalone_driver.py`: wrapper, signal, and staged-runtime regression tests.

### Task 1: Evidence paths and deterministic archive format

**Files:**
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/run_mcore_training.py:610-680`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/finalize_mcore_rank_evidence.py`
- Create: `tests/unit/experiments/test_mcore_rank_evidence.py`

**Interfaces:**
- Produces: `rank_evidence_paths()`, `RankEvidencePaths`, `EvidenceIndex`, `build_rank_evidence_bundle()`, `verify_rank_evidence_bundle()`.
- Consumes: existing `derive_run_identity()`, `rank_result_dir()`, `result_path()` validation.

- [ ] **Step 1: Write path and deterministic-archive RED tests**

```python
def test_bundle_is_deterministic_across_creation_order_and_mtime(tmp_path: Path) -> None:
    first = rank_tree(tmp_path / "first", reverse=False, mtime=1)
    second = rank_tree(tmp_path / "second", reverse=True, mtime=2)
    first_bundle = build_fixture_bundle(first)
    second_bundle = build_fixture_bundle(second)
    assert first_bundle.tar_sha256 == second_bundle.tar_sha256
    assert first_bundle.index_bytes == second_bundle.index_bytes

@pytest.mark.parametrize("unsafe", ("escape", "symlink", "fifo", "oversize", "too_many"))
def test_bundle_rejects_unsafe_or_unbounded_tree(tmp_path: Path, unsafe: str) -> None:
    tree = unsafe_rank_tree(tmp_path, unsafe)
    with pytest.raises(ValueError):
        build_fixture_bundle(tree)
```

Add candidate-kind/SHA/row/run-identity escape tests and enforce at most 1,024 files and 64 MiB uncompressed evidence.

- [ ] **Step 2: Run RED**

Run: `.venv/bin/pytest -q --confcutdir=tests/unit/experiments tests/unit/experiments/test_mcore_rank_evidence.py -k 'path or deterministic or unsafe'`

Expected: FAIL because the finalizer module and evidence paths are absent.

- [ ] **Step 3: Implement exact paths and canonical format**

```python
@dataclass(frozen=True)
class RankEvidencePaths:
    transient: Path
    finalizing: Path
    archive: Path
    index: Path
    quarantine: Path

@dataclass(frozen=True)
class EvidenceIndex:
    schema_version: int
    run_identity: str
    scheduler_job_id: str
    runner_exit_code: int
    termination_reason: str
    expected_paths: tuple[str, ...]
    present_paths: tuple[str, ...]
    missing_paths: tuple[str, ...]
    archive_sha256: str
    archive_size: int
```

Use `rank-evidence/<kind>/<sha>/<row>/<run_identity>.tar` and `.json`. Add sorted tar members with `mtime=uid=gid=0`, empty owner names, normalized safe modes, no absolute paths, and canonical JSON (`allow_nan=False`, sorted keys, newline). Index expected paths from `world_size` and matrix node count rather than trusting directory contents.

- [ ] **Step 4: Verify independently and commit**

Reopen both files, recompute every member digest/size and the archive digest, reject extra/missing/duplicate members, and ensure index identity matches the requested run. Run focused tests and commit with `git commit -s -m 'feat: define bounded rank evidence'`.

### Task 2: Atomic publication, removal, and quarantine

**Files:**
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/finalize_mcore_rank_evidence.py`
- Modify: `tests/unit/experiments/test_mcore_rank_evidence.py`

**Interfaces:**
- Consumes: Task 1 archive/index builder and verifier.
- Produces: `finalize_rank_exchange()` and the CLI `main()`.

- [ ] **Step 1: Write finalization RED tests**

Cover complete success, collection failure, first-node failure, missing final ranks, byte-identical idempotence, conflicting existing archive, archive/index tamper, symlink destination, finalizing-dir recovery, and failed publication quarantine.

```python
def test_verified_publication_removes_transient_tree(tmp_path: Path) -> None:
    paths = complete_rank_fixture(tmp_path)
    result = finalize_fixture(paths, runner_exit_code=0)
    verify_rank_evidence_bundle(result.archive, result.index)
    assert result.archive.stat().st_mode & 0o222 == 0
    assert not paths.transient.exists()
    assert not paths.finalizing.exists()

def test_publication_failure_preserves_quarantine(tmp_path: Path) -> None:
    paths = incomplete_rank_fixture(tmp_path)
    occupy_with_conflict(paths.archive)
    with pytest.raises(RuntimeError):
        finalize_fixture(paths, runner_exit_code=1)
    assert paths.quarantine.is_dir()
```

- [ ] **Step 2: Run RED**

Run: `.venv/bin/pytest -q --confcutdir=tests/unit/experiments tests/unit/experiments/test_mcore_rank_evidence.py -k finalize`

Expected: FAIL because no publication lifecycle exists.

- [ ] **Step 3: Implement freeze/publish/verify/delete ordering**

Atomically rename the exact transient directory to a hidden same-filesystem `.finalizing` path. Build temporary siblings, fsync, hard-link publish no-clobber, accept only a byte-identical verified existing pair, and verify the published pair through a fresh read. Remove the read-only frozen tree only after verification. On any safety failure, move it no-clobber to `RUN_LOG_ROOT/rank-results-quarantine/<kind>/<candidate_sha>/<row_id>/<run_identity>` and return nonzero.

- [ ] **Step 4: Bind runner status to the current result**

For runner exit `0`, require the content-bound final result to be a regular non-writable JSON with `status=passed` and exact `run_identity`. For nonzero runner status, record failure/missing evidence and reject an inconsistent passed current-run result. Never treat an older row result as evidence for the new run.

- [ ] **Step 5: Run full finalizer tests and commit**

Run the test module, Ruff, `py_compile`, and diff-check. Commit with `git commit -s -m 'feat: finalize distributed rank evidence'`.

### Task 3: Outer batch ownership and signal handling

**Files:**
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/run_mcore_scope.sub:10-260`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_mcore_matrix.sh`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_bridge_matrix.sh`
- Modify: `tests/unit/experiments/test_mcore_standalone_driver.py`

**Interfaces:**
- Consumes: finalizer CLI from Task 2.
- Produces: exactly-once finalization for normal and catchably signaled worker srun.

- [ ] **Step 1: Write fake-srun/finalizer RED tests**

```python
@pytest.mark.parametrize(("signal_name", "expected"), (("HUP", 129), ("INT", 130), ("TERM", 143)))
def test_scope_forwards_signal_and_finalizes_once(tmp_path: Path, signal_name: str, expected: int) -> None:
    process = launch_scope_with_sleeping_worker(tmp_path)
    send_signal(process, signal_name)
    assert process.wait(timeout=10) == expected
    assert finalizer_invocations(tmp_path) == [(expected, f"SIG{signal_name}")]
```

Also test worker exit `0`, worker nonzero, successful worker/failing finalizer, already-failed worker/failing finalizer, and no duplicate finalization.

- [ ] **Step 2: Run RED**

Run: `.venv/bin/pytest -q --confcutdir=tests/unit/experiments tests/unit/experiments/test_mcore_standalone_driver.py -k 'finalizer or signal'`

Expected: bare foreground `srun` exits under `set -e` without finalization.

- [ ] **Step 3: Implement shell lifecycle**

Require/export an absolute `RANK_EVIDENCE_FINALIZER`. Immediately before worker execution, launch worker `srun` in the background, retain its PID, and temporarily disable `set -e` around `wait`. Implement one guarded `finalize_once(exit_code, termination_reason)` call. Signal handlers reset traps, forward the signal to the worker, wait boundedly, finalize once, and exit conventionally. Normal failure preserves the worker status after successful finalization; successful worker plus finalizer failure exits nonzero; a worker already failing retains its primary status while logging finalization failure.

- [ ] **Step 4: Run shell tests GREEN and commit**

Update the staged-runtime harness to preserve its exact verifier+worker `srun` call count and add the finalizer invocation. Run `bash -n` on MCore scope and Bridge scope, then commit with `git commit -s -m 'fix: finalize rank evidence from batch scope'`.

### Task 4: Regression and independent review gate

**Files:**
- Modify only files required by newly proven failures.

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: reviewed bounded-evidence range ready for retention.

- [ ] **Step 1: Run full compatible tests**

Run `test_mcore_rank_evidence.py`, the full standalone driver, matrix submitters, relevant launcher/runtime-attestation suites, `bash -n`, Ruff, `py_compile`, and `git diff --check`.

- [ ] **Step 2: Run inode-bound adversaries**

For success, collection failure, node failure, missing ranks, and each catchable signal, assert the durable result is exactly archive+index+existing fixed result/log entries and the transient tree is absent. For finalizer corruption, assert one bounded quarantine tree and no silent deletion.

- [ ] **Step 3: Request independent review**

Require review of deterministic archive bytes, no-clobber collisions, stale result substitution, signal races, primary exit preservation, quarantine path safety, and Bridge inheritance. Resolve every blocker/high/medium via strict RED/GREEN.

- [ ] **Step 4: Record completion**

Write exact commands/results/commit range to the SDD ledger. Do not push or submit a job.
