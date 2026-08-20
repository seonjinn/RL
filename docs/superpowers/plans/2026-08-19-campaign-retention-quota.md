# Campaign Retention and Inode Admission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent inode exhaustion and unsafe whole-campaign deletion by protecting current/active artifacts, retaining the latest two successes and failures, and rejecting large publication before projected quota use exceeds policy thresholds.

**Architecture:** Add a typed inventory/protection core, a dry-run-first retention CLI with plan-SHA apply, and a separate quota adapter/admission gate. Launchers publish small immutable job bindings and invoke the gate before any large directory creation; runtime/source manifests provide measured entry counts without rescanning whole environments at submit time.

**Tech Stack:** Python 3.13 dataclasses/enums/pathlib/json/subprocess, Linux openat/O_NOFOLLOW safety, Bash, Slurm JSON, Lustre quota output, pytest.

**Spec:** `docs/superpowers/specs/2026-08-19-campaign-inode-lifecycle-design.md`

## Global Constraints

- Always protect the canonical current profile and every `PENDING`, `RUNNING`, or `COMPLETING` campaign job.
- Retain newest two terminal successes and newest two terminal failures per candidate/run key, plus every explicit pin.
- Default retention is mutation-free dry-run; apply requires an exact prior plan SHA and a fresh identical protected set.
- Refuse campaign root, category roots, outside-root paths, symlink traversal, device changes, incomplete scheduler/profile evidence, and unmanaged legacy artifacts.
- Evaluate projected post-operation inode use: below 80% allow, 80%-90% require matching retention report, above 90% reject.
- Missing/malformed quota rejects large creation; only a cached artifact with projected durable delta at most 32 may proceed.
- Preserve existing source/runtime read-only, provenance, marker, and attestation gates.
- No remote cleanup, ownership-marker adoption, push, SLURM submission, or CUDA-capture fix is allowed.

---

## File Structure

- Create `experiments/cuda_graph/nemotron_thd_te_graph_20260731/campaign_inode_lifecycle.py`: layout, inventory, artifact/job/protection models, canonical hashing, path safety.
- Create `experiments/cuda_graph/nemotron_thd_te_graph_20260731/campaign_retention.py`: Slurm adapter, retention selection, dry-run/apply CLI, audit.
- Create `experiments/cuda_graph/nemotron_thd_te_graph_20260731/campaign_inode_quota.py`: strict Lustre parser and admission decision.
- Modify submitters and `run_scope.sh`: campaign job comments and immutable job bindings.
- Modify `scripts/create_source_snapshot.sh`: pre-create gate and measured manifest counts.
- Modify `scripts/validate_oci_container_runtime.sub`: pre-stage gate and final stage count manifest.
- Create three focused test modules and extend existing launcher/container tests.

### Task 1: Campaign layout, inventory, and protected set

**Files:**
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/campaign_inode_lifecycle.py`
- Create: `tests/unit/experiments/test_campaign_inode_lifecycle.py`

**Interfaces:**
- Produces: `CampaignLayout`, `ArtifactRef`, `JobBinding`, `ProtectedSet`, `load_campaign_layout()`, `load_artifact_inventory()`, `compute_protected_set()`, `validate_deletion_target()`.
- Consumes: canonical profile fields from `profile_snapshot.py` and lifecycle manifests emitted by later tasks.

- [ ] **Step 1: Write ownership and deletion-safety RED tests**

```python
@pytest.mark.parametrize("target", ("campaign", "run_log", "runtime", "checkouts", "snapshots", "lifecycle"))
def test_category_roots_can_never_be_deleted(tmp_path: Path, target: str) -> None:
    layout = campaign_layout_fixture(tmp_path)
    with pytest.raises(ValueError, match="root"):
        validate_deletion_target(layout, category_path(layout, target))

@pytest.mark.parametrize("unsafe", ("outside", "symlink", "device", "markerless", "wrong_owner"))
def test_unsafe_target_is_preserved(tmp_path: Path, unsafe: str) -> None:
    layout, artifact = unsafe_artifact_fixture(tmp_path, unsafe)
    with pytest.raises(ValueError):
        validate_deletion_target(layout, artifact)
```

Add exact current-profile closure and active-job closure tests; incomplete profile/job data must produce `ProtectedSet.complete is False`.

- [ ] **Step 2: Run RED**

Run: `.venv/bin/pytest -q --confcutdir=tests/unit/experiments tests/unit/experiments/test_campaign_inode_lifecycle.py`

Expected: FAIL because the lifecycle module does not exist.

- [ ] **Step 3: Implement typed layout and protection models**

```python
@dataclass(frozen=True)
class CampaignLayout:
    campaign_root: Path
    run_log_root: Path
    runtime_validation_root: Path
    source_checkouts_root: Path
    lifecycle_root: Path
    root_device: int
    owner_uid: int

@dataclass(frozen=True)
class ProtectedSet:
    paths: frozenset[Path]
    reasons: Mapping[Path, tuple[str, ...]]
    active_jobs_digest: str
    inventory_digest: str
    complete: bool
    errors: tuple[str, ...]
```

Require `<campaign-root>/.cuda-graph-campaign.json` schema v1 binding canonical root, UID, device, and category roots. Inventory only exact manifest-declared leaf artifacts. Traverse components with `openat`/`O_NOFOLLOW`, use `lstat`, reject device/owner/root changes, and never follow symlinks.

- [ ] **Step 4: Implement current profile and active job closure**

Protect runtime attestation, exact stage/namespace, source checkout, container, run-log root, exact current candidate snapshot, each active job's source/intent/rank/log/runtime references, lifecycle metadata, and pins. Any missing campaign binding, malformed manifest, scheduler uncertainty, or unresolved reference sets `complete=False`.

- [ ] **Step 5: Run GREEN and commit**

Run the test module, Ruff, `py_compile`, and diff-check. Commit with `git commit -s -m 'feat: model protected campaign artifacts'`.

### Task 2: Dry-run-first retention planner and apply

**Files:**
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/campaign_retention.py`
- Create: `tests/unit/experiments/test_campaign_retention.py`

**Interfaces:**
- Consumes: Task 1 inventory/protected set.
- Produces: `SlurmActiveJobAdapter`, `RunKey`, `RetentionPlan`, `build_retention_plan()`, `apply_retention_plan()`, CLI.

- [ ] **Step 1: Write retention-selection RED tests**

```python
def test_retains_latest_two_successes_and_failures_per_run_key(tmp_path: Path) -> None:
    inventory = terminal_runs(tmp_path, successes=4, failures=4)
    plan = build_fixture_plan(inventory)
    assert retained_statuses(plan) == {"passed": (4, 3), "failed": (4, 3)}
    assert deletion_statuses(plan) == {"passed": (2, 1), "failed": (2, 1)}

def test_apply_rejects_changed_protected_set(tmp_path: Path) -> None:
    dry_run = dry_run_fixture(tmp_path)
    add_running_job_binding(tmp_path)
    with pytest.raises(RuntimeError, match="plan SHA"):
        apply_fixture(tmp_path, dry_run.plan_sha256)
```

Cover `FAILED`, `TIMEOUT`, `CANCELLED`, `NODE_FAIL`, explicit pins, unmanaged legacy entries, scheduler failure, missing binding, dry-run zero mutation, apply-without-SHA, partial apply audit, and root/symlink/device adversaries.

- [ ] **Step 2: Run RED**

Run: `.venv/bin/pytest -q --confcutdir=tests/unit/experiments tests/unit/experiments/test_campaign_retention.py`

Expected: FAIL because planner/apply APIs are absent.

- [ ] **Step 3: Implement deterministic retention selection**

```python
@dataclass(frozen=True)
class RunKey:
    candidate_kind: str
    candidate_sha: str
    row_id: str
    variant: str

@dataclass(frozen=True)
class RetentionPlan:
    protected: tuple[ArtifactRef, ...]
    candidates: tuple[ArtifactRef, ...]
    profile_digest: str
    active_jobs_digest: str
    inventory_digest: str
    plan_sha256: str
```

Sort terminal runs by `(finished_at_utc, job_id, run_identity)` descending. Infer no status from mtime or file presence. Compute retained reference closure from result/evidence to intent, candidate snapshot, runtime attestation/stage/namespace, and source checkout.

- [ ] **Step 4: Implement safe dry-run and apply**

Dry-run prints canonical JSON and creates no lock/report/audit entry. Apply requires `--expected-plan-sha256`, acquires the shared campaign mutation lock, re-queries scheduler/profile/inventory, recomputes the plan, prewalks every leaf against manifest counts, then deletes bottom-up. Create a unique `O_EXCL` immutable audit JSON containing per-target expected/observed counts and results. Never broaden a leaf target to its parent.

- [ ] **Step 5: Run GREEN and commit**

Run the module plus Task 1 tests and commit with `git commit -s -m 'feat: add fail-closed campaign retention'`.

### Task 3: Strict quota evidence and admission decisions

**Files:**
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/campaign_inode_quota.py`
- Create: `tests/unit/experiments/test_campaign_inode_quota.py`

**Interfaces:**
- Produces: `QuotaEvidence`, `AdmissionRequest`, `AdmissionDecision`, `parse_lustre_quota()`, `decide_admission()` and CLI.
- Consumes: matching complete retention dry-run report from Task 2.

- [ ] **Step 1: Write threshold/parser RED tests**

```python
@pytest.mark.parametrize(
    ("used", "delta", "report", "allowed"),
    ((79, 0, False, True), (79, 1, False, False), (80, 0, True, True), (90, 0, True, True), (90, 1, True, False)),
)
def test_projected_percentage_policy(used: int, delta: int, report: bool, allowed: bool) -> None:
    decision = decide_fixture(used=used, limit=100, delta=delta, report=report)
    assert decision.allowed is allowed

@pytest.mark.parametrize("bad", ("bool", "negative", "overflow", "human_unit", "multiple_rows", "used_over_limit"))
def test_malformed_quota_fails_closed(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_fixture_quota(bad)
```

Also test unavailable evidence with cached delta `32` allowed and `33` rejected, soft/hard limits, filesystem device mismatch, stale report beyond 15 minutes, and profile/inventory/job digest mismatch.

- [ ] **Step 2: Run RED**

Run: `.venv/bin/pytest -q --confcutdir=tests/unit/experiments tests/unit/experiments/test_campaign_inode_quota.py`

Expected: FAIL because quota APIs are absent.

- [ ] **Step 3: Implement typed parsing and integer policy**

```python
MAX_UNVERIFIED_SMALL_DELTA = 32

@dataclass(frozen=True)
class QuotaEvidence:
    files_used: int
    files_soft_limit: int
    files_hard_limit: int
    source_argv: tuple[str, ...]
    stdout_sha256: str
    observed_at_utc: datetime
    filesystem_device: int

@dataclass(frozen=True)
class AdmissionRequest:
    operation: ArtifactOperation
    projected_inode_delta: int
    cached_artifact: bool
```

Use integer cross-multiplication, not float percentages. Run Lustre quota without human-readable units, bind argv/stdout digest/device, and reject ambiguous rows or invalid integer domains. Validate a retention report's plan/profile/inventory/active-job digests and 15-minute maximum age before accepting the 80%-90% band.

- [ ] **Step 4: Run GREEN and commit**

Run quota and lifecycle tests, static checks, then commit with `git commit -s -m 'feat: gate campaign inode growth'`.

### Task 4: Immutable job bindings

**Files:**
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_mcore_matrix.sh`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_bridge_matrix.sh`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/run_scope.sh`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/validate_oci_container_runtime.sub`
- Modify: `tests/unit/experiments/test_matrix_submitters.py`
- Modify: `tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py`

**Interfaces:**
- Consumes: Tasks 1-3 models and the submission executor from the content-addressed plan.
- Produces: exact scheduler comment and immutable job-binding records for protected-set computation.

- [ ] **Step 1: Write launcher RED tests**

Assert every real submit includes `--comment=cuda-graph-campaign:<campaign-id>`, accepted job IDs create one mode `0444` binding with exact profile/source/intent/rank/log/runtime references, dry modes create none, scheduler rejection creates none, and binding publication failure makes submitter return nonzero without deleting artifacts.

- [ ] **Step 2: Run RED**

Run focused matrix and launcher tests. Expected: no campaign comments or binding records exist.

- [ ] **Step 3: Implement binding publication**

Publish `<campaign-root>/.campaign-lifecycle/job-bindings/<job-id>.json` atomically after scheduler acceptance. Bind job ID, campaign ID, profile SHA, source snapshot, intent, rank root, log, runtime namespace/stage/attestation, and source checkout. Use the shared mutation lock. Do not adopt or rewrite legacy jobs.

- [ ] **Step 4: Run GREEN and commit**

Run focused/full launcher tests and shell syntax. Commit with `git commit -s -m 'feat: bind active campaign jobs to artifacts'`.

### Task 5: Source snapshot and runtime-stage admission

**Files:**
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/create_source_snapshot.sh:70-125`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/validate_oci_container_runtime.sub:150-200,700-1070`
- Modify: `tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py:4570-5050`
- Modify: `tests/unit/experiments/test_container_harness_hardening.py`

**Interfaces:**
- Consumes: quota gate and retention evidence from Tasks 2-3.
- Produces: measured source/stage entry manifests used by subsequent projections.

- [ ] **Step 1: Write gate-ordering RED tests**

Assert a rejected source snapshot creates no `mktemp` directory; a rejected runtime stage creates no keyed root; verified cache hits use delta zero; source manifest contains entry/byte counts; runtime manifest is written and made read-only before marker publication; attestation of an existing stage has delta zero.

- [ ] **Step 2: Run RED**

Run focused launcher/container harness tests. Expected: creation begins without quota admission and no measured count manifests exist.

- [ ] **Step 3: Gate source snapshot publication**

Compute projected entries from recursive Git tree metadata before `mktemp`, invoke the gate, and write schema/entry/byte counts into the source manifest. Existing exact verified snapshots bypass large creation with delta zero.

- [ ] **Step 4: Gate runtime-stage publication**

Before creating `RUNTIME_STAGE_ROOT`, use the largest completed matching stage manifest plus 5% safety margin and source-tree delta. If no matching manifest exists, use a conservative fixed projection of `500000`; never treat unknown as zero. After successful construction, measure once during the existing full-tree walk, publish the read-only manifest before the stage marker, and verify it with the stage.

- [ ] **Step 5: Run GREEN and commit**

Run the affected launcher/container modules, `bash -n`, Ruff, `py_compile`, and diff-check. Commit with `git commit -s -m 'fix: gate large campaign artifacts by inode quota'`.

### Task 6: Full integration and independent safety review

**Files:**
- Modify only files required by failures proven here.

**Interfaces:**
- Consumes: all prior tasks and the preceding submission/rank-evidence plans.
- Produces: reviewed local branch with bounded inode lifecycle.

- [ ] **Step 1: Run all focused and compatible experiment suites**

Run all new lifecycle/retention/quota/evidence tests, standalone driver, matrix submitters, launcher, runtime-attestation, and container harness tests, followed by all shell syntax and Python static checks.

- [ ] **Step 2: Run filesystem-backed lifecycle simulation**

Simulate two candidates with four successes, four failures, one current profile, one pending job, one running job, one completing job, a stale SIGKILL tree, a symlink attack, and a changed-inventory race. Prove dry-run mutation zero, apply retains exact 2+2/current/active/pinned closure, and roots remain untouched.

- [ ] **Step 3: Run admission boundary simulation**

Prove exact behavior at projected 79.99%, 80%, 90%, and above 90%; unavailable quota cached deltas 32/33; stale/mismatched retention evidence; and rejection before any large directory creation.

- [ ] **Step 4: Request independent review**

Require review of deletion targets, openat/symlink/device safety, scheduler binding races, plan-SHA TOCTOU, quota parser ambiguity, runtime marker ordering, current profile closure, and latest-two success/failure selection. Resolve all blocker/high/medium findings through RED/GREEN.

- [ ] **Step 5: Record completion and stop before operations**

Record exact tests/commits in the SDD ledger. Do not initialize remote ownership markers, delete legacy artifacts, push, or submit jobs without separate authorization.
