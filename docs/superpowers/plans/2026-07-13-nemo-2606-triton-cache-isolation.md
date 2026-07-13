# NeMo 26.06 Triton Cache Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make multinode NeMo-RL CuTeDSL runs use a job- and restart-scoped node-local Triton cache, preserve bounded diagnostics on driver failure, and repair the kernel-attribution false negative found in job 2369786.

**Architecture:** Keep the benchmark result and checkpoint roots on shared storage, but derive both worker virtual environments and Triton cache from one `/tmp/${USER}/nemo2606-factorial/${RUN_ID}` root in existing-Ray mode. Add a standard-library-only cache scanner and an opt-in `ray.sub` failure hook that executes inside each still-running Ray container before `ENDED` cleanup. Keep timing behavior unchanged; the reliability state is represented only by a sanitized cache-scope enum and bounded diagnostic artifacts.

**Tech Stack:** Bash, SLURM/Pyxis, Ray, Python 3.13 standard library, pytest, Ruff, Nsight Systems kernel-stat text.

## Global Constraints

- The source baseline is commit `9c8962a5b`; the first cache-scope test must fail there.
- Existing-Ray multinode mode must use `/tmp/${USER}/nemo2606-factorial/${RUN_ID}/triton_cache` and scope `job_node_local`.
- Non-existing-Ray mode must retain `${CONTAINER_RUNTIME_DIR}/triton_cache` and scope `run_local_container`.
- CuTeDSL ON and OFF arms must receive the same cache path and scope.
- Diagnostics run only after a nonzero driver exit, finish or time out within 60 seconds, scan at most 256 candidates and 1 MiB, and never retain raw bytes, absolute paths, hostnames, IPs, or credentials.
- Do not change the pinned container, Triton, Transformer Engine, model workload, warmup count, measured count, or CuTeDSL selector.
- Do not add a retry, delete cache entries during training, or fall back to rank-local cache in this change.
- GPU jobs are submitted only after local tests, review, signed commit, push, `git pull`, and `sbatch --test-only` pass.
- Every new production Python or shell file starts with the repository's 2026 NVIDIA Apache-2.0 header; files under `tests/` are exempt.

---

## File Map

- Modify `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/run_cutedsl_matrix.sbatch`: select and export cache scope, write scope to manifests, and pass the failure diagnostic command to `ray.sub`.
- Create `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/collect_triton_cache_diagnostics.py`: bounded per-node cache scanner.
- Modify `ray.sub`: run an optional failure command in the head and worker containers before cleanup.
- Modify `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/render_cutedsl_report.py`: expose sanitized cache scope and bounded incident counts.
- Modify `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/README.md`: document scope, failure artifacts, and fallback gate.
- Modify `tests/test_nemo2606_multinode_factorial_harness.py`: shell contract and failure-hook tests.
- Create `tests/test_cutedsl_triton_cache_diagnostics.py`: scanner unit tests.
- Modify `tests/test_cutedsl_report.py`: public sanitization and attribution-regression tests.

### Task 1: Node-local cache scope

**Files:**
- Modify: `tests/test_nemo2606_multinode_factorial_harness.py`
- Modify: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/run_cutedsl_matrix.sbatch`

**Interfaces:**
- Consumes: `EXISTING_RAY`, `USER`, `RUN_ID`, and `CONTAINER_RUNTIME_DIR` already defined by the matrix payload.
- Produces: exported `TRITON_CACHE_DIR: str`, exported `NEMO2606_TRITON_CACHE_SCOPE: Literal["job_node_local", "run_local_container"]`, and manifest field `triton_cache_scope: str`.

- [ ] **Step 1: Write the failing shell-contract tests**

```python
def test_existing_ray_uses_job_scoped_node_local_triton_cache() -> None:
    source = MATRIX_PAYLOAD.read_text()
    assert 'NODE_LOCAL_RUNTIME_ROOT="/tmp/${USER}/nemo2606-factorial/${RUN_ID}"' in source
    assert 'NODE_LOCAL_WORKER_VENV_ROOT="${NODE_LOCAL_RUNTIME_ROOT}/worker_venvs"' in source
    assert 'TRITON_CACHE_DIR="${NODE_LOCAL_RUNTIME_ROOT}/triton_cache"' in source
    assert 'NEMO2606_TRITON_CACHE_SCOPE="job_node_local"' in source
    assert '"triton_cache_scope": os.environ["NEMO2606_TRITON_CACHE_SCOPE"]' in source


def test_non_existing_ray_retains_run_local_container_cache() -> None:
    source = MATRIX_PAYLOAD.read_text()
    assert 'TRITON_CACHE_DIR="${CONTAINER_RUNTIME_DIR}/triton_cache"' in source
    assert 'NEMO2606_TRITON_CACHE_SCOPE="run_local_container"' in source


def test_existing_ray_triton_cache_is_not_under_shared_roots() -> None:
    source = MATRIX_PAYLOAD.read_text()
    existing_ray = source.split('if [[ "${EXISTING_RAY}" == "1" ]]', 1)[1].split("else", 1)[0]
    assert 'TRITON_CACHE_DIR="${NODE_LOCAL_RUNTIME_ROOT}/triton_cache"' in existing_ray
    assert "CONTAINER_RUNTIME_DIR" not in existing_ray.split("TRITON_CACHE_DIR=", 1)[1].splitlines()[0]
    assert "RESULT_DIR" not in existing_ray.split("TRITON_CACHE_DIR=", 1)[1].splitlines()[0]
    assert "MEGATRON_CHECKPOINT_ROOT" not in existing_ray.split("TRITON_CACHE_DIR=", 1)[1].splitlines()[0]
```

- [ ] **Step 2: Prove the tests are RED on the baseline**

Run: `uv run pytest -q tests/test_nemo2606_multinode_factorial_harness.py -k 'node_local_triton or run_local_container_cache'`

Expected: FAIL because the existing-Ray path still exports `${CONTAINER_RUNTIME_DIR}/triton_cache` and does not define `NEMO2606_TRITON_CACHE_SCOPE`.

- [ ] **Step 3: Implement one canonical runtime root and sanitized scope**

Replace the current worker-venv/cache block with:

```bash
export CUDA_HOME="/usr/local/cuda"
export NVTE_CUDA_ARCHS="100"
if [[ "${EXISTING_RAY}" == "1" ]]; then
    NODE_LOCAL_RUNTIME_ROOT="/tmp/${USER}/nemo2606-factorial/${RUN_ID}"
    NODE_LOCAL_WORKER_VENV_ROOT="${NODE_LOCAL_RUNTIME_ROOT}/worker_venvs"
    TRITON_CACHE_DIR="${NODE_LOCAL_RUNTIME_ROOT}/triton_cache"
    NEMO2606_TRITON_CACHE_SCOPE="job_node_local"
else
    NODE_LOCAL_RUNTIME_ROOT="${CONTAINER_RUNTIME_DIR}"
    NODE_LOCAL_WORKER_VENV_ROOT="${CONTAINER_RUNTIME_DIR}/worker_venvs"
    TRITON_CACHE_DIR="${CONTAINER_RUNTIME_DIR}/triton_cache"
    NEMO2606_TRITON_CACHE_SCOPE="run_local_container"
fi
readonly NODE_LOCAL_RUNTIME_ROOT NODE_LOCAL_WORKER_VENV_ROOT
export NEMO_RL_VENV_DIR="${NODE_LOCAL_WORKER_VENV_ROOT}"
export TRITON_CACHE_DIR NEMO2606_TRITON_CACHE_SCOPE
mkdir -p "${TRITON_CACHE_DIR}"
```

Add to `benchmark_manifest.json` construction:

```python
"triton_cache_scope": os.environ["NEMO2606_TRITON_CACHE_SCOPE"],
```

Add `NEMO2606_TRITON_CACHE_SCOPE` to the fixed config evidence for both arms and assert the value is identical while producing `config_equivalence.json`.

- [ ] **Step 4: Run focused and syntax tests**

Run: `uv run pytest -q tests/test_nemo2606_multinode_factorial_harness.py -k 'triton or cache_scope' && bash -n experiments/cutedsl_qwen3_30ba3b_oci_1n4g/run_cutedsl_matrix.sbatch`

Expected: PASS and no Bash syntax output.

- [ ] **Step 5: Commit the isolated cache-scope change**

```bash
git add tests/test_nemo2606_multinode_factorial_harness.py experiments/cutedsl_qwen3_30ba3b_oci_1n4g/run_cutedsl_matrix.sbatch
git commit -s -m "fix: isolate multinode Triton caches per node"
```

### Task 2: Bounded sanitized cache diagnostics

**Files:**
- Create: `tests/test_cutedsl_triton_cache_diagnostics.py`
- Create: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/collect_triton_cache_diagnostics.py`

**Interfaces:**
- Produces: `DiagnosticLimits(max_files: int = 256, max_total_bytes: int = 1_048_576)`.
- Produces: `collect_cache_diagnostics(root: Path, node_index: int, limits: DiagnosticLimits) -> dict[str, Any]`.
- Produces: `merge_cache_diagnostics(summary_dir: Path, expected_nodes: int) -> dict[str, Any]`.
- CLI: direct unit-test mode uses `collect_triton_cache_diagnostics.py --cache-root PATH --output-dir PATH`; cluster mode uses `--from-slurm-env`, derives `run_id` from `SLURM_JOB_ID` plus `SLURM_RESTART_COUNT`, derives the cache as `Path("/tmp") / USER / "nemo2606-factorial" / run_id / "triton_cache"`, and derives output from `CUTEDSL_BENCHMARK_RESULT_ROOT`. Both modes read the nonnegative integer `FAILURE_DIAGNOSTIC_NODE_INDEX` supplied by `ray.sub` and create `f"node-{node_index}.json"` themselves.

- [ ] **Step 1: Write RED tests for valid, malformed, empty, symlink, and bounded input**

```python
def test_collect_cache_diagnostics_is_bounded_and_sanitized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_diagnostic_module()
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _: "3.6.0")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "valid.json").write_text('{"ok": true}')
    (cache / "empty.json").write_bytes(b"")
    (cache / "broken.json").write_text("{")
    outside = tmp_path / "secret.json"
    outside.write_text('{"token": "must-not-leak"}')
    (cache / "link.json").symlink_to(outside)

    result = module.collect_cache_diagnostics(
        cache,
        node_index=3,
        limits=module.DiagnosticLimits(max_files=2, max_total_bytes=32),
    )

    assert result["schema_version"] == 1
    assert result["node_index"] == 3
    assert result["candidate_count"] == 3
    assert result["scanned_count"] == 2
    assert result["rejected_symlink_count"] == 1
    assert result["truncated"] is True
    assert result["cache_scope"] == "job_node_local"
    assert result["job_id"] == "synthetic"
    serialized = json.dumps(result)
    assert str(tmp_path) not in serialized
    assert "must-not-leak" not in serialized
    for record in result["files"]:
        assert set(record) == {
            "relative_name_sha256",
            "file_type",
            "size",
            "inode",
            "mtime_ns",
            "json_valid",
            "prefix_sha256",
            "bytes_read",
        }


def test_merge_rejects_duplicate_nonfinite_and_symlinked_summaries(tmp_path: Path) -> None:
    module = load_diagnostic_module()
    summaries = tmp_path / "summaries"
    summaries.mkdir()
    write_node_summary(summaries / "node-0.json", node_index=0)
    write_node_summary(summaries / "node-1.json", node_index=0)
    with pytest.raises(ValueError, match="duplicate node_index"):
        module.merge_cache_diagnostics(summaries, expected_nodes=2)
    write_node_summary(summaries / "node-1.json", node_index=1, size=math.inf)
    with pytest.raises(ValueError, match="finite integer"):
        module.merge_cache_diagnostics(summaries, expected_nodes=2)
    (summaries / "node-1.json").unlink()
    outside = tmp_path / "outside.json"
    write_node_summary(outside, node_index=1)
    (summaries / "node-1.json").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        module.merge_cache_diagnostics(summaries, expected_nodes=2)
```

- [ ] **Step 2: Confirm the module import is RED**

Run: `uv run pytest -q tests/test_cutedsl_triton_cache_diagnostics.py`

Expected: FAIL because `collect_triton_cache_diagnostics.py` does not exist.

- [ ] **Step 3: Implement the scanner exactly at the approved bounds**

```python
@dataclass(frozen=True)
class DiagnosticLimits:
    max_files: int = 256
    max_total_bytes: int = 1_048_576


def _is_candidate(path: Path) -> bool:
    return path.suffix == ".json" or path.name.startswith("__grp__")


def collect_cache_diagnostics(
    root: Path,
    node_index: int,
    limits: DiagnosticLimits,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    candidates: list[Path] = []
    rejected_symlink_count = 0
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if not _is_candidate(path):
            continue
        if path.is_symlink():
            rejected_symlink_count += 1
            continue
        if path.is_file() and path.resolve().is_relative_to(root):
            candidates.append(path)

    files: list[dict[str, int | str | bool]] = []
    total_bytes_read = 0
    for path in candidates[: limits.max_files]:
        remaining = limits.max_total_bytes - total_bytes_read
        if remaining <= 0:
            break
        payload = path.read_bytes()[:remaining]
        total_bytes_read += len(payload)
        stat = path.stat(follow_symlinks=False)
        try:
            json.loads(payload)
            json_valid = True
        except (UnicodeDecodeError, json.JSONDecodeError):
            json_valid = False
        relative = path.relative_to(root).as_posix().encode()
        files.append(
            {
                "relative_name_sha256": hashlib.sha256(relative).hexdigest(),
                "file_type": "regular",
                "size": stat.st_size,
                "inode": stat.st_ino,
                "mtime_ns": stat.st_mtime_ns,
                "json_valid": json_valid,
                "prefix_sha256": hashlib.sha256(payload).hexdigest(),
                "bytes_read": len(payload),
            }
        )
    return {
        "schema_version": 1,
        "node_index": node_index,
        "job_id": os.environ.get("SLURM_JOB_ID", "synthetic"),
        "restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
        "slurm_procid": int(os.environ.get("SLURM_PROCID", str(node_index))),
        "cache_scope": os.environ.get(
            "NEMO2606_TRITON_CACHE_SCOPE", "job_node_local"
        ),
        "triton_version": importlib.metadata.version("triton"),
        "candidate_count": len(candidates),
        "scanned_count": len(files),
        "rejected_symlink_count": rejected_symlink_count,
        "total_bytes_read": total_bytes_read,
        "truncated": len(files) < len(candidates),
        "files": files,
    }
```

Validate `node_index >= 0`, `1 <= max_files <= 256`, and `1 <= max_total_bytes <= 1_048_576`. Tests monkeypatch `importlib.metadata.version` to a finite string when Triton is unavailable locally. Write the CLI output through `NamedTemporaryFile(dir=output.parent, delete=False)` followed by `os.replace`.

Implement `merge_cache_diagnostics` with these exact gates:

```python
def merge_cache_diagnostics(summary_dir: Path, expected_nodes: int) -> dict[str, Any]:
    if expected_nodes < 1:
        raise ValueError("expected_nodes must be positive")
    root = summary_dir.resolve(strict=True)
    nodes: dict[int, dict[str, Any]] = {}
    for path in sorted(summary_dir.glob("node-*.json")):
        if path.is_symlink():
            raise ValueError("node summary must not be a symlink")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError("node summary must be a contained regular file")
        value = json.loads(resolved.read_text())
        node_index = _finite_nonnegative_integer(value.get("node_index"), "node_index")
        if node_index in nodes:
            raise ValueError(f"duplicate node_index: {node_index}")
        _validate_summary_schema(value)
        nodes[node_index] = value
    return {
        "schema_version": 1,
        "expected_nodes": expected_nodes,
        "observed_nodes": sorted(nodes),
        "missing_nodes": sorted(set(range(expected_nodes)) - set(nodes)),
        "timed_out": len(nodes) != expected_nodes,
        "truncated": any(value["truncated"] for value in nodes.values()),
        "nodes": [nodes[index] for index in sorted(nodes)],
    }
```

`_validate_summary_schema` must allow only the documented keys, at most 256 file records per node, at most 1,048,576 summed `bytes_read`, SHA-256 fields matching `[0-9a-f]{64}`, `file_type == "regular"`, JSON booleans for `json_valid/truncated`, finite nonnegative integers for every numeric value, and cache scopes from the two-value enum.

Implement its numeric primitive as:

```python
def _finite_nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a finite nonnegative integer")
    return value


def _validate_summary_schema(value: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "node_index",
        "job_id",
        "restart_count",
        "slurm_procid",
        "cache_scope",
        "triton_version",
        "candidate_count",
        "scanned_count",
        "rejected_symlink_count",
        "total_bytes_read",
        "truncated",
        "files",
    }
    if set(value) != required or value["schema_version"] != 1:
        raise ValueError("invalid node-summary schema")
    files = value["files"]
    if not isinstance(files, list) or len(files) > 256:
        raise ValueError("node-summary file listing exceeds 256")
    for label in (
        "node_index",
        "restart_count",
        "slurm_procid",
        "candidate_count",
        "scanned_count",
        "rejected_symlink_count",
        "total_bytes_read",
    ):
        _finite_nonnegative_integer(value[label], label)
    if value["cache_scope"] not in {"job_node_local", "run_local_container"}:
        raise ValueError("invalid cache scope")
    if re.fullmatch(r"(?:[0-9]+|synthetic)", value.get("job_id", "")) is None:
        raise ValueError("invalid job identity")
    if re.fullmatch(r"[A-Za-z0-9_.+-]{1,64}", value.get("triton_version", "")) is None:
        raise ValueError("invalid Triton version")
    if not isinstance(value["truncated"], bool):
        raise ValueError("truncated must be boolean")
    total = 0
    for record in files:
        if record.get("file_type") != "regular":
            raise ValueError("diagnostic record must describe a regular file")
        for digest in ("relative_name_sha256", "prefix_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", record.get(digest, "")) is None:
                raise ValueError(f"invalid {digest}")
        for label in ("size", "inode", "mtime_ns", "bytes_read"):
            _finite_nonnegative_integer(record.get(label), label)
        if not isinstance(record.get("json_valid"), bool):
            raise ValueError("json_valid must be boolean")
        total += record["bytes_read"]
    if total > 1_048_576 or total != value["total_bytes_read"]:
        raise ValueError("diagnostic byte total is invalid")
```

- [ ] **Step 4: Run unit tests and Ruff**

Run: `uv run pytest -q tests/test_cutedsl_triton_cache_diagnostics.py && uv run ruff check experiments/cutedsl_qwen3_30ba3b_oci_1n4g/collect_triton_cache_diagnostics.py tests/test_cutedsl_triton_cache_diagnostics.py`

Expected: PASS with no Ruff findings.

- [ ] **Step 5: Commit the diagnostic helper**

```bash
git add experiments/cutedsl_qwen3_30ba3b_oci_1n4g/collect_triton_cache_diagnostics.py tests/test_cutedsl_triton_cache_diagnostics.py
git commit -s -m "feat: collect bounded Triton cache diagnostics"
```

### Task 3: Opt-in per-node failure hook in `ray.sub`

**Files:**
- Modify: `tests/test_nemo2606_multinode_factorial_harness.py`
- Modify: `ray.sub`
- Modify: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/submit_nemo2606_2n4g_factorial.sh`
- Modify: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/run_cutedsl_matrix.sbatch`

**Interfaces:**
- Consumes: optional `FAILURE_COMMAND: str`; empty means current behavior.
- Produces: shared signal `DRIVER_FAILED`, indexed signals `f"FAILURE_DIAGNOSTIC_DONE_{node_index}"`, and per-node files `f"triton_cache_diagnostics/node-{node_index}.json"`.
- The command receives `FAILURE_DIAGNOSTIC_NODE_INDEX`, `TRITON_CACHE_DIR`, and the benchmark result directory.

- [ ] **Step 1: Write failing source-contract tests**

```python
def test_ray_sub_runs_failure_hook_before_ended_cleanup() -> None:
    source = (PROJECT_ROOT / "ray.sub").read_text()
    assert 'FAILURE_COMMAND_FILE=""' in source
    assert 'touch "$LOG_DIR/DRIVER_FAILED"' in source
    assert 'FAILURE_DIAGNOSTIC_DONE_0' in source
    assert 'FAILURE_DIAGNOSTIC_DEADLINE=$((SECONDS + 60))' in source
    assert source.index('touch "$LOG_DIR/DRIVER_FAILED"') < source.index('touch "$LOG_DIR/ENDED"')


def test_submitter_wires_sanitized_triton_failure_command() -> None:
    source = SUBMITTER.read_text()
    assert "collect_triton_cache_diagnostics.py" in source
    assert "--from-slurm-env" in source
    assert "CUTEDSL_BENCHMARK_RESULT_ROOT" in source
```

- [ ] **Step 2: Confirm RED behavior**

Run: `uv run pytest -q tests/test_nemo2606_multinode_factorial_harness.py -k 'failure_hook or failure_command'`

Expected: FAIL because `ray.sub` does not accept or coordinate `FAILURE_COMMAND`.

- [ ] **Step 3: Write `FAILURE_COMMAND_FILE` beside the driver command**

```bash
FAILURE_COMMAND_FILE=""
if [[ -n "${FAILURE_COMMAND:-}" ]]; then
  FAILURE_COMMAND_FILE="$LOG_DIR/failure_command.sh"
  printf '%s' "$FAILURE_COMMAND" > "$FAILURE_COMMAND_FILE"
  chmod +x "$FAILURE_COMMAND_FILE"
fi
```

- [ ] **Step 4: Add one-shot diagnostics in the existing head and worker containers**

After a nonzero head driver exit, run:

```bash
if [[ "\$exit_code" -ne 0 ]] && [[ -n "$FAILURE_COMMAND_FILE" ]]; then
  touch "$LOG_DIR/DRIVER_FAILED"
  export FAILURE_DIAGNOSTIC_NODE_INDEX=0
  bash "$FAILURE_COMMAND_FILE" || true
  touch "$LOG_DIR/FAILURE_DIAGNOSTIC_DONE_0"
  FAILURE_DIAGNOSTIC_DEADLINE=\$((SECONDS + 60))
  while (( SECONDS <= FAILURE_DIAGNOSTIC_DEADLINE )); do
    done_count=\$(find "$LOG_DIR" -maxdepth 1 -name 'FAILURE_DIAGNOSTIC_DONE_*' -type f | wc -l)
    [[ "\$done_count" -ge "$SLURM_JOB_NUM_NODES" ]] && break
    sleep 1
  done
  export FAILURE_DIAGNOSTIC_MERGE=1
  bash "$FAILURE_COMMAND_FILE" || true
  unset FAILURE_DIAGNOSTIC_MERGE
fi
```

Add a worker sidecar that polls `DRIVER_FAILED` once per second, exports `FAILURE_DIAGNOSTIC_NODE_INDEX=$((SLURM_PROCID + 1))`, runs the same file once, and touches its indexed done signal. It must exit without invoking `exit-dramatically`; normal `ENDED` cleanup remains authoritative. On the second head invocation, `--from-slurm-env` sees `FAILURE_DIAGNOSTIC_MERGE=1`, calls `merge_cache_diagnostics(output_dir, SLURM_JOB_NUM_NODES)`, and atomically writes `summary.json`; missing nodes remain explicit after the 60-second deadline.

- [ ] **Step 5: Wire the cache scanner from the submitter before Ray starts**

The matrix driver cannot mutate worker-container environments after Ray starts, and its exit trap removes the shared runtime venv. Therefore the submitter must pass a standard-library `python3` command and result root into `ray.sub` before submission:

```bash
"CUTEDSL_BENCHMARK_RESULT_ROOT=${RESULT_ROOT}" \
"FAILURE_COMMAND=exec python3 ${EXPERIMENT_DIR}/collect_triton_cache_diagnostics.py --from-slurm-env"
```

The helper's `--from-slurm-env` path computes:

```python
restart = os.environ.get("SLURM_RESTART_COUNT")
run_id = os.environ["SLURM_JOB_ID"] + (f"-r{restart}" if restart else "")
cache_root = Path("/tmp") / os.environ["USER"] / "nemo2606-factorial" / run_id / "triton_cache"
output_dir = Path(os.environ["CUTEDSL_BENCHMARK_RESULT_ROOT"]) / run_id / "triton_cache_diagnostics"
```

The submitter must include both variables in its clean `env -0` export allowlist. The successful path must not create `DRIVER_FAILED` or a diagnostics directory.

- [ ] **Step 6: Run shell and focused tests**

Run: `bash -n ray.sub && bash -n experiments/cutedsl_qwen3_30ba3b_oci_1n4g/run_cutedsl_matrix.sbatch && uv run pytest -q tests/test_nemo2606_multinode_factorial_harness.py -k 'failure or triton'`

Expected: PASS; generated head and worker scripts also pass their existing runtime `bash -n` checks.

- [ ] **Step 7: Commit the failure hook**

```bash
git add ray.sub experiments/cutedsl_qwen3_30ba3b_oci_1n4g/submit_nemo2606_2n4g_factorial.sh experiments/cutedsl_qwen3_30ba3b_oci_1n4g/run_cutedsl_matrix.sbatch tests/test_nemo2606_multinode_factorial_harness.py
git commit -s -m "feat: preserve multinode cache evidence on failure"
```

### Task 4: Repair kernel attribution and publish the incident correctly

**Files:**
- Modify: `tests/test_nemo2606_multinode_factorial_harness.py`
- Modify: `tests/test_cutedsl_report.py`
- Modify: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/run_cutedsl_matrix.sbatch`
- Modify: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/render_cutedsl_report.py`
- Modify: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/README.md`

**Interfaces:**
- Produces: boundary-safe signature expressions matching suffixes such as `_object_at_0x1`.
- Produces: attribution fields `fused_glu_match_count`, `fused_dglu_match_count`, `fused_quant_match_count`, `fused_grouped_gemm_match_count`, and `baseline_expert_gemm_match_count`.
- Report statement: job 2369786's profile failure was a matcher false negative, not missing GPU trace.

- [ ] **Step 1: Add a regression fixture using the actual kernel-name shape**

```python
def test_kernel_matchers_accept_cudnn_object_suffix_and_reject_off_arm() -> None:
    on = """
    kernel_cutlass_kernel_cudnngrouped_gemm_BlockScaledMoEGroupedGemmQuantKernel_object_at_0x1
    kernel_cutlass_kernel_cudnngrouped_gemm_BlockScaledMoEGroupedGemmGluBiasKernel_object_at_0x2
    kernel_cutlass_kernel_cudnngrouped_gemm_BlockScaledMoEGroupedGemmDgluDbiasKernel_object_at_0x3
    """
    off = "nvjet_sm100_128x128"
    counts = run_attribution_fixture(on=on, off=off)
    assert counts["on"]["fused_glu_match_count"] == 1
    assert counts["on"]["fused_dglu_match_count"] == 1
    assert counts["on"]["fused_quant_match_count"] == 1
    assert counts["off"]["fused_glu_match_count"] == 0
    assert counts["off"]["baseline_expert_gemm_match_count"] == 1
```

- [ ] **Step 2: Run the regression and observe the word-boundary failure**

Run: `uv run pytest -q tests/test_nemo2606_multinode_factorial_harness.py -k kernel_matchers_accept`

Expected: FAIL because `_` is a word character and the current trailing `Kernel\b` cannot match `Kernel_object_at_0x1`.

- [ ] **Step 3: Replace ambiguous boundaries with exact class/suffix patterns**

```python
FUSED_GLU_SIGNATURES = {
    "cudnn_frontend_fused_glu": r"BlockScaledMoEGroupedGemmGluBiasKernel(?:_|\b)",
}
FUSED_DGLU_SIGNATURES = {
    "cudnn_frontend_fused_dglu": r"BlockScaledMoEGroupedGemmDgluDbiasKernel(?:_|\b)",
}
FUSED_QUANT_SIGNATURES = {
    "cudnn_frontend_fused_quant": r"BlockScaledMoEGroupedGemmQuantKernel(?:_|\b)",
}
FUSED_GROUPED_GEMM_SIGNATURES = {
    "cudnn_frontend_grouped_gemm": r"BlockScaledMoEGroupedGemm\w*Kernel(?:_|\b)",
}
BASELINE_EXPERT_GEMM_SIGNATURES = {
    "nvjet_sm100": r"(?:^|[^A-Za-z0-9])nvjet_sm100_[A-Za-z0-9_]*",
}
```

Require CuTeDSL ON to have fused GLU, dGLU, quant, and grouped-GEMM counts greater than zero; require all fused counts to be zero OFF; require OFF `nvjet_sm100` count greater than zero together with `moe_grouped_gemm=true` and `use_transformer_engine_op_fuser=true`. Do not label the bare `nvjet_sm100` name sufficient without those config predicates.

- [ ] **Step 4: Record the observed evidence and cache incident without private paths**

Add a tracked report incident containing only:

```json
{
  "job_id": "2369786",
  "classification": "kernel_matcher_false_negative",
  "on_kernel_stat_rows": 4664,
  "off_kernel_stat_rows": 4765,
  "on_fused_glu_instances": 241152,
  "on_fused_dglu_instances": 161280,
  "on_fused_quant_instances": 402432,
  "off_fused_instances": 0,
  "performance_claim_impact": "recollect_after_matcher_fix"
}
```

Document job 2369788 separately as `triton_group_metadata_json_decode_error`, with cause bounded to a shared Lustre cache boundary rather than an unproven writer race.

- [ ] **Step 5: Run report and attribution tests**

Run: `uv run pytest -q tests/test_nemo2606_multinode_factorial_harness.py tests/test_cutedsl_report.py -k 'kernel or attribution or incident or sanit'`

Expected: PASS, and serialized report fixtures contain no absolute path, hostname, IP, token, or raw cache bytes.

- [ ] **Step 6: Commit the RCA and matcher repair**

```bash
git add experiments/cutedsl_qwen3_30ba3b_oci_1n4g/run_cutedsl_matrix.sbatch experiments/cutedsl_qwen3_30ba3b_oci_1n4g/render_cutedsl_report.py experiments/cutedsl_qwen3_30ba3b_oci_1n4g/README.md tests/test_nemo2606_multinode_factorial_harness.py tests/test_cutedsl_report.py
git commit -s -m "fix: recognize CuTeDSL fused kernel names"
```

### Task 5: Local verification, review, and Pre-Tyche reliability gate

**Files:**
- Modify only if a test or review exposes a defect in files already listed above.

**Interfaces:**
- Produces: one reviewed, pushed source SHA and one three-update functional result using `job_node_local` on every node.

- [ ] **Step 1: Run the full local verification set**

```bash
uv run pytest -q \
  tests/test_cutedsl_triton_cache_diagnostics.py \
  tests/test_nemo2606_multinode_factorial_harness.py \
  tests/test_cutedsl_hf_cache.py \
  tests/test_cutedsl_replicate_collector.py \
  tests/test_cutedsl_report.py \
  tests/test_cutedsl_policy_recipe.py
uv run ruff check \
  experiments/cutedsl_qwen3_30ba3b_oci_1n4g \
  tests/test_cutedsl_triton_cache_diagnostics.py \
  tests/test_nemo2606_multinode_factorial_harness.py
bash -n ray.sub
bash -n experiments/cutedsl_qwen3_30ba3b_oci_1n4g/run_cutedsl_matrix.sbatch
bash -n experiments/cutedsl_qwen3_30ba3b_oci_1n4g/submit_nemo2606_4n4g_performance.sh
```

Expected: all tests pass, Ruff is clean, and Bash emits no syntax errors.

- [ ] **Step 2: Request independent spec and code review**

Use `superpowers:requesting-code-review` with both the cache design and this plan. Resolve every correctness or safety finding, rerun Step 1, and record the reviewer result in the branch notes.

- [ ] **Step 3: Push only the feature branch**

```bash
git status --short
git push fork sna/nemo-2606-cutedsl-a2a-factorial-20260712
```

Expected: clean status and the fork branch advances; no remote default branch changes.

- [ ] **Step 4: Fast-forward and preflight Pre-Tyche**

On `login-ptyche`, fast-forward the existing feature worktree, run `git submodule status --recursive`, verify the pinned image SHA256, then run:

```bash
NEMO2606_FUNCTIONAL_GATE=1 \
NEMO2606_FACTORIAL_CONTEXTS=g0a0 \
NEMO2606_FACTORIAL_REPLICATES=3 \
NEMO2606_FACTORIAL_WARMUP_UPDATES=5 \
NEMO2606_FACTORIAL_MEASURED_UPDATES=20 \
experiments/cutedsl_qwen3_30ba3b_oci_1n4g/submit_nemo2606_4n4g_performance.sh --test-only
```

Expected: the exact four-node request is schedulable and no job is submitted.

- [ ] **Step 5: Submit and monitor the three-update functional gate**

Submit the same command without `--test-only`. Monitor Slurm state, driver log, Ray head, and every worker log for five minutes. Acceptance requires successful generation, refit, reference/policy logprob, PolicyTraining, offload, and a second mature boundary; every manifest must report `triton_cache_scope=job_node_local`; no successful run may contain `triton_cache_diagnostics`.

- [ ] **Step 6: Exercise the diagnostic path with a controlled CPU-only failure**

Use the harness test mode to make the driver exit nonzero before model initialization. Acceptance requires one sanitized summary per node, no GPU timing samples, no raw cache bytes, and aggregate completion within 60 seconds. Do not inject failure into an accepted performance job.

- [ ] **Step 7: Record the gate result**

Update the experiment README and HTML incident timeline with source SHA, image SHA, public job IDs, cache scopes, pass/fail boundary, and reproducible commands; exclude cluster paths and host identities. Commit and push that evidence before starting the 235B implementation plan.
