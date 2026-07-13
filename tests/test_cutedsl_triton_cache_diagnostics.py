import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_MODULE = (
    PROJECT_ROOT
    / "experiments/cutedsl_qwen3_30ba3b_oci_1n4g"
    / "collect_triton_cache_diagnostics.py"
)


def load_diagnostic_module() -> ModuleType:
    """Load the standalone diagnostic script as a testable module."""
    spec = importlib.util.spec_from_file_location(
        "collect_triton_cache_diagnostics", DIAGNOSTIC_MODULE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load diagnostic module: {DIAGNOSTIC_MODULE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_node_summary(
    path: Path,
    *,
    node_index: int,
    size: int | float = 0,
) -> None:
    """Write one complete node-summary fixture."""
    record = {
        "relative_name_sha256": hashlib.sha256(b"cache-entry").hexdigest(),
        "file_type": "regular",
        "size": size,
        "inode": 1,
        "mtime_ns": 2,
        "json_valid": True,
        "prefix_sha256": hashlib.sha256(b"").hexdigest(),
        "bytes_read": 0,
    }
    value = {
        "schema_version": 1,
        "node_index": node_index,
        "job_id": "synthetic",
        "restart_count": 0,
        "slurm_procid": node_index,
        "cache_scope": "job_node_local",
        "triton_version": "3.6.0",
        "candidate_count": 1,
        "scanned_count": 1,
        "rejected_symlink_count": 0,
        "total_bytes_read": 0,
        "truncated": False,
        "files": [record],
    }
    path.write_text(json.dumps(value))


def test_collect_cache_diagnostics_is_bounded_and_sanitized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_diagnostic_module()
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _: "3.6.0")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
    monkeypatch.delenv("SLURM_PROCID", raising=False)
    monkeypatch.delenv("NEMO2606_TRITON_CACHE_SCOPE", raising=False)
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
    assert "valid.json" not in serialized
    assert "empty.json" not in serialized
    assert "broken.json" not in serialized
    assert "link.json" not in serialized
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


def test_collect_cache_diagnostics_handles_an_empty_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_diagnostic_module()
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _: "3.6.0")
    cache = tmp_path / "cache"
    cache.mkdir()

    result = module.collect_cache_diagnostics(
        cache, node_index=0, limits=module.DiagnosticLimits()
    )

    assert result["candidate_count"] == 0
    assert result["scanned_count"] == 0
    assert result["total_bytes_read"] == 0
    assert result["truncated"] is False
    assert result["files"] == []


def test_collect_cache_diagnostics_classifies_json_and_marks_partial_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_diagnostic_module()
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _: "3.6.0")
    cache = tmp_path / "cache"
    cache.mkdir()
    payloads = {
        "a-valid.json": b'{"payload": "valid-content-marker"}',
        "b-malformed.json": b'{"payload": "malformed-content-marker"',
        "c-empty.json": b"",
        "d-oversized.json": b"must-not-leak-" + b"x" * 1_048_576,
    }
    for name, payload in payloads.items():
        (cache / name).write_bytes(payload)

    result = module.collect_cache_diagnostics(
        cache, node_index=0, limits=module.DiagnosticLimits()
    )

    records = {record["relative_name_sha256"]: record for record in result["files"]}
    valid = records[hashlib.sha256(b"a-valid.json").hexdigest()]
    malformed = records[hashlib.sha256(b"b-malformed.json").hexdigest()]
    empty = records[hashlib.sha256(b"c-empty.json").hexdigest()]
    oversized = records[hashlib.sha256(b"d-oversized.json").hexdigest()]
    assert valid["json_valid"] is True
    assert valid["bytes_read"] == len(payloads["a-valid.json"])
    assert malformed["json_valid"] is False
    assert malformed["bytes_read"] == len(payloads["b-malformed.json"])
    assert empty["json_valid"] is False
    assert empty["bytes_read"] == 0
    expected_oversized_bytes = 1_048_576 - sum(
        len(payloads[name])
        for name in ("a-valid.json", "b-malformed.json", "c-empty.json")
    )
    assert oversized["bytes_read"] == expected_oversized_bytes
    assert oversized["bytes_read"] < oversized["size"]
    assert result["total_bytes_read"] == 1_048_576
    assert result["total_bytes_read"] <= module.DiagnosticLimits().max_total_bytes
    assert result["truncated"] is True
    serialized = json.dumps(result)
    for marker in (
        "valid-content-marker",
        "malformed-content-marker",
        "must-not-leak",
    ):
        assert marker not in serialized
    for name in payloads:
        assert name not in serialized


@pytest.mark.parametrize(
    ("node_index", "max_files", "max_total_bytes", "message"),
    [
        (-1, 1, 1, "node_index must be a finite nonnegative integer"),
        (0, 0, 1, "max_files must be between 1 and 256"),
        (0, 257, 1, "max_files must be between 1 and 256"),
        (0, 1, 0, "max_total_bytes must be between 1 and 1048576"),
        (0, 1, 1_048_577, "max_total_bytes must be between 1 and 1048576"),
    ],
)
def test_collect_cache_diagnostics_rejects_invalid_limits(
    tmp_path: Path,
    node_index: int,
    max_files: int,
    max_total_bytes: int,
    message: str,
) -> None:
    module = load_diagnostic_module()
    cache = tmp_path / "cache"
    cache.mkdir()

    with pytest.raises(ValueError, match=message):
        module.collect_cache_diagnostics(
            cache,
            node_index=node_index,
            limits=module.DiagnosticLimits(
                max_files=max_files, max_total_bytes=max_total_bytes
            ),
        )


def test_merge_cache_diagnostics_reports_missing_nodes(tmp_path: Path) -> None:
    module = load_diagnostic_module()
    summaries = tmp_path / "summaries"
    summaries.mkdir()
    write_node_summary(summaries / "node-2.json", node_index=2)
    write_node_summary(summaries / "node-0.json", node_index=0)

    result = module.merge_cache_diagnostics(summaries, expected_nodes=4)

    assert result["schema_version"] == 1
    assert result["expected_nodes"] == 4
    assert result["observed_nodes"] == [0, 2]
    assert result["missing_nodes"] == [1, 3]
    assert result["timed_out"] is True
    assert result["truncated"] is False
    assert [node["node_index"] for node in result["nodes"]] == [0, 2]


def test_merge_rejects_node_indexes_outside_expected_range(tmp_path: Path) -> None:
    module = load_diagnostic_module()
    summaries = tmp_path / "summaries"
    summaries.mkdir()
    write_node_summary(summaries / "node-2.json", node_index=2)
    write_node_summary(summaries / "node-3.json", node_index=3)

    with pytest.raises(ValueError, match="node_index is outside expected range"):
        module.merge_cache_diagnostics(summaries, expected_nodes=2)


def test_merge_rejects_duplicate_nonfinite_and_symlinked_summaries(
    tmp_path: Path,
) -> None:
    module = load_diagnostic_module()
    summaries = tmp_path / "summaries"
    summaries.mkdir()
    write_node_summary(summaries / "node-0.json", node_index=0)
    write_node_summary(summaries / "node-1.json", node_index=0)
    with pytest.raises(ValueError, match="duplicate node_index"):
        module.merge_cache_diagnostics(summaries, expected_nodes=2)
    write_node_summary(summaries / "node-1.json", node_index=1, size=math.inf)
    with pytest.raises(ValueError, match="finite nonnegative integer"):
        module.merge_cache_diagnostics(summaries, expected_nodes=2)
    (summaries / "node-1.json").unlink()
    outside = tmp_path / "outside.json"
    write_node_summary(outside, node_index=1)
    (summaries / "node-1.json").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        module.merge_cache_diagnostics(summaries, expected_nodes=2)


def test_merge_rejects_undocumented_record_fields(tmp_path: Path) -> None:
    module = load_diagnostic_module()
    summaries = tmp_path / "summaries"
    summaries.mkdir()
    summary = summaries / "node-0.json"
    write_node_summary(summary, node_index=0)
    value = json.loads(summary.read_text())
    value["files"][0]["filename"] = "/tmp/secret.json"
    summary.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="invalid diagnostic record schema"):
        module.merge_cache_diagnostics(summaries, expected_nodes=1)


def test_direct_cli_writes_its_node_summary_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_diagnostic_module()
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _: "3.6.0")
    monkeypatch.setenv("FAILURE_DIAGNOSTIC_NODE_INDEX", "4")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "entry.json").write_text('{"ok": true}')
    output_dir = tmp_path / "diagnostics"

    exit_code = module.main(
        ["--cache-root", str(cache), "--output-dir", str(output_dir)]
    )

    assert exit_code == 0
    output = output_dir / "node-4.json"
    assert output.is_file()
    assert json.loads(output.read_text())["node_index"] == 4
    assert list(output_dir.iterdir()) == [output]


def test_slurm_cli_derives_job_local_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_diagnostic_module()
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _: "3.6.0")
    synthetic_user = f"cache-diagnostics-test-{os.getpid()}-{tmp_path.name}"
    runtime_root = Path("/tmp") / synthetic_user
    cache = runtime_root / "nemo2606-factorial" / "12345-2" / "triton_cache"
    cache.mkdir(parents=True)
    (cache / "entry.json").write_text('{"ok": true}')
    output_dir = tmp_path / "benchmark-results"
    monkeypatch.setenv("USER", synthetic_user)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "2")
    monkeypatch.setenv("SLURM_PROCID", "7")
    monkeypatch.setenv("FAILURE_DIAGNOSTIC_NODE_INDEX", "1")
    monkeypatch.setenv("CUTEDSL_BENCHMARK_RESULT_ROOT", str(output_dir))

    try:
        exit_code = module.main(["--from-slurm-env"])
    finally:
        shutil.rmtree(runtime_root)

    assert exit_code == 0
    value = json.loads((output_dir / "node-1.json").read_text())
    assert value["job_id"] == "12345"
    assert value["restart_count"] == 2
    assert value["slurm_procid"] == 7
