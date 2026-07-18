import io
import json
import ast
from pathlib import Path

import pytest

from nemo_rl.models.megatron.refit_offload_diagnostics import (
    HostMemorySnapshot,
    measure_refit_phase,
)


def test_measure_refit_phase_emits_rank_resource_deltas() -> None:
    snapshots = iter(
        (
            HostMemorySnapshot(
                rss_bytes=1000,
                major_faults=7,
                mem_available_bytes=9000,
                minor_faults=100,
            ),
            HostMemorySnapshot(
                rss_bytes=1600,
                major_faults=9,
                mem_available_bytes=8100,
                minor_faults=140,
            ),
        )
    )
    clock = iter((10.0, 12.25))
    output = io.StringIO()

    result = measure_refit_phase(
        lambda: "complete",
        phase="offload_before_refit.optimizer_d2h",
        rank=17,
        optimizer_cuda_bytes=4096,
        capture_snapshot=lambda: next(snapshots),
        monotonic=lambda: next(clock),
        hostname="node-a",
        stream=output,
    )

    assert result == "complete"
    prefix, serialized = output.getvalue().strip().split(" ", 1)
    assert prefix == "[NRL_REFIT_OFFLOAD]"
    assert json.loads(serialized) == {
        "elapsed_s": 2.25,
        "event": "refit_offload_phase",
        "hostname": "node-a",
        "major_faults_after": 9,
        "major_faults_before": 7,
        "major_faults_delta": 2,
        "mem_available_bytes_after": 8100,
        "mem_available_bytes_before": 9000,
        "mem_available_bytes_delta": -900,
        "minor_faults_after": 140,
        "minor_faults_before": 100,
        "minor_faults_delta": 40,
        "optimizer_cuda_bytes": 4096,
        "phase": "offload_before_refit.optimizer_d2h",
        "rank": 17,
        "rss_bytes_after": 1600,
        "rss_bytes_before": 1000,
        "rss_bytes_delta": 600,
        "schema_version": 1,
        "status": "ok",
    }


def test_measure_refit_phase_records_failure_before_reraising() -> None:
    snapshot = HostMemorySnapshot(
        rss_bytes=None,
        major_faults=0,
        mem_available_bytes=None,
    )
    output = io.StringIO()

    def fail() -> None:
        raise RuntimeError("injected failure")

    with pytest.raises(RuntimeError, match="injected failure"):
        measure_refit_phase(
            fail,
            phase="offload_before_refit.gradient_release",
            rank=3,
            optimizer_cuda_bytes=None,
            capture_snapshot=lambda: snapshot,
            monotonic=iter((1.0, 1.5)).__next__,
            hostname="node-b",
            stream=output,
        )

    payload = json.loads(output.getvalue().strip().split(" ", 1)[1])
    assert payload["status"] == "error"
    assert payload["error_type"] == "RuntimeError"
    assert payload["elapsed_s"] == 0.5


def test_measure_refit_phase_runs_operation_when_initial_snapshot_fails() -> None:
    calls = 0

    def fail_snapshot() -> HostMemorySnapshot:
        raise OSError("procfs unavailable")

    def operation() -> str:
        nonlocal calls
        calls += 1
        return "complete"

    assert (
        measure_refit_phase(
            operation,
            phase="offload_before_refit.optimizer_d2h",
            rank=1,
            optimizer_cuda_bytes=1024,
            capture_snapshot=fail_snapshot,
            stream=io.StringIO(),
        )
        == "complete"
    )
    assert calls == 1


def test_measure_refit_phase_preserves_operation_error_when_logging_fails() -> None:
    snapshot_calls = 0

    def fail_final_snapshot() -> HostMemorySnapshot:
        nonlocal snapshot_calls
        snapshot_calls += 1
        if snapshot_calls == 1:
            return HostMemorySnapshot(1, 0, 2)
        raise OSError("procfs disappeared")

    class BrokenStream(io.StringIO):
        def write(self, text: str) -> int:
            raise OSError("log destination closed")

    def fail_operation() -> None:
        raise RuntimeError("original refit failure")

    with pytest.raises(RuntimeError, match="original refit failure"):
        measure_refit_phase(
            fail_operation,
            phase="offload_before_refit.optimizer_d2h",
            rank=2,
            optimizer_cuda_bytes=2048,
            capture_snapshot=fail_final_snapshot,
            stream=BrokenStream(),
        )


def test_measure_refit_phase_uses_null_fault_deltas_when_snapshot_fails() -> None:
    snapshots = iter(
        (
            HostMemorySnapshot(1, 7, 2, 11),
            OSError("procfs disappeared"),
        )
    )
    output = io.StringIO()

    def capture_snapshot() -> HostMemorySnapshot:
        value = next(snapshots)
        if isinstance(value, Exception):
            raise value
        return value

    measure_refit_phase(
        lambda: None,
        phase="offload_before_refit.gc",
        rank=3,
        optimizer_cuda_bytes=None,
        capture_snapshot=capture_snapshot,
        stream=output,
    )

    payload = json.loads(output.getvalue().strip().split(" ", 1)[1])
    assert payload["major_faults_delta"] is None
    assert payload["minor_faults_delta"] is None


def test_megatron_worker_exposes_pageable_and_coalesced_pinned_paths() -> None:
    worker_path = (
        Path(__file__).parents[2]
        / "nemo_rl/models/policy/workers/megatron_policy_worker.py"
    )
    tree = ast.parse(worker_path.read_text(encoding="utf-8"))
    worker = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MegatronPolicyWorkerImpl"
    )
    methods = {node.name for node in worker.body if isinstance(node, ast.FunctionDef)}

    assert {
        "_optimizer_to_cpu",
        "_optimizer_to_cuda",
        "_coalesced_optimizer_to_cpu",
        "_coalesced_optimizer_to_cuda",
        "_get_or_alloc_pinned_buf",
    }.issubset(methods)
