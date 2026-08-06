#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Run one allowlisted distributed pytest row and publish attested results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")
FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
SCHEDULER_JOB_ID = re.compile(r"^[1-9][0-9]*$")
RUN_IDENTITY = re.compile(r"^slurm-[1-9][0-9]*-[0-9]+-[0-9a-f]{64}$")
ROW_ID = re.compile(r"^[a-z][a-z0-9_]*$")
PYTEST_NODE = re.compile(r"^tests/[A-Za-z0-9_./-]+\.py::[A-Za-z0-9_\[\].-]+$")
ALLOWED_ALLOCATIONS = frozenset(
    ((1, 8, 8), (2, 4, 8), (4, 4, 16), (8, 4, 32), (16, 4, 64))
)
CAPABILITY_DEVICE_FIELDS = frozenset(
    ("global_rank", "node_rank", "local_rank", "cuda_device_index")
)
PRIMARY_CAPABILITY_NODE = (
    "tests/unit_tests/transformer/test_cuda_graphs.py::"
    "test_te_make_graphed_callables_supports_eval_no_grad"
)
REUSE_CAPABILITY_NODE = (
    "tests/unit_tests/transformer/test_cuda_graphs.py::"
    "test_te_eval_graph_input_output_buffer_reuse_capability"
)


@dataclass(frozen=True)
class MatrixRow:
    """One immutable distributed pytest selection."""

    row_id: str
    world_size: int
    allocations: tuple[tuple[int, int], ...]
    pytest_nodes: tuple[str, ...]
    pytest_filters: tuple[str, ...]


@dataclass(frozen=True)
class SubmissionArtifacts:
    """Fresh immutable source and intent artifacts for one submission."""

    snapshot_root: Path
    snapshot_sha256: str
    intent_path: Path
    intent_sha256: str


def _directory_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root).as_posix()
        if relative == ".snapshot-sha256":
            continue
        metadata = path.lstat()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(f"exec={stat.S_IMODE(metadata.st_mode) & 0o111:o}".encode())
        digest.update(b"\0")
        if stat.S_ISREG(metadata.st_mode):
            digest.update(b"file\0")
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
        elif stat.S_ISDIR(metadata.st_mode):
            digest.update(b"directory\0")
        elif stat.S_ISLNK(metadata.st_mode):
            target = os.readlink(path)
            resolved = (path.parent / target).resolve()
            if not resolved.is_relative_to(root.resolve()):
                raise ValueError(f"snapshot contains an escaping symlink: {relative}")
            digest.update(b"symlink\0")
            digest.update(target.encode())
        else:
            raise ValueError(f"snapshot contains an unsupported file type: {relative}")
        digest.update(b"\0")
    return digest.hexdigest()


def _make_tree_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            continue
        mode = stat.S_IMODE(path.stat().st_mode)
        path.chmod(mode & ~0o222)
    root.chmod(stat.S_IMODE(root.stat().st_mode) & ~0o222)


def verify_source_snapshot(
    *, source_root: Path, candidate_sha: str, expected_sha256: str
) -> None:
    """Reject mutable, unsafe, or content-mismatched submission snapshots."""
    if FULL_COMMIT.fullmatch(candidate_sha) is None:
        raise ValueError("candidate SHA must be a full lowercase 40-character SHA")
    if FULL_SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("snapshot SHA256 must be lowercase hexadecimal")
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError("candidate source root is missing or unsafe")
    for path in (source_root, *source_root.rglob("*")):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            continue
        if metadata.st_mode & 0o222:
            raise ValueError(f"snapshot contains a writable path: {path}")
    marker = source_root / ".candidate-sha"
    digest_marker = source_root / ".snapshot-sha256"
    if not marker.is_file() or marker.read_text().strip() != candidate_sha:
        raise ValueError("candidate source snapshot does not match candidate SHA")
    if (
        not digest_marker.is_file()
        or digest_marker.read_text().strip() != expected_sha256
    ):
        raise ValueError("snapshot SHA256 marker mismatch")
    if _directory_sha256(source_root) != expected_sha256:
        raise ValueError("snapshot SHA256 does not match snapshot contents")


def load_submission_intent(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    """Load one immutable intent only when its literal submit-time digest matches."""
    if FULL_SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("submission intent SHA256 must be lowercase hexadecimal")
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222:
        raise ValueError("submission intent must be a non-writable regular file")
    serialized = path.read_bytes()
    if hashlib.sha256(serialized).hexdigest() != expected_sha256:
        raise ValueError("submission intent SHA256 does not match intent contents")
    try:
        payload = json.loads(serialized)
    except json.JSONDecodeError as error:
        raise ValueError("submission intent is invalid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("submission intent must be a JSON object")
    return payload


def _archive_commit(repository: Path, commit: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    archive = subprocess.Popen(
        ["git", "-C", str(repository), "archive", "--format=tar", commit],
        stdout=subprocess.PIPE,
    )
    assert archive.stdout is not None
    try:
        extracted = subprocess.run(
            ["tar", "-xf", "-", "-C", str(destination)],
            stdin=archive.stdout,
            check=False,
        )
    finally:
        archive.stdout.close()
    archive_status = archive.wait()
    if archive_status != 0 or extracted.returncode != 0:
        raise RuntimeError(f"failed to archive {repository} at {commit}")


def prepare_candidate_submission(
    *,
    archive_sources: tuple[tuple[Path, str, Path], ...],
    run_log_root: Path,
    candidate_kind: str,
    candidate_sha: str,
    intent_payload: Mapping[str, Any],
) -> SubmissionArtifacts:
    """Publish a fresh, content-bound snapshot and exclusive immutable intent."""
    if not run_log_root.is_absolute():
        raise ValueError("RUN_LOG_ROOT must be absolute")
    if candidate_kind not in {"mcore", "bridge"}:
        raise ValueError("candidate kind must be mcore or bridge")
    if FULL_COMMIT.fullmatch(candidate_sha) is None:
        raise ValueError("candidate SHA must be a full lowercase 40-character SHA")
    if not archive_sources:
        raise ValueError("at least one archive source is required")
    submission_id = f"{time.time_ns()}-{os.getpid()}-{uuid.uuid4().hex}"
    snapshot_parent = run_log_root / "source-snapshots" / candidate_kind / candidate_sha
    snapshot_parent.mkdir(parents=True, exist_ok=True)
    temporary_root = snapshot_parent / f".{submission_id}.tmp"
    final_root = snapshot_parent / submission_id
    temporary_root.mkdir(mode=0o700)
    try:
        for repository, commit, relative_destination in archive_sources:
            if relative_destination.is_absolute() or ".." in relative_destination.parts:
                raise ValueError("archive destination must stay inside the snapshot")
            destination = temporary_root / relative_destination
            _archive_commit(repository, commit, destination)
        candidate_marker = temporary_root / ".candidate-sha"
        digest_marker = temporary_root / ".snapshot-sha256"
        if candidate_marker.exists() or candidate_marker.is_symlink():
            raise ValueError("candidate archive contains a reserved snapshot marker")
        if digest_marker.exists() or digest_marker.is_symlink():
            raise ValueError("candidate archive contains a reserved digest marker")
        candidate_marker.write_text(f"{candidate_sha}\n")
        snapshot_sha256 = _directory_sha256(temporary_root)
        digest_marker.write_text(f"{snapshot_sha256}\n")
        _make_tree_read_only(temporary_root)
        os.replace(temporary_root, final_root)
        directory_fd = os.open(snapshot_parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if temporary_root.exists():
            _make_tree_read_only(temporary_root)
            for path in sorted(temporary_root.rglob("*"), reverse=True):
                if not path.is_symlink():
                    path.chmod(stat.S_IMODE(path.stat().st_mode) | 0o200)
            temporary_root.chmod(0o700)
            shutil.rmtree(temporary_root)
        raise
    verify_source_snapshot(
        source_root=final_root,
        candidate_sha=candidate_sha,
        expected_sha256=snapshot_sha256,
    )

    payload = dict(intent_payload)
    payload["snapshot_path"] = str(final_root)
    payload["snapshot_sha256"] = snapshot_sha256
    serialized = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    intent_sha256 = hashlib.sha256(serialized).hexdigest()
    intent_parent = run_log_root / "submission-intents" / candidate_kind / candidate_sha
    intent_parent.mkdir(parents=True, exist_ok=True)
    intent_path = intent_parent / f"{submission_id}.json"
    temporary_intent = intent_parent / f".{submission_id}.tmp"
    descriptor = os.open(temporary_intent, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as output:
            output.write(serialized)
            output.flush()
            os.fsync(output.fileno())
        temporary_intent.chmod(0o444)
        read_descriptor = os.open(temporary_intent, os.O_RDONLY)
        try:
            os.fsync(read_descriptor)
        finally:
            os.close(read_descriptor)
        os.link(temporary_intent, intent_path)
        directory_fd = os.open(intent_parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_intent.unlink(missing_ok=True)
    load_submission_intent(intent_path, expected_sha256=intent_sha256)
    return SubmissionArtifacts(final_root, snapshot_sha256, intent_path, intent_sha256)


def pytest_commands(
    row: MatrixRow, *, python_executable: Path
) -> tuple[tuple[str, ...], ...]:
    """Render each literal pytest node as a separate argv-only invocation."""
    return tuple(
        (
            str(python_executable),
            "-m",
            "pytest",
            "-q",
            *row.pytest_filters,
            node,
        )
        for node in row.pytest_nodes
    )


def validate_pytest_node_collection(
    *,
    source_root: Path,
    rows: Mapping[str, MatrixRow],
    python_executable: Path,
) -> tuple[str, ...]:
    """Require every literal node to collect from the exact candidate source tree."""
    expected_nodes = tuple(
        dict.fromkeys(node for row in rows.values() for node in row.pytest_nodes)
    )
    if not expected_nodes:
        raise ValueError("candidate collection requires at least one literal pytest node")
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        (
            str(python_executable),
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            *expected_nodes,
        ),
        cwd=source_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    collected = frozenset(
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip().startswith("tests/") and "::" in line
    )
    missing = tuple(node for node in expected_nodes if node not in collected)
    if completed.returncode != 0 or missing:
        owners = tuple(
            f"{row.row_id}: {node}"
            for row in rows.values()
            for node in row.pytest_nodes
            if node in missing
        )
        detail = "; ".join(owners) or completed.stderr.strip() or "collection failed"
        raise ValueError(f"candidate archive is missing literal pytest nodes: {detail}")
    return expected_nodes


def validate_allocation(*, num_nodes: int, gpus_per_node: int, world_size: int) -> int:
    """Require one explicitly supported allocation matching the row world size."""
    if (num_nodes, gpus_per_node, world_size) not in ALLOWED_ALLOCATIONS:
        raise ValueError(
            "unsupported allocation/world size layout: "
            f"{num_nodes}x{gpus_per_node} for world size {world_size}"
        )
    if num_nodes * gpus_per_node != world_size:
        raise ValueError("allocation does not match world size")
    return world_size


def validate_device_bindings(
    bindings: tuple[Mapping[str, Any], ...],
    *,
    world_size: int,
    num_nodes: int,
    gpus_per_node: int,
) -> tuple[dict[str, int], ...]:
    """Require one unique local-rank/CUDA-device slot for every global rank."""
    validate_allocation(
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        world_size=world_size,
    )
    if len(bindings) != world_size:
        raise ValueError("device bindings must contain every global rank")
    normalized: list[dict[str, int]] = []
    expected_keys = {"global_rank", "node_rank", "local_rank", "cuda_device_index"}
    for binding in bindings:
        if not isinstance(binding, Mapping) or set(binding) != expected_keys:
            raise ValueError("device binding has an invalid schema")
        if any(
            not isinstance(binding[key], int) or isinstance(binding[key], bool)
            for key in expected_keys
        ):
            raise ValueError("device binding values must be integers")
        global_rank = binding["global_rank"]
        node_rank = binding["node_rank"]
        local_rank = binding["local_rank"]
        cuda_device_index = binding["cuda_device_index"]
        if global_rank not in range(world_size):
            raise ValueError("device binding global rank is out of range")
        if node_rank not in range(num_nodes):
            raise ValueError("device binding node rank is out of range")
        if local_rank not in range(gpus_per_node):
            raise ValueError("device binding local rank is out of range")
        if cuda_device_index != local_rank:
            raise ValueError("device binding CUDA device does not match local rank")
        normalized.append(dict(binding))
    normalized.sort(key=lambda binding: binding["global_rank"])
    if [binding["global_rank"] for binding in normalized] != list(range(world_size)):
        raise ValueError("device bindings contain duplicate or missing global ranks")
    actual_slots = {
        (binding["node_rank"], binding["local_rank"]) for binding in normalized
    }
    expected_slots = {
        (node_rank, local_rank)
        for node_rank in range(num_nodes)
        for local_rank in range(gpus_per_node)
    }
    if actual_slots != expected_slots or len(actual_slots) != world_size:
        raise ValueError(
            "device bindings contain duplicate or missing per-node device slots"
        )
    if any(
        binding["global_rank"]
        != binding["node_rank"] * gpus_per_node + binding["local_rank"]
        for binding in normalized
    ):
        raise ValueError("device binding global rank does not match its device slot")
    return tuple(normalized)


def _require_string_list(value: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a JSON string list")
    return tuple(value)


def load_matrix(path: Path, *, candidate_kind: str) -> dict[str, MatrixRow]:
    """Load a typed manifest without accepting executable command payloads."""
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"test matrix is invalid JSON: {path}") from error
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("test matrix must use schema version 1")
    if payload.get("candidate_kind") != candidate_kind:
        raise ValueError("test matrix candidate kind mismatch")
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, Mapping) or not raw_rows:
        raise ValueError("test matrix rows must be a non-empty JSON object")

    rows: dict[str, MatrixRow] = {}
    for row_id, raw_row in raw_rows.items():
        if not isinstance(row_id, str) or ROW_ID.fullmatch(row_id) is None:
            raise ValueError(f"invalid test row ID: {row_id!r}")
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"test row {row_id} must be a JSON object")
        unknown = set(raw_row).difference(
            {"world_size", "allocations", "pytest_nodes", "pytest_filters"}
        )
        if unknown:
            raise ValueError(
                f"test row {row_id} contains unsupported command fields: {sorted(unknown)}"
            )
        world_size = raw_row.get("world_size")
        if not isinstance(world_size, int) or isinstance(world_size, bool):
            raise ValueError(f"test row {row_id} world_size must be an integer")
        raw_allocations = raw_row.get("allocations")
        if not isinstance(raw_allocations, list) or not raw_allocations:
            raise ValueError(f"test row {row_id} allocations must be a non-empty list")
        allocations: list[tuple[int, int]] = []
        for allocation in raw_allocations:
            if (
                not isinstance(allocation, Mapping)
                or set(allocation) != {"num_nodes", "gpus_per_node"}
                or not isinstance(allocation["num_nodes"], int)
                or not isinstance(allocation["gpus_per_node"], int)
            ):
                raise ValueError(f"test row {row_id} has an invalid allocation")
            num_nodes = allocation["num_nodes"]
            gpus_per_node = allocation["gpus_per_node"]
            validate_allocation(
                num_nodes=num_nodes,
                gpus_per_node=gpus_per_node,
                world_size=world_size,
            )
            allocations.append((num_nodes, gpus_per_node))
        pytest_nodes = _require_string_list(
            raw_row.get("pytest_nodes"), label=f"test row {row_id} pytest_nodes"
        )
        if not pytest_nodes or any(
            PYTEST_NODE.fullmatch(node) is None for node in pytest_nodes
        ):
            raise ValueError(f"test row {row_id} contains a nonliteral pytest node")
        pytest_filters = _require_string_list(
            raw_row.get("pytest_filters"), label=f"test row {row_id} pytest_filters"
        )
        rows[row_id] = MatrixRow(
            row_id=row_id,
            world_size=world_size,
            allocations=tuple(allocations),
            pytest_nodes=pytest_nodes,
            pytest_filters=pytest_filters,
        )
    return rows


def result_path(
    *, run_log_root: Path, candidate_kind: str, candidate_sha: str, row_id: str
) -> Path:
    """Return the content-bound attestation path without allowing path escape."""
    if not run_log_root.is_absolute():
        raise ValueError("RUN_LOG_ROOT must be absolute")
    if candidate_kind not in {"mcore", "bridge"}:
        raise ValueError("candidate kind must be mcore or bridge")
    if FULL_COMMIT.fullmatch(candidate_sha) is None:
        raise ValueError("candidate SHA must be a full lowercase 40-character SHA")
    if ROW_ID.fullmatch(row_id) is None:
        raise ValueError("row ID must be filesystem-safe")
    return (
        run_log_root
        / "attestations"
        / candidate_kind
        / candidate_sha
        / f"{row_id}.json"
    )


def derive_run_identity(
    *,
    scheduler_job_id: str,
    scheduler_restart_count: int,
    submission_intent_sha256: str,
) -> str:
    """Bind rank exchange to one scheduler attempt and immutable intent."""
    if SCHEDULER_JOB_ID.fullmatch(scheduler_job_id) is None:
        raise ValueError("scheduler job ID must be a positive decimal integer")
    if scheduler_restart_count < 0:
        raise ValueError("scheduler restart count must be non-negative")
    if FULL_SHA256.fullmatch(submission_intent_sha256) is None:
        raise ValueError("submission intent SHA256 must be lowercase hexadecimal")
    return (
        f"slurm-{scheduler_job_id}-{scheduler_restart_count}-{submission_intent_sha256}"
    )


def rank_result_dir(
    *,
    run_log_root: Path,
    candidate_kind: str,
    candidate_sha: str,
    row_id: str,
    run_identity: str,
) -> Path:
    """Return a run-unique rank-exchange directory."""
    result_path(
        run_log_root=run_log_root,
        candidate_kind=candidate_kind,
        candidate_sha=candidate_sha,
        row_id=row_id,
    )
    if RUN_IDENTITY.fullmatch(run_identity) is None:
        raise ValueError("run identity is invalid")
    return (
        run_log_root
        / "rank-results"
        / candidate_kind
        / candidate_sha
        / row_id
        / run_identity
    )


def validate_rank_payloads(
    payloads: tuple[Mapping[str, Any], ...],
    *,
    run_identity: str,
    candidate_kind: str,
    candidate_sha: str,
    row_id: str,
    world_size: int,
    num_nodes: int,
    gpus_per_node: int,
    pytest_nodes: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    """Validate rank identity, topology, node results, and capability consensus."""
    validate_allocation(
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        world_size=world_size,
    )
    if RUN_IDENTITY.fullmatch(run_identity) is None:
        raise ValueError("run identity is invalid")
    if len(payloads) != world_size:
        raise ValueError("rank payloads must contain every global rank")
    expected_keys = {
        "run_identity",
        "rank",
        "world_size",
        "num_nodes",
        "gpus_per_node",
        "candidate_kind",
        "candidate_sha",
        "test_row_id",
        "node_results",
        "capability",
    }
    normalized: list[dict[str, Any]] = []
    semantic_capability: dict[str, Any] | None = None
    for expected_rank, payload in enumerate(payloads):
        if not isinstance(payload, Mapping) or set(payload) != expected_keys:
            raise ValueError("rank payload has an invalid schema")
        expected_values = {
            "run_identity": run_identity,
            "rank": expected_rank,
            "world_size": world_size,
            "num_nodes": num_nodes,
            "gpus_per_node": gpus_per_node,
            "candidate_kind": candidate_kind,
            "candidate_sha": candidate_sha,
            "test_row_id": row_id,
        }
        for field, expected in expected_values.items():
            if payload.get(field) != expected:
                label = field.replace("_", " ")
                raise ValueError(f"rank payload {label} mismatch")
        node_results = payload.get("node_results")
        if not isinstance(node_results, list) or len(node_results) != len(pytest_nodes):
            raise ValueError("rank payload node results mismatch")
        for expected_node, node_result in zip(pytest_nodes, node_results, strict=True):
            if (
                not isinstance(node_result, Mapping)
                or set(node_result) != {"node", "status", "exit_code"}
                or node_result.get("node") != expected_node
                or node_result.get("status") not in {"passed", "failed"}
                or not isinstance(node_result.get("exit_code"), int)
                or isinstance(node_result.get("exit_code"), bool)
            ):
                raise ValueError("rank payload node result has an invalid schema")
        capability = payload.get("capability")
        if not isinstance(capability, Mapping):
            raise ValueError("rank payload capability must be a JSON object")
        expected_device_binding = {
            "global_rank": expected_rank,
            "node_rank": expected_rank // gpus_per_node,
            "local_rank": expected_rank % gpus_per_node,
            "cuda_device_index": expected_rank % gpus_per_node,
        }
        if any(
            capability.get(key) != value
            for key, value in expected_device_binding.items()
        ):
            raise ValueError("rank payload device binding mismatch")
        current_semantic_capability = {
            key: value
            for key, value in capability.items()
            if key not in CAPABILITY_DEVICE_FIELDS
        }
        if semantic_capability is None:
            semantic_capability = current_semantic_capability
        elif current_semantic_capability != semantic_capability:
            raise ValueError("semantic capability metadata differs across ranks")
        normalized.append(dict(payload))
    return tuple(normalized)


def build_result(
    *,
    run_identity: str,
    candidate_kind: str,
    candidate_sha: str,
    integration_sha: str,
    row_id: str,
    world_size: int,
    num_nodes: int,
    gpus_per_node: int,
    joined_ranks: tuple[int, ...],
    device_bindings: tuple[Mapping[str, Any], ...],
    node_results: tuple[Mapping[str, Any], ...],
    container_sha256: str,
    transformer_engine_version: str,
    transformer_engine_source_commit: str,
    transformer_engine_version_base_commit: str,
    all_eval_callables_supported: bool,
    mcore_eval_reuse_graph_io: bool | str,
    raw_te_eval_reuse_graph_io: bool,
    capability_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one passed result only after every rank and node has passed."""
    if RUN_IDENTITY.fullmatch(run_identity) is None:
        raise ValueError("run identity is invalid")
    result_path(
        run_log_root=Path("/attestation-validation"),
        candidate_kind=candidate_kind,
        candidate_sha=candidate_sha,
        row_id=row_id,
    )
    if FULL_COMMIT.fullmatch(integration_sha) is None:
        raise ValueError("integration SHA must be a full lowercase 40-character SHA")
    for label, commit in (
        ("Transformer Engine source commit", transformer_engine_source_commit),
        (
            "Transformer Engine version-base commit",
            transformer_engine_version_base_commit,
        ),
    ):
        if FULL_COMMIT.fullmatch(commit) is None:
            raise ValueError(f"{label} must be a full lowercase SHA")
    if FULL_SHA256.fullmatch(container_sha256) is None:
        raise ValueError("container SHA256 must be 64 lowercase hexadecimal characters")
    if joined_ranks != tuple(range(world_size)):
        raise ValueError("joined ranks must contain every global rank exactly once")
    normalized_bindings = validate_device_bindings(
        device_bindings,
        world_size=world_size,
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
    )
    if not node_results:
        raise ValueError("node results must not be empty")
    if any(
        result.get("status") != "passed" or result.get("exit_code") != 0
        for result in node_results
    ):
        raise ValueError("every pytest node must pass")
    return {
        "schema_version": 1,
        "status": "passed",
        "run_identity": run_identity,
        "candidate_kind": candidate_kind,
        "candidate_sha": candidate_sha,
        "integration_sha": integration_sha,
        "test_row_id": row_id,
        "container_sha256": container_sha256,
        "transformer_engine_version": transformer_engine_version,
        "transformer_engine_source_commit": transformer_engine_source_commit,
        "transformer_engine_version_base_commit": transformer_engine_version_base_commit,
        "all_eval_callables_supported": all_eval_callables_supported,
        "mcore_eval_reuse_graph_io": mcore_eval_reuse_graph_io,
        "raw_te_eval_reuse_graph_io": raw_te_eval_reuse_graph_io,
        "capability_evidence": dict(capability_evidence or {}),
        "topology": {
            "world_size": world_size,
            "num_nodes": num_nodes,
            "gpus_per_node": gpus_per_node,
            "joined_ranks": list(joined_ranks),
            "device_bindings": list(normalized_bindings),
        },
        "node_results": [dict(result) for result in node_results],
    }


def write_json_atomic(payload: Mapping[str, Any], output: Path) -> None:
    """Publish one JSON object atomically without following a final symlink."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise ValueError(f"result path must not be a symlink: {output}")
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output.parent,
            prefix=f".{output.name}.",
            delete=False,
        ) as temporary:
            temporary_path = temporary.name
            json.dump(payload, temporary, allow_nan=False, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, output)
        output.chmod(0o644)
    except BaseException:
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)
        raise


def _capability_from_output(output: str) -> tuple[dict[str, Any], ...]:
    capabilities: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.startswith("TE_CAPABILITY_JSON="):
            continue
        try:
            parsed = json.loads(line.removeprefix("TE_CAPABILITY_JSON="))
        except json.JSONDecodeError as error:
            raise ValueError("TE capability output is invalid JSON") from error
        if not isinstance(parsed, Mapping):
            raise ValueError("TE capability output must be a JSON object")
        capabilities.append(dict(parsed))
    return tuple(capabilities)


def _exact_capability_marker(
    node_capabilities: Mapping[str, tuple[dict[str, Any], ...]],
    *,
    node: str,
    expected_keys: frozenset[str],
) -> dict[str, Any]:
    markers = node_capabilities.get(node)
    if not isinstance(markers, (tuple, list)) or len(markers) != 1:
        raise ValueError(
            f"TE capability evidence requires exactly one marker for {node}"
        )
    marker = markers[0]
    if not isinstance(marker, Mapping) or set(marker) != expected_keys:
        raise ValueError(f"TE capability evidence has an invalid schema for {node}")
    return dict(marker)


def validate_row_capability(
    *,
    row_id: str,
    node_capabilities: Mapping[str, tuple[dict[str, Any], ...]],
    expected_device_binding: Mapping[str, int],
) -> dict[str, Any]:
    """Require exact, affirmative evidence from both TE capability tests."""
    device_keys = frozenset(("node_rank", "local_rank", "cuda_device_index"))
    if (
        not isinstance(expected_device_binding, Mapping)
        or set(expected_device_binding) != device_keys
        or any(
            not isinstance(expected_device_binding[key], int)
            or isinstance(expected_device_binding[key], bool)
            for key in device_keys
        )
    ):
        raise ValueError("expected capability device binding has an invalid schema")
    if row_id != "te_eval_capability_8":
        return dict(expected_device_binding)
    primary = _exact_capability_marker(
        node_capabilities,
        node=PRIMARY_CAPABILITY_NODE,
        expected_keys=device_keys
        | frozenset(
            (
                "all_eval_callables_supported",
                "backward_executed",
                "fallback_forward_counter_increment",
                "forward_invocations_after_capture",
                "no_parameter_grads",
                "outputs_changed",
                "replay_forward_counter_increment",
            )
        ),
    )
    reuse = _exact_capability_marker(
        node_capabilities,
        node=REUSE_CAPABILITY_NODE,
        expected_keys=device_keys
        | frozenset(
            (
                "mcore_eval_reuse_graph_io",
                "raw_te_eval_reuse_graph_io",
                "raw_te_eval_reuse_rejection",
                "raw_te_eval_reuse_eager_parity",
                "raw_te_eval_reuse_fallback_forward_counter_increment",
                "raw_te_eval_reuse_no_parameter_grads",
                "raw_te_eval_reuse_outputs_changed",
                "raw_te_eval_reuse_replay_forward_counter_increment",
            )
        ),
    )
    for marker in (primary, reuse):
        if any(marker[key] != expected_device_binding[key] for key in device_keys):
            raise ValueError(
                "TE capability evidence reports an unexpected measured device binding"
            )
    required_primary = {
        "all_eval_callables_supported": True,
        "backward_executed": False,
        "fallback_forward_counter_increment": 1,
        "no_parameter_grads": True,
        "outputs_changed": True,
        "replay_forward_counter_increment": 0,
    }
    if any(primary.get(key) != value for key, value in required_primary.items()):
        raise ValueError("TE capability evidence does not prove safe eval replay")
    forward_count = primary.get("forward_invocations_after_capture")
    if (
        not isinstance(forward_count, int)
        or isinstance(forward_count, bool)
        or forward_count <= 0
    ):
        raise ValueError("TE capability evidence has an invalid forward counter")
    if reuse.get("mcore_eval_reuse_graph_io") != "not_implemented":
        raise ValueError("TE capability evidence has an invalid MCore reuse status")
    if reuse.get("raw_te_eval_reuse_no_parameter_grads") is not True:
        raise ValueError("TE capability evidence does not prove reuse is no-grad")
    raw_reuse = reuse.get("raw_te_eval_reuse_graph_io")
    if not isinstance(raw_reuse, bool):
        raise ValueError("TE capability evidence has an invalid raw TE reuse status")
    if raw_reuse:
        accepted = {
            "raw_te_eval_reuse_rejection": None,
            "raw_te_eval_reuse_eager_parity": True,
            "raw_te_eval_reuse_fallback_forward_counter_increment": 1,
            "raw_te_eval_reuse_outputs_changed": True,
            "raw_te_eval_reuse_replay_forward_counter_increment": 0,
        }
        if any(reuse.get(key) != value for key, value in accepted.items()):
            raise ValueError("TE capability evidence does not prove safe reuse replay")
    else:
        rejection = reuse.get("raw_te_eval_reuse_rejection")
        if not isinstance(rejection, str) or not rejection.strip():
            raise ValueError("TE capability evidence lacks the raw TE reuse rejection")
        unavailable = (
            "raw_te_eval_reuse_eager_parity",
            "raw_te_eval_reuse_fallback_forward_counter_increment",
            "raw_te_eval_reuse_outputs_changed",
            "raw_te_eval_reuse_replay_forward_counter_increment",
        )
        if any(reuse.get(key) is not None for key in unavailable):
            raise ValueError(
                "TE capability evidence reports inconsistent rejected reuse"
            )
    return {**primary, **reuse}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--row-id", required=True)
    parser.add_argument("--candidate-kind", choices=("mcore", "bridge"), required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--integration-sha", required=True)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--snapshot-sha256", required=True)
    parser.add_argument("--run-log-root", required=True, type=Path)
    parser.add_argument("--num-nodes", required=True, type=int)
    parser.add_argument("--gpus-per-node", required=True, type=int)
    parser.add_argument("--container-sha256", required=True)
    parser.add_argument("--transformer-engine-version", required=True)
    parser.add_argument("--transformer-engine-source-commit", required=True)
    parser.add_argument("--transformer-engine-version-base-commit", required=True)
    parser.add_argument("--scheduler-job-id", required=True)
    parser.add_argument("--scheduler-restart-count", required=True, type=int)
    parser.add_argument("--submission-intent-sha256", required=True)
    parser.add_argument("--launch-agent", action="store_true")
    return parser.parse_args()


def _launch_torchrun_agent(args: argparse.Namespace) -> None:
    node_rank = int(os.environ.get("SLURM_NODEID", "-1"))
    if node_rank not in range(args.num_nodes):
        raise ValueError("SLURM_NODEID does not match the typed node allocation")
    child_arguments = [
        argument for argument in sys.argv[1:] if argument != "--launch-agent"
    ]
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nnodes={args.num_nodes}",
        f"--nproc-per-node={args.gpus_per_node}",
        f"--node-rank={node_rank}",
        f"--master-addr={os.environ['MASTER_ADDR']}",
        f"--master-port={os.environ['MASTER_PORT']}",
        str(Path(__file__).resolve()),
        *child_arguments,
    ]
    os.execv(sys.executable, command)


def main() -> int:
    args = _parse_args()
    if args.launch_agent:
        _launch_torchrun_agent(args)
        raise AssertionError("torchrun agent exec unexpectedly returned")
    run_identity = derive_run_identity(
        scheduler_job_id=args.scheduler_job_id,
        scheduler_restart_count=args.scheduler_restart_count,
        submission_intent_sha256=args.submission_intent_sha256,
    )
    rows = load_matrix(args.matrix, candidate_kind=args.candidate_kind)
    try:
        row = rows[args.row_id]
    except KeyError as error:
        raise ValueError(f"unknown test row: {args.row_id}") from error
    validate_allocation(
        num_nodes=args.num_nodes,
        gpus_per_node=args.gpus_per_node,
        world_size=row.world_size,
    )
    rank = int(os.environ.get("RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    if world_size != row.world_size or rank not in range(world_size):
        raise ValueError("torchrun rank environment does not match the typed row")
    verify_source_snapshot(
        source_root=args.source_root,
        candidate_sha=args.candidate_sha,
        expected_sha256=args.snapshot_sha256,
    )
    validate_pytest_node_collection(
        source_root=args.source_root,
        rows=rows,
        python_executable=Path(sys.executable),
    )

    node_results: list[dict[str, Any]] = []
    node_capabilities: dict[str, tuple[dict[str, Any], ...]] = {}
    for node, command in zip(
        row.pytest_nodes,
        pytest_commands(row, python_executable=Path(sys.executable)),
        strict=True,
    ):
        completed = subprocess.run(
            command,
            cwd=args.source_root,
            check=False,
            capture_output=True,
            text=True,
        )
        output = completed.stdout + completed.stderr
        sys.stdout.write(output)
        node_capabilities[node] = _capability_from_output(output)
        node_results.append(
            {
                "node": node,
                "status": "passed" if completed.returncode == 0 else "failed",
                "exit_code": completed.returncode,
            }
        )

    rank_dir = rank_result_dir(
        run_log_root=args.run_log_root,
        candidate_kind=args.candidate_kind,
        candidate_sha=args.candidate_sha,
        row_id=args.row_id,
        run_identity=run_identity,
    )
    capability = validate_row_capability(
        row_id=args.row_id,
        node_capabilities=node_capabilities,
        expected_device_binding={
            "node_rank": rank // args.gpus_per_node,
            "local_rank": rank % args.gpus_per_node,
            "cuda_device_index": rank % args.gpus_per_node,
        },
    )
    capability["global_rank"] = rank
    rank_payload = {
        "run_identity": run_identity,
        "rank": rank,
        "world_size": world_size,
        "num_nodes": args.num_nodes,
        "gpus_per_node": args.gpus_per_node,
        "candidate_kind": args.candidate_kind,
        "candidate_sha": args.candidate_sha,
        "test_row_id": args.row_id,
        "node_results": node_results,
        "capability": capability,
    }
    write_json_atomic(rank_payload, rank_dir / f"rank-{rank}.json")
    if rank != 0:
        return 0 if all(item["exit_code"] == 0 for item in node_results) else 1

    deadline = time.monotonic() + float(
        os.environ.get("RUNNER_JOIN_TIMEOUT_SECONDS", "300")
    )
    rank_files = [rank_dir / f"rank-{item}.json" for item in range(world_size)]
    while not all(path.is_file() for path in rank_files):
        if time.monotonic() >= deadline:
            raise RuntimeError("timed out waiting for every global rank to join")
        time.sleep(0.25)
    rank_payloads = validate_rank_payloads(
        tuple(json.loads(path.read_text()) for path in rank_files),
        run_identity=run_identity,
        candidate_kind=args.candidate_kind,
        candidate_sha=args.candidate_sha,
        row_id=args.row_id,
        world_size=world_size,
        num_nodes=args.num_nodes,
        gpus_per_node=args.gpus_per_node,
        pytest_nodes=row.pytest_nodes,
    )
    joined_ranks = tuple(sorted(payload["rank"] for payload in rank_payloads))
    combined_results: list[dict[str, Any]] = []
    for node_index, node in enumerate(row.pytest_nodes):
        per_rank = [payload["node_results"][node_index] for payload in rank_payloads]
        passed = all(
            item["status"] == "passed" and item["exit_code"] == 0 for item in per_rank
        )
        combined_results.append(
            {
                "node": node,
                "status": "passed" if passed else "failed",
                "exit_code": 0 if passed else 1,
            }
        )
    rank_zero_capability = rank_payloads[0]["capability"]
    device_bindings = tuple(
        {
            "global_rank": payload["rank"],
            "node_rank": payload["capability"].get("node_rank"),
            "local_rank": payload["capability"].get("local_rank"),
            "cuda_device_index": payload["capability"].get("cuda_device_index"),
        }
        for payload in rank_payloads
    )
    payload = build_result(
        run_identity=run_identity,
        candidate_kind=args.candidate_kind,
        candidate_sha=args.candidate_sha,
        integration_sha=args.integration_sha,
        row_id=args.row_id,
        world_size=world_size,
        num_nodes=args.num_nodes,
        gpus_per_node=args.gpus_per_node,
        joined_ranks=joined_ranks,
        device_bindings=device_bindings,
        node_results=tuple(combined_results),
        container_sha256=args.container_sha256,
        transformer_engine_version=args.transformer_engine_version,
        transformer_engine_source_commit=args.transformer_engine_source_commit,
        transformer_engine_version_base_commit=(
            args.transformer_engine_version_base_commit
        ),
        all_eval_callables_supported=bool(
            rank_zero_capability.get("all_eval_callables_supported", False)
        ),
        mcore_eval_reuse_graph_io=rank_zero_capability.get(
            "mcore_eval_reuse_graph_io", "not_implemented"
        ),
        raw_te_eval_reuse_graph_io=bool(
            rank_zero_capability.get("raw_te_eval_reuse_graph_io", False)
        ),
        capability_evidence={
            key: value
            for key, value in rank_zero_capability.items()
            if key not in CAPABILITY_DEVICE_FIELDS
        },
    )
    write_json_atomic(
        payload,
        result_path(
            run_log_root=args.run_log_root,
            candidate_kind=args.candidate_kind,
            candidate_sha=args.candidate_sha,
            row_id=args.row_id,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
