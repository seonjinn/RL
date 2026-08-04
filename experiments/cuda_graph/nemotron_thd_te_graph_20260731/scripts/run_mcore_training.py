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
import json
import os
import re
import subprocess
import sys
import tempfile
import time
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
ALLOWED_ALLOCATIONS = frozenset(((1, 8, 8), (2, 4, 8), (4, 4, 16), (8, 4, 32)))
CAPABILITY_DEVICE_FIELDS = frozenset(
    ("global_rank", "node_rank", "local_rank", "cuda_device_index")
)


@dataclass(frozen=True)
class MatrixRow:
    """One immutable distributed pytest selection."""

    row_id: str
    world_size: int
    allocations: tuple[tuple[int, int], ...]
    pytest_nodes: tuple[str, ...]
    pytest_filters: tuple[str, ...]


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
        if not pytest_nodes or any(PYTEST_NODE.fullmatch(node) is None for node in pytest_nodes):
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
    return run_log_root / "attestations" / candidate_kind / candidate_sha / f"{row_id}.json"


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
        f"slurm-{scheduler_job_id}-{scheduler_restart_count}-"
        f"{submission_intent_sha256}"
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
        ("Transformer Engine version-base commit", transformer_engine_version_base_commit),
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


def _capability_from_output(output: str) -> dict[str, Any]:
    capability: dict[str, Any] = {}
    for line in output.splitlines():
        if not line.startswith("TE_CAPABILITY_JSON="):
            continue
        try:
            parsed = json.loads(line.removeprefix("TE_CAPABILITY_JSON="))
        except json.JSONDecodeError as error:
            raise ValueError("TE capability output is invalid JSON") from error
        if not isinstance(parsed, Mapping):
            raise ValueError("TE capability output must be a JSON object")
        capability.update(parsed)
    return capability


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--row-id", required=True)
    parser.add_argument("--candidate-kind", choices=("mcore", "bridge"), required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--integration-sha", required=True)
    parser.add_argument("--source-root", required=True, type=Path)
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
    child_arguments = [argument for argument in sys.argv[1:] if argument != "--launch-agent"]
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
    if args.source_root.is_symlink() or not args.source_root.is_dir():
        raise ValueError("candidate source root is missing or unsafe")
    marker = args.source_root / ".candidate-sha"
    if not marker.is_file() or marker.read_text().strip() != args.candidate_sha:
        raise ValueError("candidate source snapshot does not match candidate SHA")

    node_results: list[dict[str, Any]] = []
    capability: dict[str, Any] = {}
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
        capability.update(_capability_from_output(output))
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

    deadline = time.monotonic() + float(os.environ.get("RUNNER_JOIN_TIMEOUT_SECONDS", "300"))
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
        passed = all(item["status"] == "passed" and item["exit_code"] == 0 for item in per_rank)
        combined_results.append(
            {"node": node, "status": "passed" if passed else "failed", "exit_code": 0 if passed else 1}
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
