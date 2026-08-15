#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Verify one immutable OCI runtime preflight artifact for a leaf job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")
FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
MINIMUM_TE_VERSION = (2, 16)
REQUIRED_TE_GROUPED_LINEAR_SYMBOLS = (
    "TEColumnParallelGroupedLinear",
    "TERowParallelGroupedLinear",
)
REQUIRED_PACKAGES = frozenset(
    (
        "torch",
        "transformer_engine.pytorch",
        "megatron.core",
        "megatron.core.extensions.transformer_engine",
        "megatron.bridge",
        "mamba_ssm",
        "causal_conv1d",
        "cupy",
    )
)
TE_EVAL_FEATURE_SET = "te_eval_capability_8"
BRIDGE_EVAL_FEATURE_SET = "bridge_forward_only_eval_8"
NARROW_EVAL_FEATURE_SETS = frozenset((TE_EVAL_FEATURE_SET, BRIDGE_EVAL_FEATURE_SET))
NANO_HYBRIDEP_FEATURE_SET = "dropless_hybridep_nano16"
QWEN30_ALLTOALL_FEATURE_SET = "dropless_alltoall_qwen30_16"
SUPER_ALLTOALL_FEATURE_SET = "dropless_alltoall_super32"
QWEN235_HYBRIDEP_FEATURE_SET = "dropless_hybridep_qwen235_64"
DROPLESS_MOE_FEATURE_SETS = frozenset(
    (
        NANO_HYBRIDEP_FEATURE_SET,
        QWEN30_ALLTOALL_FEATURE_SET,
        SUPER_ALLTOALL_FEATURE_SET,
        QWEN235_HYBRIDEP_FEATURE_SET,
    )
)
HYBRIDEP_FEATURE_SETS = frozenset(
    (NANO_HYBRIDEP_FEATURE_SET, QWEN235_HYBRIDEP_FEATURE_SET)
)
ALLTOALL_FEATURE_SETS = frozenset(
    (QWEN30_ALLTOALL_FEATURE_SET, SUPER_ALLTOALL_FEATURE_SET)
)
TE_EVAL_EXCLUDED_PACKAGES = (
    "causal-conv1d",
    "deep-ep",
    "fast-hadamard-transform",
    "mamba-ssm",
)
HYBRIDEP_EXCLUDED_PACKAGES = ("fast-hadamard-transform",)
ALLTOALL_EXCLUDED_PACKAGES = ("deep-ep", "fast-hadamard-transform")
RUNTIME_FEATURE_EXCLUSIONS = {
    TE_EVAL_FEATURE_SET: TE_EVAL_EXCLUDED_PACKAGES,
    BRIDGE_EVAL_FEATURE_SET: TE_EVAL_EXCLUDED_PACKAGES,
    **{
        feature_set: HYBRIDEP_EXCLUDED_PACKAGES for feature_set in HYBRIDEP_FEATURE_SETS
    },
    **{
        feature_set: ALLTOALL_EXCLUDED_PACKAGES for feature_set in ALLTOALL_FEATURE_SETS
    },
}
TE_EVAL_OPTIONAL_PACKAGES = frozenset(("mamba_ssm", "causal_conv1d"))
MATRIX_ROWS = {
    "mcore": frozenset(
        (
            "te_eval_capability_8",
            "dropless_hybridep_nano16",
            "dropless_alltoall_qwen30_16",
            "dropless_alltoall_super32",
            "dropless_hybridep_qwen235_64",
        )
    ),
    "bridge": frozenset(("bridge_forward_only_eval_8",)),
}
MATRIX_ROW_WORLD_SIZES = {
    "mcore": {
        "te_eval_capability_8": 8,
        "dropless_hybridep_nano16": 16,
        "dropless_alltoall_qwen30_16": 16,
        "dropless_alltoall_super32": 32,
        "dropless_hybridep_qwen235_64": 64,
    },
    "bridge": {"bridge_forward_only_eval_8": 8},
}
ALLOWED_ALLOCATIONS = frozenset(
    ((1, 8, 8), (2, 4, 8), (4, 4, 16), (8, 4, 32), (16, 4, 64))
)
ROW_ID = re.compile(r"^[a-z][a-z0-9_]*$")


def _require_nvte_environment(
    *, expected_nvte_with_nccl_ep: str, environment: Mapping[str, str]
) -> None:
    actual = environment.get("NVTE_WITH_NCCL_EP")
    if actual != expected_nvte_with_nccl_ep:
        raise ValueError(
            "NVTE_WITH_NCCL_EP process environment mismatch: "
            f"expected {expected_nvte_with_nccl_ep}, got {actual!r}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version_pair(version: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)(?:\D|$)", version)
    if match is None:
        raise ValueError(f"unparseable Transformer Engine version: {version!r}")
    return int(match.group(1)), int(match.group(2))


def _require_full_commit(label: str, commit: str) -> None:
    if FULL_COMMIT.fullmatch(commit) is None:
        raise ValueError(f"{label} must be a full lowercase 40-character SHA")


def _container_identity(container: Path) -> dict[str, int]:
    if container.is_symlink():
        raise ValueError(f"immutable container must not be a symlink: {container}")
    if not container.is_file():
        raise ValueError(f"immutable container is missing: {container}")
    status = container.stat()
    return {
        "container_device": status.st_dev,
        "container_inode": status.st_ino,
        "container_size": status.st_size,
        "container_mtime_seconds": int(status.st_mtime),
        "container_ctime_seconds": int(status.st_ctime),
    }


def _require_path_within(*, label: str, path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} is outside {root}: {path}") from error


def _read_attestation(attestation: Path) -> dict[str, Any]:
    if attestation.is_symlink():
        raise ValueError(f"runtime attestation must not be a symlink: {attestation}")
    if not attestation.is_file():
        raise ValueError(f"runtime attestation is missing: {attestation}")
    try:
        payload = json.loads(attestation.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(
            f"runtime attestation is not valid JSON: {attestation}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("runtime attestation must contain a JSON object")
    return payload


def _runtime_contract_for_rows(
    *, candidate_kind: str, required_rows: tuple[str, ...]
) -> tuple[str, tuple[str, ...]]:
    """Resolve the one narrow runtime contract that can authorize these rows."""
    feature_sets: dict[tuple[str, tuple[str, ...]], str] = {
        ("mcore", (TE_EVAL_FEATURE_SET,)): TE_EVAL_FEATURE_SET,
        **{
            ("mcore", (feature_set,)): feature_set
            for feature_set in DROPLESS_MOE_FEATURE_SETS
        },
        ("bridge", (BRIDGE_EVAL_FEATURE_SET,)): BRIDGE_EVAL_FEATURE_SET,
    }
    feature_set = feature_sets.get((candidate_kind, required_rows))
    if feature_set is None:
        raise ValueError(
            "runtime attestation must authorize exactly one supported matrix row"
        )
    return feature_set, RUNTIME_FEATURE_EXCLUSIONS[feature_set]


def _expected_pytest_nodes(candidate_kind: str, row_id: str) -> tuple[str, ...]:
    """Read the exact committed pytest-node contract for one matrix row."""
    matrix_path = Path(__file__).with_name(f"{candidate_kind}_test_matrix.json")
    matrix = _read_attestation(matrix_path)
    if (
        matrix.get("schema_version") != 1
        or matrix.get("candidate_kind") != candidate_kind
    ):
        raise ValueError("test matrix identity is invalid")
    rows = matrix.get("rows")
    row = rows.get(row_id) if isinstance(rows, Mapping) else None
    nodes = row.get("pytest_nodes") if isinstance(row, Mapping) else None
    if (
        not isinstance(nodes, list)
        or not nodes
        or len(nodes) != len(set(nodes))
        or any(not isinstance(node, str) or not node for node in nodes)
    ):
        raise ValueError(f"test matrix row {row_id!r} has invalid pytest nodes")
    return tuple(nodes)


def _validate_device_bindings(
    bindings: Any,
    *,
    world_size: int,
    num_nodes: int,
    gpus_per_node: int,
) -> None:
    if (num_nodes, gpus_per_node, world_size) not in ALLOWED_ALLOCATIONS:
        raise ValueError("matrix result has an unsupported allocation/world size")
    if not isinstance(bindings, list) or len(bindings) != world_size:
        raise ValueError("matrix result device bindings must contain every global rank")
    expected_keys = {"global_rank", "node_rank", "local_rank", "cuda_device_index"}
    normalized: list[dict[str, int]] = []
    for binding in bindings:
        if not isinstance(binding, Mapping) or set(binding) != expected_keys:
            raise ValueError("matrix result device binding has an invalid schema")
        if any(
            not isinstance(binding[key], int) or isinstance(binding[key], bool)
            for key in expected_keys
        ):
            raise ValueError("matrix result device binding values must be integers")
        global_rank = binding["global_rank"]
        node_rank = binding["node_rank"]
        local_rank = binding["local_rank"]
        cuda_device_index = binding["cuda_device_index"]
        if global_rank not in range(world_size):
            raise ValueError("matrix result device binding global rank is out of range")
        if node_rank not in range(num_nodes):
            raise ValueError("matrix result device binding node rank is out of range")
        if local_rank not in range(gpus_per_node):
            raise ValueError("matrix result device binding local rank is out of range")
        if cuda_device_index != local_rank:
            raise ValueError("matrix result CUDA device does not match local rank")
        normalized.append(dict(binding))
    if sorted(binding["global_rank"] for binding in normalized) != list(
        range(world_size)
    ):
        raise ValueError("matrix result has duplicate or missing global rank bindings")
    actual_slots = {
        (binding["node_rank"], binding["local_rank"]) for binding in normalized
    }
    expected_slots = {
        (node_rank, local_rank)
        for node_rank in range(num_nodes)
        for local_rank in range(gpus_per_node)
    }
    if actual_slots != expected_slots or len(actual_slots) != world_size:
        raise ValueError("matrix result has duplicate or missing per-node device slots")
    if any(
        binding["global_rank"]
        != binding["node_rank"] * gpus_per_node + binding["local_rank"]
        for binding in normalized
    ):
        raise ValueError("matrix result global rank does not match its device slot")


def _validate_te_capability_evidence(payload: Mapping[str, Any]) -> None:
    evidence = payload.get("capability_evidence")
    expected_keys = {
        "all_eval_callables_supported",
        "backward_executed",
        "fallback_forward_counter_increment",
        "forward_invocations_after_capture",
        "no_parameter_grads",
        "outputs_changed",
        "replay_forward_counter_increment",
        "mcore_eval_reuse_graph_io",
        "raw_te_eval_reuse_graph_io",
        "raw_te_eval_reuse_rejection",
        "raw_te_eval_reuse_eager_parity",
        "raw_te_eval_reuse_fallback_forward_counter_increment",
        "raw_te_eval_reuse_no_parameter_grads",
        "raw_te_eval_reuse_outputs_changed",
        "raw_te_eval_reuse_replay_forward_counter_increment",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != expected_keys:
        raise ValueError("matrix result capability evidence has an invalid schema")
    required = {
        "all_eval_callables_supported": True,
        "backward_executed": False,
        "fallback_forward_counter_increment": 1,
        "no_parameter_grads": True,
        "outputs_changed": True,
        "replay_forward_counter_increment": 0,
        "mcore_eval_reuse_graph_io": "not_implemented",
        "raw_te_eval_reuse_no_parameter_grads": True,
    }
    if any(evidence.get(key) != value for key, value in required.items()):
        raise ValueError(
            "matrix result capability evidence does not prove safe eval replay"
        )
    forward_count = evidence.get("forward_invocations_after_capture")
    if (
        not isinstance(forward_count, int)
        or isinstance(forward_count, bool)
        or forward_count <= 0
    ):
        raise ValueError("matrix result capability evidence has an invalid counter")
    raw_reuse = evidence.get("raw_te_eval_reuse_graph_io")
    if not isinstance(raw_reuse, bool):
        raise ValueError(
            "matrix result capability evidence has an invalid reuse status"
        )
    if raw_reuse:
        accepted = {
            "raw_te_eval_reuse_rejection": None,
            "raw_te_eval_reuse_eager_parity": True,
            "raw_te_eval_reuse_fallback_forward_counter_increment": 1,
            "raw_te_eval_reuse_outputs_changed": True,
            "raw_te_eval_reuse_replay_forward_counter_increment": 0,
        }
        if any(evidence.get(key) != value for key, value in accepted.items()):
            raise ValueError(
                "matrix result capability evidence does not prove safe reuse"
            )
    else:
        rejection = evidence.get("raw_te_eval_reuse_rejection")
        unavailable = (
            "raw_te_eval_reuse_eager_parity",
            "raw_te_eval_reuse_fallback_forward_counter_increment",
            "raw_te_eval_reuse_outputs_changed",
            "raw_te_eval_reuse_replay_forward_counter_increment",
        )
        if (
            not isinstance(rejection, str)
            or not rejection.strip()
            or any(evidence.get(key) is not None for key in unavailable)
        ):
            raise ValueError(
                "matrix result capability evidence has inconsistent reuse rejection"
            )
    summaries = {
        "all_eval_callables_supported": evidence["all_eval_callables_supported"],
        "mcore_eval_reuse_graph_io": evidence["mcore_eval_reuse_graph_io"],
        "raw_te_eval_reuse_graph_io": raw_reuse,
    }
    if any(payload.get(key) != value for key, value in summaries.items()):
        raise ValueError("matrix result capability evidence disagrees with its summary")


def validate_matrix_results(
    *,
    candidate_kind: str,
    candidate_sha: str,
    integration_sha: str,
    expected_container_sha256: str,
    expected_te_commit: str,
    expected_te_version_base_commit: str,
    test_result_dir: Path,
    required_rows: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    """Require an exact set of passed, candidate-bound matrix artifacts."""
    if candidate_kind not in MATRIX_ROWS:
        raise ValueError("candidate kind must be mcore or bridge")
    for label, commit in (
        ("candidate SHA", candidate_sha),
        ("integration SHA", integration_sha),
        ("Transformer Engine source commit", expected_te_commit),
        ("Transformer Engine version-base commit", expected_te_version_base_commit),
    ):
        _require_full_commit(label, commit)
    if FULL_SHA256.fullmatch(expected_container_sha256) is None:
        raise ValueError(
            "expected container SHA256 must be 64 lowercase hexadecimal characters"
        )
    if not required_rows or len(required_rows) != len(set(required_rows)):
        raise ValueError("required rows must be non-empty and unique")
    unknown_rows = set(required_rows).difference(MATRIX_ROWS[candidate_kind])
    if unknown_rows:
        raise ValueError(f"unknown required matrix rows: {sorted(unknown_rows)}")
    if test_result_dir.is_symlink() or not test_result_dir.is_dir():
        raise ValueError(
            f"test result directory is missing or unsafe: {test_result_dir}"
        )
    candidate_dir = test_result_dir / candidate_kind / candidate_sha
    if candidate_dir.is_symlink() or not candidate_dir.is_dir():
        raise ValueError(
            f"candidate result directory is missing or unsafe: {candidate_dir}"
        )
    actual_files = {path.name for path in candidate_dir.iterdir() if path.is_file()}
    expected_files = {f"{row_id}.json" for row_id in required_rows}
    extra_files = actual_files.difference(expected_files)
    missing_files = expected_files.difference(actual_files)
    if extra_files:
        raise ValueError(f"extra matrix result files: {sorted(extra_files)}")
    if missing_files:
        raise ValueError(f"missing matrix result files: {sorted(missing_files)}")

    results: dict[str, dict[str, Any]] = {}
    for row_id in required_rows:
        result_file = candidate_dir / f"{row_id}.json"
        payload = _read_attestation(result_file)
        expected = {
            "schema_version": 1,
            "status": "passed",
            "candidate_kind": candidate_kind,
            "candidate_sha": candidate_sha,
            "integration_sha": integration_sha,
            "container_sha256": expected_container_sha256,
            "transformer_engine_source_commit": expected_te_commit,
            "transformer_engine_version_base_commit": (expected_te_version_base_commit),
            "test_row_id": row_id,
        }
        mismatches = {
            key: {"expected": value, "actual": payload.get(key)}
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "matrix result content binding mismatch: "
                + json.dumps(mismatches, sort_keys=True)
            )
        topology = payload.get("topology")
        if not isinstance(topology, Mapping):
            raise ValueError("matrix result topology must be a JSON object")
        world_size = topology.get("world_size")
        num_nodes = topology.get("num_nodes")
        gpus_per_node = topology.get("gpus_per_node")
        joined_ranks = topology.get("joined_ranks")
        if (
            not isinstance(world_size, int)
            or isinstance(world_size, bool)
            or world_size != MATRIX_ROW_WORLD_SIZES[candidate_kind][row_id]
            or not isinstance(num_nodes, int)
            or isinstance(num_nodes, bool)
            or not isinstance(gpus_per_node, int)
            or isinstance(gpus_per_node, bool)
            or joined_ranks != list(range(world_size))
        ):
            raise ValueError("matrix result does not prove every global rank joined")
        _validate_device_bindings(
            topology.get("device_bindings"),
            world_size=world_size,
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
        )
        expected_nodes = _expected_pytest_nodes(candidate_kind, row_id)
        node_results = payload.get("node_results")
        if not isinstance(node_results, list) or len(node_results) != len(
            expected_nodes
        ):
            raise ValueError("matrix result pytest node results are incomplete")
        for expected_node, node_result in zip(
            expected_nodes, node_results, strict=True
        ):
            if (
                not isinstance(node_result, Mapping)
                or set(node_result) != {"node", "status", "exit_code"}
                or node_result.get("node") != expected_node
                or node_result.get("status") != "passed"
                or node_result.get("exit_code") != 0
            ):
                raise ValueError("matrix result pytest node results are invalid")
        if not isinstance(payload.get("transformer_engine_version"), str):
            raise ValueError("matrix result lacks Transformer Engine version")
        if row_id == "te_eval_capability_8":
            _validate_te_capability_evidence(payload)
        results[row_id] = payload
    return results


def validate_attestation(
    *,
    attestation: Path,
    container: Path,
    expected_container_sha256: str,
    nemo_rl_commit: str,
    bridge_commit: str,
    mcore_commit: str,
    uv_lock: Path,
    expected_te_commit: str,
    expected_device_count: int,
    expected_python_version: str,
    expected_python_install_dir: Path,
    expected_uv_version: str,
    expected_uv_executable: Path,
    expected_nvte_with_nccl_ep: str = "0",
    expected_runtime_attestation_job_id: int | None = None,
    expected_te_version_base_commit: str | None = None,
    expected_runtime_feature_set: str | None = None,
    expected_excluded_packages: tuple[str, ...] | None = None,
    expected_torch_cuda_arch_list: str | None = None,
    expected_nvte_cuda_archs: str | None = None,
) -> dict[str, Any]:
    """Require exact source, image, TE, GPU, and worker-stack provenance."""
    if FULL_SHA256.fullmatch(expected_container_sha256) is None:
        raise ValueError(
            "expected container SHA256 must be 64 lowercase hexadecimal characters"
        )
    for label, commit in (
        ("NeMo-RL commit", nemo_rl_commit),
        ("Megatron-Bridge commit", bridge_commit),
        ("Megatron-LM commit", mcore_commit),
        ("Transformer Engine commit", expected_te_commit),
    ):
        _require_full_commit(label, commit)
    if expected_te_version_base_commit is not None:
        _require_full_commit(
            "Transformer Engine version-base commit",
            expected_te_version_base_commit,
        )
    if expected_device_count <= 0:
        raise ValueError("expected device count must be positive")
    if (
        expected_runtime_attestation_job_id is not None
        and expected_runtime_attestation_job_id <= 0
    ):
        raise ValueError("expected runtime attestation job ID must be positive")
    if re.fullmatch(r"\d+\.\d+\.\d+", expected_python_version) is None:
        raise ValueError("expected Python version must be an exact X.Y.Z version")
    if not expected_python_install_dir.is_absolute():
        raise ValueError("expected Python install directory must be absolute")
    if expected_python_install_dir.is_symlink():
        raise ValueError(
            "expected Python install directory must not be a symlink: "
            f"{expected_python_install_dir}"
        )
    expected_python_install_dir = expected_python_install_dir.resolve(strict=False)
    if not expected_python_install_dir.is_dir():
        raise ValueError(
            "expected Python install directory is missing: "
            f"{expected_python_install_dir}"
        )
    if re.fullmatch(r"\d+\.\d+\.\d+", expected_uv_version) is None:
        raise ValueError("expected uv version must be an exact X.Y.Z version")
    if expected_nvte_with_nccl_ep not in {"0", "1"}:
        raise ValueError("expected NVTE_WITH_NCCL_EP must be 0 or 1")
    if not expected_uv_executable.is_absolute():
        raise ValueError("expected uv executable must be absolute")
    if expected_uv_executable.is_symlink():
        raise ValueError(
            f"expected uv executable must not be a symlink: {expected_uv_executable}"
        )
    expected_uv_executable = expected_uv_executable.resolve(strict=False)
    if not expected_uv_executable.is_file():
        raise ValueError(f"expected uv executable is missing: {expected_uv_executable}")
    if not uv_lock.is_file():
        raise ValueError(f"uv.lock is missing: {uv_lock}")

    payload = _read_attestation(attestation)
    if payload.get("status") != "passed":
        raise ValueError("runtime attestation status is not passed")
    runtime_contract = (
        expected_runtime_feature_set,
        expected_excluded_packages,
        expected_torch_cuda_arch_list,
        expected_nvte_cuda_archs,
    )
    if any(value is not None for value in runtime_contract):
        if (
            expected_runtime_feature_set is None
            or expected_excluded_packages is None
            or expected_torch_cuda_arch_list is None
            or expected_nvte_cuda_archs is None
        ):
            raise ValueError("runtime feature contract must be provided together")
        feature_exclusions = RUNTIME_FEATURE_EXCLUSIONS.get(
            expected_runtime_feature_set
        )
        if (
            feature_exclusions is None
            or expected_excluded_packages != feature_exclusions
            or expected_torch_cuda_arch_list != "10.0a"
            or expected_nvte_cuda_archs != "100a"
        ):
            raise ValueError("unsupported runtime feature contract")
        expected_contract = runtime_contract
        actual_contract = (
            payload.get("runtime_feature_set"),
            tuple(payload.get("excluded_packages", ())),
            payload.get("torch_cuda_arch_list"),
            payload.get("nvte_cuda_archs"),
        )
        if actual_contract != expected_contract:
            raise ValueError("runtime attestation feature contract mismatch")

    expected_provenance: dict[str, object] = {
        "container_image": str(container),
        "container_sha256": expected_container_sha256,
        "nemo_rl_commit": nemo_rl_commit,
        "bridge_commit": bridge_commit,
        "mcore_commit": mcore_commit,
        "expected_te_commit": expected_te_commit,
        "transformer_engine_vcs_commit": expected_te_commit,
        "device_count": expected_device_count,
        "expected_device_count": expected_device_count,
        "expected_python_version": expected_python_version,
        "python_version": expected_python_version,
        "uv_python_install_dir": str(expected_python_install_dir),
        "expected_uv_version": expected_uv_version,
        "uv_version": expected_uv_version,
        "uv_executable": str(expected_uv_executable),
        "expected_nvte_with_nccl_ep": expected_nvte_with_nccl_ep,
        "nvte_with_nccl_ep": expected_nvte_with_nccl_ep,
        "transformer_engine_nccl_ep_available": (expected_nvte_with_nccl_ep == "1"),
        "transformer_engine_nccl_ep_symbols": (
            [
                "ep_initialize",
                "ep_finalize",
                "ep_get_zero_copy",
                "ep_handle_mem_size",
                "ep_prepare",
                "ep_dispatch",
                "ep_combine",
                "ep_dispatch_bwd",
                "ep_combine_bwd",
            ]
            if expected_nvte_with_nccl_ep == "1"
            else []
        ),
        "transformer_engine_grouped_linear_symbols": list(
            REQUIRED_TE_GROUPED_LINEAR_SYMBOLS
        ),
    }
    if expected_runtime_attestation_job_id is not None:
        expected_provenance["runtime_attestation_job_id"] = (
            expected_runtime_attestation_job_id
        )
    if expected_te_version_base_commit is not None:
        expected_provenance.update(
            {
                "transformer_engine_source_commit": expected_te_commit,
                "transformer_engine_version_base_commit": (
                    expected_te_version_base_commit
                ),
            }
        )
    mismatches = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in expected_provenance.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "runtime attestation provenance mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )

    actual_identity = _container_identity(container)
    identity_mismatches = {
        key: {"expected": payload.get(key), "actual": actual}
        for key, actual in actual_identity.items()
        if payload.get(key) != actual
    }
    if identity_mismatches:
        raise ValueError(
            "container identity mismatch after runtime preflight: "
            + json.dumps(identity_mismatches, sort_keys=True)
        )

    lock_sha256 = _sha256(uv_lock)
    if payload.get("uv_lock_sha256") != lock_sha256:
        raise ValueError(
            "uv.lock SHA256 mismatch after runtime preflight: "
            f"expected {payload.get('uv_lock_sha256')}, got {lock_sha256}"
        )

    python_base_executable_value = payload.get("python_base_executable")
    if not isinstance(python_base_executable_value, str):
        raise ValueError("runtime attestation lacks managed Python executable")
    python_base_executable = Path(python_base_executable_value)
    if not python_base_executable.is_absolute():
        raise ValueError("managed Python executable must be absolute")
    if python_base_executable.is_symlink():
        raise ValueError(
            f"managed Python executable must not be a symlink: {python_base_executable}"
        )
    python_base_executable = python_base_executable.resolve(strict=False)
    _require_path_within(
        label="managed Python executable",
        path=python_base_executable,
        root=expected_python_install_dir,
    )
    if not python_base_executable.is_file():
        raise ValueError(
            f"managed Python executable is missing: {python_base_executable}"
        )
    expected_python_sha256 = payload.get("python_base_executable_sha256")
    if (
        not isinstance(expected_python_sha256, str)
        or FULL_SHA256.fullmatch(expected_python_sha256) is None
    ):
        raise ValueError("runtime attestation lacks managed Python executable SHA256")
    actual_python_sha256 = _sha256(python_base_executable)
    if actual_python_sha256 != expected_python_sha256:
        raise ValueError(
            "managed Python executable SHA256 mismatch: "
            f"expected {expected_python_sha256}, got {actual_python_sha256}"
        )

    expected_uv_sha256 = payload.get("uv_executable_sha256")
    if (
        not isinstance(expected_uv_sha256, str)
        or FULL_SHA256.fullmatch(expected_uv_sha256) is None
    ):
        raise ValueError("runtime attestation lacks uv executable SHA256")
    actual_uv_sha256 = _sha256(expected_uv_executable)
    if actual_uv_sha256 != expected_uv_sha256:
        raise ValueError(
            "uv executable SHA256 mismatch: "
            f"expected {expected_uv_sha256}, got {actual_uv_sha256}"
        )

    packages = payload.get("packages")
    if not isinstance(packages, Mapping):
        raise ValueError("runtime attestation packages must be a JSON object")
    required_packages = REQUIRED_PACKAGES
    if expected_runtime_feature_set in NARROW_EVAL_FEATURE_SETS:
        required_packages = required_packages.difference(TE_EVAL_OPTIONAL_PACKAGES)
    elif expected_runtime_feature_set in HYBRIDEP_FEATURE_SETS:
        required_packages = required_packages.union(("deep_ep",))
        if payload.get("hybridep_buffer_available") is not True:
            raise ValueError("runtime attestation does not prove DeepEP HybridEPBuffer")
        deep_ep_commit = payload.get("deep_ep_vcs_commit")
        if not isinstance(deep_ep_commit, str) or FULL_COMMIT.fullmatch(deep_ep_commit) is None:
            raise ValueError("runtime attestation lacks a full DeepEP VCS commit")
    if expected_runtime_feature_set in DROPLESS_MOE_FEATURE_SETS:
        required_packages = required_packages.union(("vllm",))
        actor_runtimes = payload.get("actor_runtimes")
        if not isinstance(actor_runtimes, Mapping):
            raise ValueError("runtime attestation lacks actor runtimes")
        vllm_runtime = actor_runtimes.get("vllm")
        if not isinstance(vllm_runtime, Mapping):
            raise ValueError("runtime attestation lacks the vLLM actor runtime")
        expected_vllm_root = (
            expected_uv_executable.parent.parent / "vllm-environment"
        )
        expected_vllm_python = expected_vllm_root / "bin" / "python"
        if (
            vllm_runtime.get("runtime_prefix") != str(expected_vllm_root)
            or vllm_runtime.get("python_executable") != str(expected_vllm_python)
            or vllm_runtime.get("cuda_available") is not True
            or vllm_runtime.get("device_count") != expected_device_count
            or vllm_runtime.get("excluded_packages") != list(feature_exclusions)
        ):
            raise ValueError("runtime attestation vLLM actor identity mismatch")
        vllm_packages = vllm_runtime.get("packages")
        if (
            not isinstance(vllm_packages, Mapping)
            or vllm_packages.get("vllm") != packages.get("vllm")
        ):
            raise ValueError("runtime attestation vllm package identity mismatch")
        if not expected_vllm_python.is_file() or not os.access(
            expected_vllm_python, os.X_OK
        ):
            raise ValueError("attested vLLM actor Python is missing")
    missing_packages = sorted(required_packages.difference(packages))
    if missing_packages:
        raise ValueError(
            "runtime attestation is missing required packages: "
            + ", ".join(missing_packages)
        )
    te_package = packages["transformer_engine.pytorch"]
    if not isinstance(te_package, Mapping) or not isinstance(
        te_package.get("version"), str
    ):
        raise ValueError("runtime attestation lacks Transformer Engine version")
    te_version = te_package["version"]
    if _version_pair(te_version) < MINIMUM_TE_VERSION:
        raise ValueError(
            f"runtime requires Transformer Engine >= 2.16, got {te_version}"
        )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attestation", type=Path)
    parser.add_argument("--container", type=Path)
    parser.add_argument("--expected-container-sha256")
    parser.add_argument("--nemo-rl-commit")
    parser.add_argument("--bridge-commit")
    parser.add_argument("--mcore-commit")
    parser.add_argument("--uv-lock", type=Path)
    parser.add_argument("--expected-te-commit")
    parser.add_argument("--expected-te-version-base-commit")
    parser.add_argument("--expected-device-count", type=int)
    parser.add_argument("--expected-python-version")
    parser.add_argument("--expected-python-install-dir", type=Path)
    parser.add_argument("--expected-uv-version")
    parser.add_argument("--expected-uv-executable", type=Path)
    parser.add_argument("--expected-nvte-with-nccl-ep")
    parser.add_argument("--expected-runtime-attestation-job-id", type=int)
    parser.add_argument("--runtime-feature-set")
    parser.add_argument("--excluded-packages")
    parser.add_argument("--torch-cuda-arch-list")
    parser.add_argument("--nvte-cuda-archs")
    parser.add_argument("--profile-file", type=Path)
    parser.add_argument("--candidate-kind", choices=("mcore", "bridge"))
    parser.add_argument("--candidate-sha")
    parser.add_argument("--test-result-dir", type=Path)
    parser.add_argument("--required-rows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.profile_file is not None:
        matrix_required = (
            args.candidate_kind,
            args.candidate_sha,
            args.test_result_dir,
            args.required_rows,
        )
        if any(value is None for value in matrix_required):
            raise ValueError(
                "matrix mode requires candidate kind/SHA, test result directory, and required rows"
            )
        values: dict[str, str] = {}
        for number, line in enumerate(args.profile_file.read_text().splitlines(), 1):
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError(f"profile line {number} is not a literal assignment")
            name, value = line.split("=", 1)
            if name in values:
                raise ValueError(f"profile line {number} duplicates {name}")
            values[name] = value
        runtime_path = Path(values["RUNTIME_ATTESTATION"])
        runtime_payload = _read_attestation(runtime_path)
        _require_nvte_environment(
            expected_nvte_with_nccl_ep=runtime_payload["expected_nvte_with_nccl_ep"],
            environment=os.environ,
        )
        repository_root = Path(__file__).resolve().parents[3]
        required_rows = tuple(args.required_rows.split())
        runtime_feature_set, runtime_exclusions = _runtime_contract_for_rows(
            candidate_kind=args.candidate_kind,
            required_rows=required_rows,
        )
        validate_attestation(
            attestation=runtime_path,
            container=Path(values["CONTAINER"]),
            expected_container_sha256=values["CONTAINER_SHA256"],
            nemo_rl_commit=values["EXPECTED_NEMORL_SHA"],
            bridge_commit=values["EXPECTED_BRIDGE_SHA"],
            mcore_commit=values["EXPECTED_MCORE_SHA"],
            uv_lock=repository_root / "uv.lock",
            expected_te_commit=values["EXPECTED_TE_SHA"],
            expected_te_version_base_commit=values["EXPECTED_TE_VERSION_BASE_SHA"],
            expected_device_count=int(values["SBATCH_GPUS_PER_NODE"]),
            expected_python_version=runtime_payload["expected_python_version"],
            expected_python_install_dir=Path(runtime_payload["uv_python_install_dir"]),
            expected_uv_version=runtime_payload["expected_uv_version"],
            expected_uv_executable=Path(values["UV_EXECUTABLE"]),
            expected_nvte_with_nccl_ep=runtime_payload["expected_nvte_with_nccl_ep"],
            expected_runtime_attestation_job_id=int(values["RUNTIME_PREFLIGHT_JOB_ID"]),
            expected_runtime_feature_set=runtime_feature_set,
            expected_excluded_packages=runtime_exclusions,
            expected_torch_cuda_arch_list="10.0a",
            expected_nvte_cuda_archs="100a",
        )
        integration_sha = (
            values["EXPECTED_MCORE_SHA"]
            if args.candidate_kind == "mcore"
            else values["EXPECTED_BRIDGE_SHA"]
        )
        results = validate_matrix_results(
            candidate_kind=args.candidate_kind,
            candidate_sha=args.candidate_sha,
            integration_sha=integration_sha,
            expected_container_sha256=values["CONTAINER_SHA256"],
            expected_te_commit=values["EXPECTED_TE_SHA"],
            expected_te_version_base_commit=values["EXPECTED_TE_VERSION_BASE_SHA"],
            test_result_dir=args.test_result_dir,
            required_rows=required_rows,
        )
        print(json.dumps(results, sort_keys=True))
        return

    legacy_required = (
        args.attestation,
        args.container,
        args.expected_container_sha256,
        args.nemo_rl_commit,
        args.bridge_commit,
        args.mcore_commit,
        args.uv_lock,
        args.expected_te_commit,
        args.expected_device_count,
        args.expected_python_version,
        args.expected_python_install_dir,
        args.expected_uv_version,
        args.expected_uv_executable,
        args.expected_nvte_with_nccl_ep,
        args.expected_runtime_attestation_job_id,
    )
    if any(value is None for value in legacy_required):
        raise ValueError("legacy mode requires every runtime-attestation argument")
    _require_nvte_environment(
        expected_nvte_with_nccl_ep=args.expected_nvte_with_nccl_ep,
        environment=os.environ,
    )
    runtime_contract = (
        args.runtime_feature_set,
        args.excluded_packages,
        args.torch_cuda_arch_list,
        args.nvte_cuda_archs,
    )
    if any(value is not None for value in runtime_contract) and any(
        value is None for value in runtime_contract
    ):
        raise ValueError("runtime feature contract arguments must be provided together")
    payload = validate_attestation(
        attestation=args.attestation,
        container=args.container,
        expected_container_sha256=args.expected_container_sha256,
        nemo_rl_commit=args.nemo_rl_commit,
        bridge_commit=args.bridge_commit,
        mcore_commit=args.mcore_commit,
        uv_lock=args.uv_lock,
        expected_te_commit=args.expected_te_commit,
        expected_device_count=args.expected_device_count,
        expected_python_version=args.expected_python_version,
        expected_python_install_dir=args.expected_python_install_dir,
        expected_uv_version=args.expected_uv_version,
        expected_uv_executable=args.expected_uv_executable,
        expected_nvte_with_nccl_ep=args.expected_nvte_with_nccl_ep,
        expected_runtime_attestation_job_id=(args.expected_runtime_attestation_job_id),
        expected_runtime_feature_set=args.runtime_feature_set,
        expected_excluded_packages=(
            tuple(args.excluded_packages.split(","))
            if args.excluded_packages is not None
            else None
        ),
        expected_torch_cuda_arch_list=args.torch_cuda_arch_list,
        expected_nvte_cuda_archs=args.nvte_cuda_archs,
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
