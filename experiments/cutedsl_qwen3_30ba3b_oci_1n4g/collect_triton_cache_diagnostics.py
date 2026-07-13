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

"""Collect bounded, sanitized diagnostics for node-local Triton caches."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MAX_DIAGNOSTIC_FILES = 256
MAX_DIAGNOSTIC_BYTES = 1_048_576
SUMMARY_KEYS = frozenset(
    {
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
)
RECORD_KEYS = frozenset(
    {
        "relative_name_sha256",
        "file_type",
        "size",
        "inode",
        "mtime_ns",
        "json_valid",
        "prefix_sha256",
        "bytes_read",
    }
)
CACHE_SCOPES = frozenset({"job_node_local", "run_local_container"})


@dataclass(frozen=True)
class DiagnosticLimits:
    """Hard limits on cache files and file-prefix bytes collected per node."""

    max_files: int = MAX_DIAGNOSTIC_FILES
    max_total_bytes: int = MAX_DIAGNOSTIC_BYTES


def _finite_nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a finite nonnegative integer")
    return value


def _validate_limits(node_index: int, limits: DiagnosticLimits) -> None:
    _finite_nonnegative_integer(node_index, "node_index")
    if (
        isinstance(limits.max_files, bool)
        or not isinstance(limits.max_files, int)
        or not 1 <= limits.max_files <= MAX_DIAGNOSTIC_FILES
    ):
        raise ValueError("max_files must be between 1 and 256")
    if (
        isinstance(limits.max_total_bytes, bool)
        or not isinstance(limits.max_total_bytes, int)
        or not 1 <= limits.max_total_bytes <= MAX_DIAGNOSTIC_BYTES
    ):
        raise ValueError("max_total_bytes must be between 1 and 1048576")


def _is_candidate(path: Path) -> bool:
    return path.suffix == ".json" or path.name.startswith("__grp__")


def _environment_nonnegative_integer(name: str, default: str | None = None) -> int:
    raw_value = os.environ.get(name, default)
    if raw_value is None or re.fullmatch(r"[0-9]+", raw_value) is None:
        raise ValueError(f"{name} must be a finite nonnegative integer")
    return int(raw_value)


def collect_cache_diagnostics(
    root: Path,
    node_index: int,
    limits: DiagnosticLimits,
) -> dict[str, Any]:
    """Collect bounded cache metadata and hashed content prefixes."""
    _validate_limits(node_index, limits)
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"cache root is not a directory: {root}")

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
    partial_read = False
    for path in candidates[: limits.max_files]:
        remaining = limits.max_total_bytes - total_bytes_read
        if remaining <= 0:
            break
        with path.open("rb") as stream:
            payload = stream.read(remaining)
        total_bytes_read += len(payload)
        stat = path.stat(follow_symlinks=False)
        partial_read = partial_read or len(payload) < stat.st_size
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

    try:
        triton_version = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        triton_version = "unavailable"

    result: dict[str, Any] = {
        "schema_version": 1,
        "node_index": node_index,
        "job_id": os.environ.get("SLURM_JOB_ID", "synthetic"),
        "restart_count": _environment_nonnegative_integer("SLURM_RESTART_COUNT", "0"),
        "slurm_procid": _environment_nonnegative_integer(
            "SLURM_PROCID", str(node_index)
        ),
        "cache_scope": os.environ.get("NEMO2606_TRITON_CACHE_SCOPE", "job_node_local"),
        "triton_version": triton_version,
        "candidate_count": len(candidates),
        "scanned_count": len(files),
        "rejected_symlink_count": rejected_symlink_count,
        "total_bytes_read": total_bytes_read,
        "truncated": len(files) < len(candidates) or partial_read,
        "files": files,
    }
    _validate_summary_schema(result)
    return result


def _validate_summary_schema(value: dict[str, Any]) -> None:
    if (
        set(value) != SUMMARY_KEYS
        or isinstance(value.get("schema_version"), bool)
        or value.get("schema_version") != 1
    ):
        raise ValueError("invalid node-summary schema")
    files = value["files"]
    if not isinstance(files, list) or len(files) > MAX_DIAGNOSTIC_FILES:
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
    cache_scope = value["cache_scope"]
    if not isinstance(cache_scope, str) or cache_scope not in CACHE_SCOPES:
        raise ValueError("invalid cache scope")
    job_id = value.get("job_id")
    if (
        not isinstance(job_id, str)
        or re.fullmatch(r"(?:[0-9]+|synthetic)", job_id) is None
    ):
        raise ValueError("invalid job identity")
    triton_version = value.get("triton_version")
    if (
        not isinstance(triton_version, str)
        or re.fullmatch(r"[A-Za-z0-9_.+-]{1,64}", triton_version) is None
    ):
        raise ValueError("invalid Triton version")
    if not isinstance(value["truncated"], bool):
        raise ValueError("truncated must be boolean")

    total = 0
    for record in files:
        if not isinstance(record, dict) or set(record) != RECORD_KEYS:
            raise ValueError("invalid diagnostic record schema")
        if record.get("file_type") != "regular":
            raise ValueError("diagnostic record must describe a regular file")
        for digest in ("relative_name_sha256", "prefix_sha256"):
            digest_value = record.get(digest)
            if (
                not isinstance(digest_value, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest_value) is None
            ):
                raise ValueError(f"invalid {digest}")
        for label in ("size", "inode", "mtime_ns", "bytes_read"):
            _finite_nonnegative_integer(record.get(label), label)
        if not isinstance(record.get("json_valid"), bool):
            raise ValueError("json_valid must be boolean")
        total += record["bytes_read"]
    if total > MAX_DIAGNOSTIC_BYTES or total != value["total_bytes_read"]:
        raise ValueError("diagnostic byte total is invalid")


def merge_cache_diagnostics(summary_dir: Path, expected_nodes: int) -> dict[str, Any]:
    """Validate and merge per-node diagnostic summaries."""
    if (
        isinstance(expected_nodes, bool)
        or not isinstance(expected_nodes, int)
        or expected_nodes < 1
    ):
        raise ValueError("expected_nodes must be positive")
    root = summary_dir.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"summary root is not a directory: {root}")

    nodes: dict[int, dict[str, Any]] = {}
    for path in sorted(summary_dir.glob("node-*.json")):
        if path.is_symlink():
            raise ValueError("node summary must not be a symlink")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError("node summary must be a contained regular file")
        loaded = json.loads(resolved.read_text())
        if not isinstance(loaded, dict):
            raise ValueError("invalid node-summary schema")
        value: dict[str, Any] = loaded
        node_index = _finite_nonnegative_integer(value.get("node_index"), "node_index")
        if node_index in nodes:
            raise ValueError(f"duplicate node_index: {node_index}")
        _validate_summary_schema(value)
        if node_index >= expected_nodes:
            raise ValueError("node_index is outside expected range")
        nodes[node_index] = value
    missing_nodes = sorted(set(range(expected_nodes)) - set(nodes))
    return {
        "schema_version": 1,
        "expected_nodes": expected_nodes,
        "observed_nodes": sorted(nodes),
        "missing_nodes": missing_nodes,
        "timed_out": bool(missing_nodes),
        "truncated": any(value["truncated"] for value in nodes.values()),
        "nodes": [nodes[index] for index in sorted(nodes)],
    }


def _write_json_atomic(output: Path, value: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, allow_nan=False, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, output)


def _slurm_paths() -> tuple[Path, Path]:
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is None or re.fullmatch(r"[0-9]+", job_id) is None:
        raise ValueError("SLURM_JOB_ID must contain only decimal digits")
    restart = os.environ.get("SLURM_RESTART_COUNT")
    if restart:
        _environment_nonnegative_integer("SLURM_RESTART_COUNT")
    user = os.environ.get("USER")
    if user is None or re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", user) is None:
        raise ValueError("USER must be a safe path component")
    result_root = os.environ.get("CUTEDSL_BENCHMARK_RESULT_ROOT")
    if result_root is None or not result_root:
        raise ValueError("CUTEDSL_BENCHMARK_RESULT_ROOT is required")
    run_id = job_id + (f"-r{restart}" if restart else "")
    cache_root = Path("/tmp") / user / "nemo2606-factorial" / run_id / "triton_cache"
    output_dir = Path(result_root) / run_id / "triton_cache_diagnostics"
    return cache_root, output_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--from-slurm-env", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Collect one node summary from explicit paths or the SLURM environment."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.from_slurm_env:
        if args.cache_root is not None or args.output_dir is not None:
            parser.error("--from-slurm-env cannot be combined with explicit paths")
        cache_root, output_dir = _slurm_paths()
    else:
        if args.cache_root is None or args.output_dir is None:
            parser.error("--cache-root and --output-dir are required together")
        cache_root = args.cache_root
        output_dir = args.output_dir

    if args.from_slurm_env and os.environ.get("FAILURE_DIAGNOSTIC_MERGE") == "1":
        expected_nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
        output_dir.mkdir(parents=True, exist_ok=True)
        result = merge_cache_diagnostics(output_dir, expected_nodes)
        _write_json_atomic(output_dir / "summary.json", result)
        return 0

    node_index = _environment_nonnegative_integer("FAILURE_DIAGNOSTIC_NODE_INDEX")
    result = collect_cache_diagnostics(
        cache_root,
        node_index=node_index,
        limits=DiagnosticLimits(),
    )
    _write_json_atomic(output_dir / f"node-{node_index}.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
