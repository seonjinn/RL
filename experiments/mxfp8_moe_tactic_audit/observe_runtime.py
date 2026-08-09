#!/usr/bin/env python3
"""Observe validation runtime identity without copying manifest expectations."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


CHUNK_BYTES = 1024 * 1024


def sha256_path(path: Path) -> str:
    """Hash a file or deterministic directory tree using bounded reads."""
    if path.is_file():
        digest = sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not path.is_dir():
        raise ValueError(f"runtime fingerprint path does not exist: {path}")
    digest = sha256()
    members = sorted(
        (member for member in path.rglob("*") if member.is_file()),
        key=lambda member: member.relative_to(path).as_posix(),
    )
    for member in members:
        relative = "./" + member.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(sha256_path(member).encode("ascii") + b"\0")
    return digest.hexdigest()


def _positive_integer(environment: Mapping[str, str], name: str) -> int:
    try:
        value = int(environment[name])
    except (KeyError, ValueError) as error:
        raise ValueError(f"runtime environment has invalid {name}") from error
    if value <= 0:
        raise ValueError(f"runtime environment has invalid {name}")
    return value


def _positive_integer_from_any(
    environment: Mapping[str, str], names: Sequence[str]
) -> int:
    for name in names:
        if name in environment:
            return _positive_integer(environment, name)
    joined_names = " or ".join(names)
    raise ValueError(f"runtime environment is missing {joined_names}")


def observe_runtime(
    *,
    nemo_rl_root: Path,
    vllm_root: Path,
    model_snapshot: Path,
    container: Path,
    cache_root: Path,
    environment: Mapping[str, str],
    torch_module: Any,
    flashinfer_module: Any,
) -> dict[str, str]:
    """Collect independently observed runtime and artifact fingerprints."""
    node_count = _positive_integer_from_any(
        environment,
        ("MXFP8_MOE_NODE_COUNT", "SLURM_JOB_NUM_NODES", "SLURM_NNODES"),
    )
    gpu_count = int(torch_module.cuda.device_count())
    if gpu_count <= 0:
        raise ValueError("runtime CUDA device count must be positive")
    tp_size = _positive_integer(environment, "VLLM_TENSOR_PARALLEL_SIZE")
    ep_size = _positive_integer(environment, "VLLM_EXPERT_PARALLEL_SIZE")
    graph_mode = environment.get("MXFP8_MOE_CUDA_GRAPH_REPLAY")
    if graph_mode not in {"required", "disabled"}:
        raise ValueError("runtime CUDA Graph mode is missing or invalid")
    cache_file = cache_root / "autotune_configs.json"
    cache_file_sha256 = sha256_path(cache_file)
    return {
        "cache_file_sha256": cache_file_sha256,
        "cache_sha256": cache_file_sha256,
        "container_sha256": sha256_path(container),
        "cuda_graph_mode": graph_mode,
        "cuda_version": str(torch_module.version.cuda),
        "dp_size": str(node_count * gpu_count),
        "ep_size": str(ep_size),
        "flashinfer_version": str(flashinfer_module.__version__),
        "gpu_name": str(torch_module.cuda.get_device_name(0)),
        "gpus_per_node": str(gpu_count),
        "model_revision": model_snapshot.resolve().name,
        "nemo_rl_commit": subprocess.check_output(
            ["git", "-C", str(nemo_rl_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "node_count": str(node_count),
        "tp_size": str(tp_size),
        "vllm_commit": subprocess.check_output(
            ["git", "-C", str(vllm_root), "rev-parse", "HEAD"], text=True
        ).strip(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Print one canonical runtime fingerprint object."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo-rl-root", type=Path, required=True)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--container", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args(argv)
    torch_module = importlib.import_module("torch")
    flashinfer_module = importlib.import_module("flashinfer")
    payload = observe_runtime(
        nemo_rl_root=args.nemo_rl_root,
        vllm_root=args.vllm_root,
        model_snapshot=args.model_snapshot,
        container=args.container,
        cache_root=args.cache_root,
        environment=os.environ,
        torch_module=torch_module,
        flashinfer_module=flashinfer_module,
    )
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
