# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provenance and bounded per-node log collection for the GB200 sleep gate."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tarfile
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CACHE_VARIABLES = (
    "HF_HOME",
    "HF_DATASETS_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "HF_MODULES_CACHE",
    "UV_CACHE_DIR",
    "XDG_CACHE_HOME",
    "TORCH_HOME",
    "TORCH_EXTENSIONS_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "VLLM_CACHE_ROOT",
    "CUDA_CACHE_PATH",
    "TMPDIR",
    "RAY_TMPDIR",
)


def _environment() -> dict[str, str]:
    # Deliberately exclude credentials such as HF_TOKEN / WANDB_API_KEY.
    return {
        key: value
        for key, value in os.environ.items()
        if not re.search(r"API_KEY|(?:^|_)TOKEN$|SECRET|PASSWORD|CREDENTIAL", key)
        and (
            key in CACHE_VARIABLES
            or key.startswith(
                (
                    "NRL_",
                    "RAY_",
                    "UV_",
                    "NCCL_",
                    "VLLM_",
                    "PYTORCH_",
                    "NEMO_",
                    "TORCH_",
                    "CUDA_",
                    "CUBLAS_",
                    "CUDNN_",
                    "FLASHINFER_",
                    "UCX_",
                    "NVSHMEM_",
                    "GLOO_",
                    "PYTHON",
                    "NVLINK_",
                    "NUM_OF_",
                    "USE_MNNVL",
                )
            )
        )
    }


def _node_record() -> dict[str, Any]:
    for variable in CACHE_VARIABLES:
        path = Path(os.environ[variable])
        assert path.is_relative_to("/raid/scratch"), (variable, str(path))
        path.mkdir(parents=True, exist_ok=True)
    devices = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
        timeout=30,
    ).splitlines()
    assert len(devices) == 4 and all("GB200" in device for device in devices), devices
    return {
        "hostname": socket.gethostname(),
        "devices": devices,
        "environment": _environment(),
    }


def verify_gb200_nodes(*, expected_nodes: int) -> None:
    import ray  # Keep the provenance writer usable before Ray is initialized.
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    nodes = [
        node
        for node in ray.nodes()
        if node["Alive"] and node["Resources"].get("GPU", 0) > 0
    ]
    assert len(nodes) == expected_nodes and all(
        node["Resources"]["GPU"] == 4 for node in nodes
    )
    task = ray.remote(num_cpus=0)(_node_record)
    refs = [
        task.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node["NodeID"], soft=False
            )
        ).remote()
        for node in nodes
    ]
    records = ray.get(refs, timeout=90)
    (Path(os.environ["NRL_REFIT_SLEEP_RUN_DIR"]) / "nodes.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )


def record_provenance() -> None:
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    assert sha == os.environ["NRL_REFIT_SLEEP_EXPECTED_SHA"], "unexpected source SHA"
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    )
    assert not status, f"GB200 gate requires committed source: {status}"
    digest = os.environ["NRL_REFIT_SLEEP_IMAGE_DIGEST"]
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", digest), (
        "an immutable image digest is required"
    )
    image = os.environ["NRL_REFIT_SLEEP_IMAGE"]
    image_path = Path(image)
    if image_path.is_file():
        with image_path.open("rb") as stream:
            assert (
                "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest() == digest
            )
    else:
        assert image.endswith("@" + digest), "registry image must be digest-pinned"
    provenance = {
        "source_sha": sha,
        "image": image,
        "image_digest": digest,
        "image_identity_source": "launcher declaration (file bytes verified for local images)",
        "python": sys.version,
        "executable": sys.executable,
        "packages": {
            dist.metadata["Name"]: dist.version
            for dist in importlib.metadata.distributions()
        },
        "environment": _environment(),
        "submodules": subprocess.check_output(
            ["git", "submodule", "status"], cwd=ROOT, text=True
        ),
    }
    (Path(os.environ["NRL_REFIT_SLEEP_RUN_DIR"]) / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


def _archive_node_logs(destination: str) -> str:
    import ray  # Only the post-pytest collection path needs Ray internals.

    node = ray._private.worker._global_node
    logs = Path(node.get_logs_dir_path())
    target = Path(destination) / f"ray-{socket.gethostname()}.tar.gz"
    with tarfile.open(target, "w:gz") as archive:
        archive.add(logs, arcname="logs")
    return str(target)


def archive_node_logs() -> None:
    import ray  # Do not start Ray during CPU collection/provenance writing.
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from nemo_rl.distributed.virtual_cluster import init_ray

    init_ray()
    task = ray.remote(num_cpus=0)(_archive_node_logs)
    refs = [
        task.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node["NodeID"], soft=False
            )
        ).remote(os.environ["NRL_REFIT_SLEEP_ARTIFACT_DIR"])
        for node in ray.nodes()
        if node["Alive"] and node["Resources"].get("GPU", 0) > 0
    ]
    assert refs, "no Ray nodes available for log collection"
    print(json.dumps(ray.get(refs, timeout=120)))


if __name__ == "__main__":
    {"record": record_provenance, "archive": archive_node_logs}[sys.argv[1]]()
