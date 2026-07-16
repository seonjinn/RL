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

"""Launch isolated Qwen3-32B DynamicSD boundary profiles on Lyris."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from experiments.vllm_0251_drafter_matrix.profile_dynamic_sd import (
    CONTAINER_PYTHON,
    DATASET_REPO_ID,
    DATASET_REVISION,
    DEFAULT_CONTAINER,
    DEFAULT_HF_HOME,
    DEFAULT_MOUNTS,
    DRAFTER_REPO_ID,
    DRAFTER_REVISION,
    PASSTHROUGH_ENVIRONMENT,
    SECRET_ENVIRONMENT,
    TARGET_REPO_ID,
    TARGET_REVISION,
    WORKER_RELATIVE_PATH,
    _parse_job_id,
    _profile_contract,
    _validate_runtime_inputs,
    snapshot_path,
    write_manifest,
)


G_BOUNDARY_CELLS = (
    (34, 3),
    (34, 5),
    (35, 3),
    (35, 5),
    (75, 2),
    (75, 3),
    (76, 2),
    (76, 3),
    (85, 1),
    (85, 2),
    (86, 1),
    (86, 2),
)
G_MANIFEST_NAME = "dynamic-sd-boundary-launch-manifest.json"


@dataclass(frozen=True, slots=True)
class BoundaryCell:
    """One independently scheduled batch-size and draft-width profile cell."""

    batch_size: int
    k: int
    job_name: str
    output_dir: Path


def build_cells(profile_root: Path) -> tuple[BoundaryCell, ...]:
    """Build the exact boundary-validation cells in deterministic order."""
    return tuple(
        BoundaryCell(
            batch_size=batch_size,
            k=k,
            job_name=f"coreai_dlalgo_llm-dynamicsd.bs{batch_size}-k{k}",
            output_dir=profile_root / "boundary-jobs" / f"bs{batch_size}-k{k}",
        )
        for batch_size, k in G_BOUNDARY_CELLS
    )


def _cell_detail(cell: BoundaryCell) -> str:
    return f"bs{cell.batch_size}-k{cell.k}"


def _require_known_cell(cell: BoundaryCell) -> None:
    if (cell.batch_size, cell.k) not in G_BOUNDARY_CELLS:
        raise ValueError(f"Unknown DynamicSD boundary cell: {cell}")
    expected_name = f"coreai_dlalgo_llm-dynamicsd.{_cell_detail(cell)}"
    if cell.job_name != expected_name:
        raise ValueError(f"Unexpected boundary job name: {cell.job_name!r}")


def _profile_venv_root(cell: BoundaryCell) -> Path:
    return Path(f"/tmp/nemorl-v0251-qwen32-dynamicsd-{_cell_detail(cell)}")


def _profile_python(cell: BoundaryCell) -> Path:
    return _profile_venv_root(cell) / "profile" / "bin" / "python"


def build_runtime_command(
    cell: BoundaryCell,
    *,
    repo_dir: Path,
    profile_root: Path,
    target_snapshot: Path,
    drafter_snapshot: Path,
    prompt_template: Path,
) -> tuple[str, ...]:
    """Build the exact one-cell worker command executed by ``ray.sub``."""
    _require_known_cell(cell)
    if target_snapshot.name != TARGET_REVISION:
        raise ValueError("Target snapshot must use the immutable Qwen3-32B revision")
    if drafter_snapshot.name != DRAFTER_REVISION:
        raise ValueError("Drafter snapshot must use the immutable matched revision")
    runtime_id = f"qwen32-dynamicsd-{_cell_detail(cell)}"
    return (
        "env",
        "CUDA_VISIBLE_DEVICES=0,1",
        "VLLM_USE_V2_MODEL_RUNNER=1",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        f"PYTHONPATH={repo_dir}",
        f"TRITON_CACHE_DIR=/tmp/nemorl-v0251-triton-{runtime_id}",
        f"TORCHINDUCTOR_CACHE_DIR=/tmp/nemorl-v0251-inductor-{runtime_id}",
        "PYTHONFAULTHANDLER=1",
        str(_profile_python(cell)),
        str(repo_dir / WORKER_RELATIVE_PATH),
        "run-k",
        "--root",
        str(profile_root),
        "--k",
        str(cell.k),
        "--target-snapshot",
        str(target_snapshot),
        "--drafter-snapshot",
        str(drafter_snapshot),
        "--prompt-template",
        str(prompt_template),
        "--port",
        "8100",
        "--batch-sizes",
        str(cell.batch_size),
    )


def build_sbatch_command(
    cell: BoundaryCell,
    *,
    repo_dir: Path,
    mode: Literal["test-only", "submit"] | str,
) -> tuple[str, ...]:
    """Build one exact Lyris preflight or submission command."""
    _require_known_cell(cell)
    if mode == "test-only":
        mode_flag = "--test-only"
    elif mode == "submit":
        mode_flag = "--parsable"
    else:
        raise ValueError(f"Unsupported scheduler mode: {mode}")
    return (
        "sbatch",
        mode_flag,
        "--dependency=",
        "--account=coreai_dlalgo_llm",
        "--partition=gb200",
        "--nodes=1",
        "--ntasks-per-node=1",
        "--exclusive",
        "--time=05:00:00",
        "--segment=1",
        f"--job-name={cell.job_name}",
        f"--output={cell.output_dir / 'slurm-%j.out'}",
        str(repo_dir / "ray.sub"),
    )


def build_venv_setup_command(cell: BoundaryCell, repo_dir: Path) -> tuple[str, ...]:
    """Materialize the checkout's locked vLLM environment for one cell."""
    _require_known_cell(cell)
    setup = (
        "from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES;"
        "from nemo_rl.utils.venvs import create_local_venv;"
        "create_local_venv(PY_EXECUTABLES.VLLM, 'profile')"
    )
    return (
        "env",
        f"PYTHONPATH={repo_dir}",
        f"NEMO_RL_VENV_DIR={_profile_venv_root(cell)}",
        CONTAINER_PYTHON,
        "-c",
        setup,
    )


def _boundary_profile_contract() -> dict[str, object]:
    contract = _profile_contract()
    contract["batch_sizes"] = sorted({batch_size for batch_size, _ in G_BOUNDARY_CELLS})
    return contract


def _result_path(profile_root: Path, cell: BoundaryCell) -> Path:
    return profile_root / f"k-{cell.k}" / f"bs-{cell.batch_size}" / "result.json"


def _manifest_payload(
    *,
    status: str,
    cells: Sequence[BoundaryCell],
    repo_dir: Path,
    profile_root: Path,
    target_snapshot: Path,
    drafter_snapshot: Path,
    prompt_template: Path,
    hf_home: Path,
    container: Path,
    mounts: str,
    job_ids: Mapping[str, str | None],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": status,
        "repo_dir": str(repo_dir),
        "profile_root": str(profile_root),
        "prompt_jsonl": str(profile_root / "prompts.jsonl"),
        "prompt_template": str(prompt_template),
        "hf_home": str(hf_home),
        "container": str(container),
        "mounts": mounts,
        "target": {
            "repo_id": TARGET_REPO_ID,
            "revision": TARGET_REVISION,
            "snapshot": str(target_snapshot),
        },
        "drafter": {
            "repo_id": DRAFTER_REPO_ID,
            "revision": DRAFTER_REVISION,
            "snapshot": str(drafter_snapshot),
        },
        "dataset": {
            "repo_id": DATASET_REPO_ID,
            "revision": DATASET_REVISION,
        },
        "profile_contract": _boundary_profile_contract(),
        "cells": [
            {
                "cell_id": _cell_detail(cell),
                "batch_size": cell.batch_size,
                "k": cell.k,
                "job_name": cell.job_name,
                "output_dir": str(cell.output_dir),
                "result_path": str(_result_path(profile_root, cell)),
                "job_id": job_ids.get(_cell_detail(cell)),
                "runtime_command": list(
                    build_runtime_command(
                        cell,
                        repo_dir=repo_dir,
                        profile_root=profile_root,
                        target_snapshot=target_snapshot,
                        drafter_snapshot=drafter_snapshot,
                        prompt_template=prompt_template,
                    )
                ),
                "venv_setup_command": list(build_venv_setup_command(cell, repo_dir)),
                "preflight_command": list(
                    build_sbatch_command(cell, repo_dir=repo_dir, mode="test-only")
                ),
                "submission_command": list(
                    build_sbatch_command(cell, repo_dir=repo_dir, mode="submit")
                ),
            }
            for cell in cells
        ],
    }


def _submission_environment(
    *,
    cell: BoundaryCell,
    repo_dir: Path,
    runtime_command: Sequence[str],
    container: Path,
    mounts: str,
    hf_home: Path,
) -> dict[str, str]:
    environment = {
        name: os.environ[name]
        for name in PASSTHROUGH_ENVIRONMENT
        if name in os.environ and name not in SECRET_ENVIRONMENT
    }
    environment.update(
        {
            "BASE_LOG_DIR": str(cell.output_dir),
            "COMMAND": shlex.join(runtime_command),
            "CONTAINER": str(container),
            "GPUS_PER_NODE": "4",
            "HF_HOME": str(hf_home),
            "MOUNTS": mounts,
            "SETUP_COMMAND": shlex.join(build_venv_setup_command(cell, repo_dir)),
        }
    )
    if SECRET_ENVIRONMENT.intersection(environment):
        raise RuntimeError("Submission environment contains a forbidden secret")
    return environment


def _run_sbatch(
    command: tuple[str, ...],
    *,
    repo_dir: Path,
    environment: Mapping[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=repo_dir,
        env=dict(environment),
        check=True,
        capture_output=True,
        text=True,
    )


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--container", type=Path, default=DEFAULT_CONTAINER)
    parser.add_argument("--mounts", default=DEFAULT_MOUNTS)
    parser.add_argument(
        "--prompt-template",
        type=Path,
        default=Path("examples/prompts/cot.txt"),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the boundary launcher command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("show", "test-only", "submit"):
        _add_arguments(subparsers.add_parser(action))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Show, preflight, or submit all boundary-validation cells."""
    args = build_parser().parse_args(argv)
    repo_dir = args.repo_dir.resolve()
    profile_root = args.output_dir.resolve()
    hf_home = args.hf_home.resolve()
    container = args.container.resolve()
    prompt_template = args.prompt_template
    if not prompt_template.is_absolute():
        prompt_template = repo_dir / prompt_template
    prompt_template = prompt_template.resolve()
    target_snapshot = snapshot_path(hf_home, TARGET_REPO_ID, TARGET_REVISION)
    drafter_snapshot = snapshot_path(hf_home, DRAFTER_REPO_ID, DRAFTER_REVISION)
    cells = build_cells(profile_root)
    job_ids: dict[str, str | None] = {_cell_detail(cell): None for cell in cells}
    manifest_path = profile_root / G_MANIFEST_NAME

    def persist(status: str) -> dict[str, Any]:
        payload = _manifest_payload(
            status=status,
            cells=cells,
            repo_dir=repo_dir,
            profile_root=profile_root,
            target_snapshot=target_snapshot,
            drafter_snapshot=drafter_snapshot,
            prompt_template=prompt_template,
            hf_home=hf_home,
            container=container,
            mounts=args.mounts,
            job_ids=job_ids,
        )
        if args.action != "show":
            write_manifest(manifest_path, payload)
        return payload

    if args.action == "show":
        print(json.dumps(persist("planned"), indent=2, sort_keys=True))
        return 0

    _validate_runtime_inputs(
        repo_dir=repo_dir,
        profile_root=profile_root,
        hf_home=hf_home,
        target_snapshot=target_snapshot,
        drafter_snapshot=drafter_snapshot,
        prompt_template=prompt_template,
        container=container,
        mounts=args.mounts,
    )
    profile_root.mkdir(parents=True, exist_ok=True)
    for cell in cells:
        cell.output_dir.mkdir(parents=True, exist_ok=True)
    persist("preflighting")

    runtime_commands = {
        _cell_detail(cell): build_runtime_command(
            cell,
            repo_dir=repo_dir,
            profile_root=profile_root,
            target_snapshot=target_snapshot,
            drafter_snapshot=drafter_snapshot,
            prompt_template=prompt_template,
        )
        for cell in cells
    }
    environments = {
        _cell_detail(cell): _submission_environment(
            cell=cell,
            repo_dir=repo_dir,
            runtime_command=runtime_commands[_cell_detail(cell)],
            container=container,
            mounts=args.mounts,
            hf_home=hf_home,
        )
        for cell in cells
    }
    try:
        for cell in cells:
            _run_sbatch(
                build_sbatch_command(cell, repo_dir=repo_dir, mode="test-only"),
                repo_dir=repo_dir,
                environment=environments[_cell_detail(cell)],
            )
    except (OSError, subprocess.CalledProcessError):
        persist("preflight-failed")
        raise
    if args.action == "test-only":
        persist("test-only")
        print(manifest_path)
        return 0

    persist("preflight-passed")
    try:
        for cell in cells:
            result = _run_sbatch(
                build_sbatch_command(cell, repo_dir=repo_dir, mode="submit"),
                repo_dir=repo_dir,
                environment=environments[_cell_detail(cell)],
            )
            job_ids[_cell_detail(cell)] = _parse_job_id(result.stdout)
            persist("submitting")
    except (OSError, subprocess.CalledProcessError, RuntimeError):
        persist("submission-failed")
        raise
    persist("submitted")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
