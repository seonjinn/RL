#!/usr/bin/env python3
"""Render or submit the dependency-gated vLLM 0.28 smoke matrix."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


_PACKAGE_ROOT = Path(__file__).resolve().parent
_CONTAINER_SOURCE = (
    "registry-1.docker.io/vllm/vllm-openai@sha256:"
    "41b54fb42c66a670a8b27e613ebef05898f24b9ab1bdab28bd00c877bd4935f4"
)
_RESULT_ROOT = Path(
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-benchmark-results/"
    "vllm028-nemotron-bf16-smoke"
)


def _load_sibling(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _PACKAGE_ROOT / f"{name}.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _smoke_rows() -> list[dict[str, Any]]:
    contract = _load_sibling("contract").build_contract_matrix()
    launcher = _load_sibling("launcher")
    plan = launcher.build_submission_plan(
        contract,
        data_parallel_size=1,
        container_image=_CONTAINER_SOURCE,
    )
    selected = launcher.select_smoke_plan_rows(plan)
    k0_canaries = [
        row
        for row in plan
        if row["method_key"] == "mtp_dynamic_max_k5" and row["batch_size"] == 512
    ]
    return selected + k0_canaries


def _result_dir(row: dict[str, Any]) -> Path:
    shape = row["shape"]
    return (
        _RESULT_ROOT
        / str(row["model_key"])
        / f"isl{shape['isl']}_osl{shape['osl']}"
        / str(row["runner_key"])
        / str(row["method_key"])
        / f"bs{row['batch_size']}"
    )


def render_dry_run(*, sbatch_runner: Callable[..., Any]) -> list[str]:
    """Render every command without touching the scheduler."""
    del sbatch_runner
    launcher = _load_sibling("launcher")
    rendered = [
        json.dumps(
            {
                "job": "stage_vllm028_container",
                "script": str(_PACKAGE_ROOT / "stage_vllm028_container.sbatch"),
            },
            sort_keys=True,
        ),
        json.dumps(
            {
                "job": "stage_super_bf16_checkpoint",
                "dependency": "afterok:stage_vllm028_container",
                "script": str(_PACKAGE_ROOT / "stage_super_bf16_checkpoint.sbatch"),
            },
            sort_keys=True,
        ),
    ]
    for row in _smoke_rows():
        rendered.append(
            launcher.render_sbatch(
                row,
                experiment_dir=_PACKAGE_ROOT,
                result_dir=_result_dir(row),
            )
        )
    return rendered


def _command_runner(*args: str) -> str:
    completed = subprocess.run(
        list(args),
        check=True,
        capture_output=True,
        text=True,
    )
    output = completed.stdout.strip()
    match = re.search(r"(\d+)(?:;[^\s]+)?$", output)
    if match is None:
        raise RuntimeError(f"unable to parse sbatch output: {output}")
    return match.group(1)


def build_stage_submission_plan(
    *,
    dry_run: bool,
    sbatch_runner: Callable[..., Any],
) -> list[dict[str, Any]]:
    """Submit container and checkpoint staging with an absolute Ray lock path."""
    lock_path = (_PACKAGE_ROOT / "ray248-aarch64.lock").resolve()
    container_argv = [
        "sbatch",
        f"--export=ALL,RAY_LOCK_PATH={lock_path}",
        str(_PACKAGE_ROOT / "stage_vllm028_container.sbatch"),
    ]
    checkpoint_template = [
        "sbatch",
        "--dependency=afterok:stage_vllm028_container",
        str(_PACKAGE_ROOT / "stage_super_bf16_checkpoint.sbatch"),
    ]
    if dry_run:
        return [{"argv": container_argv}, {"argv": checkpoint_template}]
    container_job_id = str(sbatch_runner(*container_argv)).strip().split(";", 1)[0]
    if not container_job_id.isdigit():
        raise RuntimeError(f"unable to parse container stage job id: {container_job_id}")
    checkpoint_argv = [
        "sbatch",
        f"--dependency=afterok:{container_job_id}",
        str(_PACKAGE_ROOT / "stage_super_bf16_checkpoint.sbatch"),
    ]
    checkpoint_job_id = str(sbatch_runner(*checkpoint_argv)).strip().split(";", 1)[0]
    if not checkpoint_job_id.isdigit():
        raise RuntimeError(f"unable to parse checkpoint stage job id: {checkpoint_job_id}")
    return [
        {"argv": container_argv, "job_id": container_job_id},
        {"argv": checkpoint_argv, "job_id": checkpoint_job_id},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--submit-staging", action="store_true")
    parsed = parser.parse_args()
    if parsed.dry_run == parsed.submit_staging:
        raise SystemExit(
            "Choose exactly one of --dry-run or --submit-staging; direct benchmark "
            "bulk submission is intentionally disabled."
        )
    if parsed.submit_staging:
        print(
            json.dumps(
                build_stage_submission_plan(
                    dry_run=False,
                    sbatch_runner=_command_runner,
                ),
                indent=2,
            )
        )
        return
    print("\n---\n".join(render_dry_run(sbatch_runner=_command_runner)))


if __name__ == "__main__":
    main()
