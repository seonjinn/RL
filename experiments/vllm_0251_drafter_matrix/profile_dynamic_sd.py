# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Launch the matched Qwen3-32B DynamicSD GPU profile on Lyris."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


TARGET_REPO_ID = "Qwen/Qwen3-32B"
TARGET_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
DRAFTER_REPO_ID = "RedHatAI/Qwen3-32B-Thinking-speculator.eagle3"
DRAFTER_REVISION = "a1403e07b73a66fc9ef561463631c31864616933"
DATASET_REPO_ID = "nvidia/OpenMathInstruct-2"
DATASET_REVISION = "469216e3f46f4dacf476b382e192485ea51a143e"
BATCH_SIZES = (1, 4, 16, 32, 64, 128, 192, 256)
K_VALUES = tuple(range(6))
MANIFEST_NAME = "dynamic-profile-launch-manifest.json"

DEFAULT_HF_HOME = Path("/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home")
DEFAULT_CONTAINER = Path(
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260715.sqsh"
)
DEFAULT_MOUNTS = "/lustre:/lustre"
CONTAINER_PYTHON = "/opt/nemo_rl_venv/bin/python"
WORKER_RELATIVE_PATH = Path(
    "experiments/vllm_0251_drafter_matrix/dynamic_profile_worker.py"
)
PASSTHROUGH_ENVIRONMENT = (
    "HOME",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOGNAME",
    "NO_PROXY",
    "PATH",
    "SHELL",
    "SLURM_CONF",
    "TMPDIR",
    "USER",
    "http_proxy",
    "https_proxy",
    "no_proxy",
)
SECRET_ENVIRONMENT = frozenset(
    {
        "AWS_SECRET_ACCESS_KEY",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "WANDB_API_KEY",
    }
)


@dataclass(frozen=True, slots=True)
class ProfileJob:
    """One independent fixed-K profiling allocation."""

    k: int
    job_name: str
    output_dir: Path


def snapshot_path(hf_home: Path, repo_id: str, revision: str) -> Path:
    """Return the immutable Hugging Face snapshot path for a repository."""
    return (
        hf_home
        / "hub"
        / f"models--{repo_id.replace('/', '--')}"
        / "snapshots"
        / revision
    )


def build_jobs(profile_root: Path) -> tuple[ProfileJob, ...]:
    """Build the complete K0-K5 job set in deterministic order."""
    return tuple(
        ProfileJob(
            k=k,
            job_name=f"nemorl-qwen32-dynamicsd-k{k}",
            output_dir=profile_root / f"k-{k}",
        )
        for k in K_VALUES
    )


def _profile_venv_root(job: ProfileJob) -> Path:
    return Path(f"/tmp/nemorl-v0251-qwen32-dynamicsd-k{job.k}")


def _profile_python(job: ProfileJob) -> Path:
    return _profile_venv_root(job) / "profile" / "bin" / "python"


def _require_known_job(job: ProfileJob) -> None:
    if job.k not in K_VALUES:
        raise ValueError(f"K must be one of {K_VALUES}, got {job.k}")
    expected_name = f"nemorl-qwen32-dynamicsd-k{job.k}"
    if job.job_name != expected_name:
        raise ValueError(f"Unexpected profile job name: {job.job_name!r}")


def build_runtime_command(
    job: ProfileJob,
    *,
    repo_dir: Path,
    profile_root: Path,
    target_snapshot: Path,
    drafter_snapshot: Path,
    prompt_template: Path,
) -> tuple[str, ...]:
    """Build the exact command executed by ``ray.sub`` on the job's head node."""
    _require_known_job(job)
    if target_snapshot.name != TARGET_REVISION:
        raise ValueError("Target snapshot must use the immutable Qwen3-32B revision")
    if drafter_snapshot.name != DRAFTER_REVISION:
        raise ValueError("Drafter snapshot must use the immutable matched revision")
    runtime_id = f"qwen32-dynamicsd-k{job.k}"
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
        str(_profile_python(job)),
        str(repo_dir / WORKER_RELATIVE_PATH),
        "run-k",
        "--root",
        str(profile_root),
        "--k",
        str(job.k),
        "--target-snapshot",
        str(target_snapshot),
        "--drafter-snapshot",
        str(drafter_snapshot),
        "--prompt-template",
        str(prompt_template),
        "--port",
        "8100",
    )


def build_venv_setup_command(job: ProfileJob, repo_dir: Path) -> tuple[str, ...]:
    """Materialize this checkout's locked vLLM extra outside the base venv."""
    setup = (
        "from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES;"
        "from nemo_rl.utils.venvs import create_local_venv;"
        "create_local_venv(PY_EXECUTABLES.VLLM, 'profile')"
    )
    return (
        "env",
        f"PYTHONPATH={repo_dir}",
        f"NEMO_RL_VENV_DIR={_profile_venv_root(job)}",
        CONTAINER_PYTHON,
        "-c",
        setup,
    )


def build_sbatch_command(
    job: ProfileJob,
    *,
    repo_dir: Path,
    mode: Literal["test-only", "submit"] | str,
) -> tuple[str, ...]:
    """Build one bounded Lyris ``ray.sub`` preflight or submission command."""
    _require_known_job(job)
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
        f"--job-name={job.job_name}",
        f"--output={job.output_dir / 'slurm-%j.out'}",
        str(repo_dir / "ray.sub"),
    )


def _profile_contract() -> dict[str, object]:
    return {
        "batch_sizes": list(BATCH_SIZES),
        "chunked_prefill": True,
        "cuda_graph_mode": "FULL_AND_PIECEWISE",
        "dataset_revision": DATASET_REVISION,
        "draft_tensor_parallel_size": 1,
        "max_model_len": 4096,
        "max_num_batched_tokens": 16384,
        "max_num_seqs": 256,
        "num_prompts_per_batch_size": "batch_size * 20",
        "output_len": 256,
        "prefix_cache": False,
        "runtime_vllm": "0.25.1",
        "target_tensor_parallel_size": 2,
        "temperature": 1.0,
        "top_p": 1.0,
        "vllm_runner": "MRv2",
    }


def _manifest_payload(
    *,
    status: str,
    jobs: Sequence[ProfileJob],
    repo_dir: Path,
    profile_root: Path,
    target_snapshot: Path,
    drafter_snapshot: Path,
    prompt_template: Path,
    job_ids: Mapping[int, str | None],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": status,
        "repo_dir": str(repo_dir),
        "profile_root": str(profile_root),
        "prompt_jsonl": str(profile_root / "prompts.jsonl"),
        "prompt_template": str(prompt_template),
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
        "profile_contract": _profile_contract(),
        "jobs": [
            {
                **asdict(job),
                "output_dir": str(job.output_dir),
                "job_id": job_ids.get(job.k),
                "runtime_command": list(
                    build_runtime_command(
                        job,
                        repo_dir=repo_dir,
                        profile_root=profile_root,
                        target_snapshot=target_snapshot,
                        drafter_snapshot=drafter_snapshot,
                        prompt_template=prompt_template,
                    )
                ),
                "preflight_command": list(
                    build_sbatch_command(job, repo_dir=repo_dir, mode="test-only")
                ),
                "submission_command": list(
                    build_sbatch_command(job, repo_dir=repo_dir, mode="submit")
                ),
            }
            for job in jobs
        ],
    }


def _assert_secret_free(payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(payload, sort_keys=True)
    leaked_names = sorted(name for name in SECRET_ENVIRONMENT if name in serialized)
    if leaked_names:
        raise ValueError(
            f"Manifest contains forbidden environment names: {leaked_names}"
        )


def write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically persist a machine-readable, secret-free launch manifest."""
    _assert_secret_free(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _parse_mount_roots(mounts: str) -> tuple[Path, ...]:
    roots: list[Path] = []
    for mount in mounts.split(","):
        fields = mount.split(":")
        source = Path(fields[0]) if fields and fields[0] else None
        destination = Path(fields[1]) if len(fields) > 1 and fields[1] else source
        flags = frozenset(
            flag for field in fields[2:] for flag in field.split("+") if flag
        )
        if source is None or not source.is_absolute():
            raise ValueError(f"Container mount source must be absolute: {mount!r}")
        if destination != source or "ro" in flags:
            raise ValueError(
                f"Profile paths require writable identity container mounts: {mount!r}"
            )
        roots.append(source.resolve())
    if not roots:
        raise ValueError("At least one container mount is required")
    return tuple(roots)


def _require_visible(path: Path, roots: Sequence[Path], label: str) -> None:
    resolved = path.resolve()
    if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
        raise ValueError(f"{label} is not visible through the container mounts: {path}")


def _validate_vllm_pin(repo_dir: Path) -> None:
    pyproject_path = repo_dir / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        pyproject = tomllib.load(handle)
    dependencies = (
        pyproject.get("project", {}).get("optional-dependencies", {}).get("vllm", [])
    )
    vllm_dependencies = [
        dependency
        for dependency in dependencies
        if isinstance(dependency, str) and dependency.startswith("vllm")
    ]
    if not vllm_dependencies or any(
        "0.25.1" not in dependency for dependency in vllm_dependencies
    ):
        raise RuntimeError("Current pyproject must pin every vLLM dependency to 0.25.1")


def _validate_runtime_inputs(
    *,
    repo_dir: Path,
    profile_root: Path,
    hf_home: Path,
    target_snapshot: Path,
    drafter_snapshot: Path,
    prompt_template: Path,
    container: Path,
    mounts: str,
) -> None:
    required_files = (
        (repo_dir / "ray.sub", "ray.sub"),
        (repo_dir / WORKER_RELATIVE_PATH, "profile worker"),
        (repo_dir / "pyproject.toml", "pyproject"),
        (prompt_template, "prompt template"),
        (container, "container"),
    )
    for path, label in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    for snapshot, label, revision in (
        (target_snapshot, "target", TARGET_REVISION),
        (drafter_snapshot, "drafter", DRAFTER_REVISION),
    ):
        if snapshot.name != revision or not (snapshot / "config.json").is_file():
            raise FileNotFoundError(f"Missing immutable {label} snapshot: {snapshot}")
    dataset_snapshot = (
        hf_home
        / "hub"
        / f"datasets--{DATASET_REPO_ID.replace('/', '--')}"
        / "snapshots"
        / DATASET_REVISION
    )
    if not dataset_snapshot.is_dir():
        raise FileNotFoundError(
            f"Missing immutable OpenMathInstruct-2 snapshot: {dataset_snapshot}"
        )
    _validate_vllm_pin(repo_dir)
    roots = _parse_mount_roots(mounts)
    for path, label in (
        (repo_dir, "repository"),
        (profile_root, "profile output"),
        (hf_home, "HF home"),
        (prompt_template, "prompt template"),
    ):
        _require_visible(path, roots, label)


def _submission_environment(
    *,
    job: ProfileJob,
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
            "BASE_LOG_DIR": str(job.output_dir),
            "COMMAND": shlex.join(runtime_command),
            "CONTAINER": str(container),
            "GPUS_PER_NODE": "4",
            "HF_HOME": str(hf_home),
            "MOUNTS": mounts,
            "SETUP_COMMAND": shlex.join(build_venv_setup_command(job, repo_dir)),
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


def _parse_job_id(stdout: str) -> str:
    job_id = stdout.strip().split(";", maxsplit=1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"sbatch --parsable returned an invalid job ID: {stdout!r}")
    return job_id


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
    """Build the typed launcher command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("show", "test-only", "submit"):
        _add_arguments(subparsers.add_parser(action))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Show, preflight, or submit all six fixed-K profile jobs."""
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
    jobs = build_jobs(profile_root)
    job_ids: dict[int, str | None] = {job.k: None for job in jobs}
    manifest_path = profile_root / MANIFEST_NAME

    def persist(status: str) -> dict[str, Any]:
        payload = _manifest_payload(
            status=status,
            jobs=jobs,
            repo_dir=repo_dir,
            profile_root=profile_root,
            target_snapshot=target_snapshot,
            drafter_snapshot=drafter_snapshot,
            prompt_template=prompt_template,
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
    for job in jobs:
        job.output_dir.mkdir(parents=True, exist_ok=True)
    persist("preflighting")

    runtime_commands = {
        job.k: build_runtime_command(
            job,
            repo_dir=repo_dir,
            profile_root=profile_root,
            target_snapshot=target_snapshot,
            drafter_snapshot=drafter_snapshot,
            prompt_template=prompt_template,
        )
        for job in jobs
    }
    environments = {
        job.k: _submission_environment(
            job=job,
            repo_dir=repo_dir,
            runtime_command=runtime_commands[job.k],
            container=container,
            mounts=args.mounts,
            hf_home=hf_home,
        )
        for job in jobs
    }
    try:
        for job in jobs:
            _run_sbatch(
                build_sbatch_command(job, repo_dir=repo_dir, mode="test-only"),
                repo_dir=repo_dir,
                environment=environments[job.k],
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
        for job in jobs:
            result = _run_sbatch(
                build_sbatch_command(job, repo_dir=repo_dir, mode="submit"),
                repo_dir=repo_dir,
                environment=environments[job.k],
            )
            job_ids[job.k] = _parse_job_id(result.stdout)
            persist("submitting")
    except (OSError, subprocess.CalledProcessError, RuntimeError):
        persist("submission-failed")
        raise
    persist("submitted")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
