# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Launch matched DynamicSD GPU profiles for controlled Qwen recipes."""

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


DATASET_REPO_ID = "nvidia/OpenMathInstruct-2"
DATASET_REVISION = "469216e3f46f4dacf476b382e192485ea51a143e"
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
DYNAMIC_PATCHER_RELATIVE_PATH = Path(
    "experiments/vllm_0251_eagle3_perfcfg/apply_vllm0251_dynamic_sd_cg_fix.py"
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
class ProfileSpec:
    """Immutable model and topology contract for one DynamicSD profile."""

    key: str
    target_repo_id: str
    target_revision: str
    drafter_repo_id: str
    drafter_revision: str
    batch_sizes: tuple[int, ...]
    k_values: tuple[int, ...]
    nodes: int
    segment: int
    target_tensor_parallel_size: int
    max_model_len: int
    max_num_batched_tokens: int
    max_num_seqs: int
    profile_max_batch_size: int
    gpu_memory_utilization: float
    enable_prefix_caching: bool
    moe_backend: str | None = None
    visible_devices: str | None = None
    cudagraph_capture_sizes: tuple[int, ...] = ()


G_QWEN32_PROFILE = ProfileSpec(
    key="qwen32",
    target_repo_id="Qwen/Qwen3-32B",
    target_revision="9216db5781bf21249d130ec9da846c4624c16137",
    drafter_repo_id="RedHatAI/Qwen3-32B-Thinking-speculator.eagle3",
    drafter_revision="a1403e07b73a66fc9ef561463631c31864616933",
    batch_sizes=(1, 4, 16, 32, 64, 128, 192, 256),
    k_values=tuple(range(6)),
    nodes=1,
    segment=1,
    target_tensor_parallel_size=2,
    max_model_len=4096,
    max_num_batched_tokens=16384,
    max_num_seqs=256,
    profile_max_batch_size=256,
    gpu_memory_utilization=0.6,
    enable_prefix_caching=False,
    visible_devices="0,1",
)
G_QWEN235_PROFILE = ProfileSpec(
    key="qwen235",
    target_repo_id="Qwen/Qwen3-235B-A22B",
    target_revision="8efa61729e24bd65b1d152b5ab5409052aa80e65",
    drafter_repo_id=("RedHatAI/Qwen3-235B-A22B-Thinking-2507-speculator.eagle3"),
    drafter_revision="3c0c5cbad8e1fa7ce9e6fb6a1b0a35458b124e87",
    batch_sizes=(1, 4, 8, 16, 32, 48, 64),
    k_values=tuple(range(4)),
    nodes=2,
    segment=2,
    target_tensor_parallel_size=8,
    max_model_len=8192,
    max_num_batched_tokens=2048,
    max_num_seqs=128,
    profile_max_batch_size=64,
    gpu_memory_utilization=0.4,
    enable_prefix_caching=True,
    moe_backend="triton",
    cudagraph_capture_sizes=(1, 2, 4, 8, 16, 32, 64, 128, 192, 256),
)
G_PROFILE_SPECS = (G_QWEN32_PROFILE, G_QWEN235_PROFILE)

# Preserve the original Qwen3-32B public constants for existing callers.
TARGET_REPO_ID = G_QWEN32_PROFILE.target_repo_id
TARGET_REVISION = G_QWEN32_PROFILE.target_revision
DRAFTER_REPO_ID = G_QWEN32_PROFILE.drafter_repo_id
DRAFTER_REVISION = G_QWEN32_PROFILE.drafter_revision
BATCH_SIZES = G_QWEN32_PROFILE.batch_sizes
K_VALUES = G_QWEN32_PROFILE.k_values


@dataclass(frozen=True, slots=True)
class ProfileJob:
    """One independent fixed-K profiling allocation."""

    k: int
    job_name: str
    output_dir: Path


def get_profile_spec(model_key: str) -> ProfileSpec:
    """Return the immutable profile contract for one supported model."""
    for profile in G_PROFILE_SPECS:
        if profile.key == model_key:
            return profile
    raise ValueError(f"Unsupported DynamicSD profile model: {model_key}")


def snapshot_path(hf_home: Path, repo_id: str, revision: str) -> Path:
    """Return the immutable Hugging Face snapshot path for a repository."""
    return (
        hf_home
        / "hub"
        / f"models--{repo_id.replace('/', '--')}"
        / "snapshots"
        / revision
    )


def build_jobs(
    profile_root: Path,
    *,
    profile: ProfileSpec = G_QWEN32_PROFILE,
) -> tuple[ProfileJob, ...]:
    """Build the complete fixed-K job set in deterministic order."""
    return tuple(
        ProfileJob(
            k=k,
            job_name=f"nemorl-{profile.key}-dynamicsd-k{k}",
            output_dir=profile_root / f"k-{k}",
        )
        for k in profile.k_values
    )


def _profile_venv_root(job: ProfileJob) -> Path:
    runtime_id = job.job_name.removeprefix("nemorl-")
    return Path(f"/tmp/nemorl-v0251-{runtime_id}")


def _profile_python(job: ProfileJob) -> Path:
    return _profile_venv_root(job) / "profile" / "bin" / "python"


def _require_known_job(job: ProfileJob, profile: ProfileSpec) -> None:
    if job.k not in profile.k_values:
        raise ValueError(f"K must be one of {profile.k_values}, got {job.k}")
    expected_name = f"nemorl-{profile.key}-dynamicsd-k{job.k}"
    if job.job_name != expected_name:
        raise ValueError(f"Unexpected profile job name: {job.job_name!r}")


def build_runtime_command(
    job: ProfileJob,
    *,
    profile: ProfileSpec = G_QWEN32_PROFILE,
    repo_dir: Path,
    profile_root: Path,
    target_snapshot: Path,
    drafter_snapshot: Path,
    prompt_template: Path,
) -> tuple[str, ...]:
    """Build the exact command executed by ``ray.sub`` on the job's head node."""
    _require_known_job(job, profile)
    if target_snapshot.name != profile.target_revision:
        raise ValueError("Target snapshot must use the immutable profile revision")
    if drafter_snapshot.name != profile.drafter_revision:
        raise ValueError("Drafter snapshot must use the immutable matched revision")
    runtime_id = f"{profile.key}-dynamicsd-k{job.k}"
    visible_devices = (
        (f"CUDA_VISIBLE_DEVICES={profile.visible_devices}",)
        if profile.visible_devices is not None
        else ()
    )
    distributed_backend = (
        ("--distributed-executor-backend", "ray") if profile.nodes > 1 else ()
    )
    capture_sizes = (
        (
            "--cudagraph-capture-sizes",
            *(str(size) for size in profile.cudagraph_capture_sizes),
        )
        if profile.cudagraph_capture_sizes
        else ()
    )
    return (
        "env",
        *visible_devices,
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
        "--model-key",
        profile.key,
        "--target-tp",
        str(profile.target_tensor_parallel_size),
        "--max-k",
        str(profile.k_values[-1]),
        "--max-model-len",
        str(profile.max_model_len),
        "--max-num-batched-tokens",
        str(profile.max_num_batched_tokens),
        "--max-num-seqs",
        str(profile.max_num_seqs),
        "--profile-max-batch-size",
        str(profile.profile_max_batch_size),
        "--gpu-memory-utilization",
        str(profile.gpu_memory_utilization),
        "--enable-prefix-caching"
        if profile.enable_prefix_caching
        else "--no-enable-prefix-caching",
        "--served-model-name",
        f"{profile.key}-profile",
        "--batch-sizes",
        *(str(size) for size in profile.batch_sizes),
        *distributed_backend,
        *(("--moe-backend", profile.moe_backend) if profile.moe_backend else ()),
        *capture_sizes,
    )


def build_venv_setup_command(job: ProfileJob, repo_dir: Path) -> tuple[str, ...]:
    """Materialize this checkout's locked vLLM extra outside the base venv."""
    patcher = repo_dir / DYNAMIC_PATCHER_RELATIVE_PATH
    setup = (
        "import os,subprocess;"
        "from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES;"
        "from nemo_rl.utils.venvs import create_local_venv;"
        "create_local_venv(PY_EXECUTABLES.VLLM, 'profile');"
        f"subprocess.run([{str(_profile_python(job))!r},{str(patcher)!r}],"
        "check=True,env={**os.environ,'NRL_VLLM_DYNAMIC_SD_SMOKE_TELEMETRY':'1'})"
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
    profile: ProfileSpec = G_QWEN32_PROFILE,
    repo_dir: Path,
    mode: Literal["test-only", "submit"] | str,
) -> tuple[str, ...]:
    """Build one bounded Lyris ``ray.sub`` preflight or submission command."""
    _require_known_job(job, profile)
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
        f"--nodes={profile.nodes}",
        "--ntasks-per-node=1",
        "--exclusive",
        "--time=05:00:00",
        f"--segment={profile.segment}",
        f"--job-name={job.job_name}",
        f"--output={job.output_dir / 'slurm-%j.out'}",
        str(repo_dir / "ray.sub"),
    )


def _profile_contract(profile: ProfileSpec = G_QWEN32_PROFILE) -> dict[str, object]:
    return {
        "batch_sizes": list(profile.batch_sizes),
        "chunked_prefill": True,
        "cuda_graph_mode": "FULL_AND_PIECEWISE",
        "dataset_revision": DATASET_REVISION,
        "cudagraph_capture_sizes": list(profile.cudagraph_capture_sizes),
        "draft_tensor_parallel_size": 1,
        "k_values": list(profile.k_values),
        "max_model_len": profile.max_model_len,
        "max_num_batched_tokens": profile.max_num_batched_tokens,
        "max_num_seqs": profile.max_num_seqs,
        "profile_max_batch_size": profile.profile_max_batch_size,
        "num_prompts_per_batch_size": "batch_size * 20",
        "output_len": 256,
        "prefix_cache": profile.enable_prefix_caching,
        "moe_backend": profile.moe_backend,
        "runtime_vllm": "0.25.1",
        "target_tensor_parallel_size": profile.target_tensor_parallel_size,
        "temperature": 1.0,
        "top_p": 1.0,
        "vllm_runner": "MRv2",
    }


def _manifest_payload(
    *,
    status: str,
    profile: ProfileSpec = G_QWEN32_PROFILE,
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
            "repo_id": profile.target_repo_id,
            "revision": profile.target_revision,
            "snapshot": str(target_snapshot),
        },
        "drafter": {
            "repo_id": profile.drafter_repo_id,
            "revision": profile.drafter_revision,
            "snapshot": str(drafter_snapshot),
        },
        "profile_contract": _profile_contract(profile),
        "jobs": [
            {
                **asdict(job),
                "output_dir": str(job.output_dir),
                "job_id": job_ids.get(job.k),
                "runtime_command": list(
                    build_runtime_command(
                        job,
                        profile=profile,
                        repo_dir=repo_dir,
                        profile_root=profile_root,
                        target_snapshot=target_snapshot,
                        drafter_snapshot=drafter_snapshot,
                        prompt_template=prompt_template,
                    )
                ),
                "preflight_command": list(
                    build_sbatch_command(
                        job,
                        profile=profile,
                        repo_dir=repo_dir,
                        mode="test-only",
                    )
                ),
                "submission_command": list(
                    build_sbatch_command(
                        job,
                        profile=profile,
                        repo_dir=repo_dir,
                        mode="submit",
                    )
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
    profile: ProfileSpec = G_QWEN32_PROFILE,
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
        (repo_dir / DYNAMIC_PATCHER_RELATIVE_PATH, "DynamicSD runtime patcher"),
        (repo_dir / "pyproject.toml", "pyproject"),
        (prompt_template, "prompt template"),
        (container, "container"),
    )
    for path, label in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    for snapshot, label, revision in (
        (target_snapshot, "target", profile.target_revision),
        (drafter_snapshot, "drafter", profile.drafter_revision),
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
    parser.add_argument(
        "--model",
        choices=tuple(profile.key for profile in G_PROFILE_SPECS),
        default=G_QWEN32_PROFILE.key,
    )
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
    """Show, preflight, or submit one model's complete fixed-K profile."""
    args = build_parser().parse_args(argv)
    profile = get_profile_spec(args.model)
    repo_dir = args.repo_dir.resolve()
    profile_root = args.output_dir.resolve()
    hf_home = args.hf_home.resolve()
    container = args.container.resolve()
    prompt_template = args.prompt_template
    if not prompt_template.is_absolute():
        prompt_template = repo_dir / prompt_template
    prompt_template = prompt_template.resolve()
    target_snapshot = snapshot_path(
        hf_home, profile.target_repo_id, profile.target_revision
    )
    drafter_snapshot = snapshot_path(
        hf_home, profile.drafter_repo_id, profile.drafter_revision
    )
    jobs = build_jobs(profile_root, profile=profile)
    job_ids: dict[int, str | None] = {job.k: None for job in jobs}
    manifest_path = profile_root / MANIFEST_NAME

    def persist(status: str) -> dict[str, Any]:
        payload = _manifest_payload(
            status=status,
            profile=profile,
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
        profile=profile,
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
            profile=profile,
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
                build_sbatch_command(
                    job,
                    profile=profile,
                    repo_dir=repo_dir,
                    mode="test-only",
                ),
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
                build_sbatch_command(
                    job,
                    profile=profile,
                    repo_dir=repo_dir,
                    mode="submit",
                ),
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
