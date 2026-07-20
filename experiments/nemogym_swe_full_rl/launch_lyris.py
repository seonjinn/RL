#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from gym_openhands_tmux import apply_gym_openhands_tmux_fix


DEFAULT_REMOTE_REPO = Path(
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-wt-nemogym-swe-full-rl"
)
DEFAULT_DATASET = Path(
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/experiments/"
    "nemogym_swe_full_rl/datasets/val-mini3.jsonl"
)
DEFAULT_CONTAINER = Path(
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260715.sqsh"
)
DEFAULT_HF_HOME = Path("/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home")
DEFAULT_RUN_ROOT = Path(
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/experiments/nemogym_swe_full_rl/runs"
)
DEFAULT_GYM_VENV_ROOT = Path(
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/experiments/nemogym_swe_full_rl/gym_venvs"
)
DFLASH_SNAPSHOT = DEFAULT_HF_HOME / (
    "hub/models--RedHatAI--Qwen3-30B-A3B-speculator.dflash/"
    "snapshots/edcff83783141eb9383e2bd6c33610d9a3104288"
)
WANDB_PROJECT = "nemo-rl-vllm0251-swe-full-grpo"
INCOMPATIBLE_INHERITED_ENVIRONMENT = (
    "CONDA_PREFIX",
    "CONDA_PREFIX_1",
    "CONDA_DEFAULT_ENV",
    "CONDA_PYTHON_EXE",
    "CONDA_EXE",
    "_CONDA_EXE",
    "CONDA_ROOT",
    "_CONDA_ROOT",
    "CONDA_SHLVL",
    "CONDA_PROMPT_MODIFIER",
    "_CE_M",
    "_CE_CONDA",
    "VIRTUAL_ENV",
)


@dataclass(frozen=True)
class Variant:
    name: str
    speculative_tokens: int | None
    capture_sizes: tuple[int, ...]


@dataclass(frozen=True)
class RunPlan:
    variant: str
    run_tag: str
    run_dir: str
    config: str
    dataset: str
    container: str
    gym_venv_dir: str
    wandb_project: str
    wandb_run_name: str
    command: tuple[str, ...]
    overrides: tuple[str, ...]
    sbatch_args: tuple[str, ...]
    submission_unset_environment: tuple[str, ...]


VARIANTS = {
    "baseline": Variant("baseline", None, ()),
    "dflash_k7": Variant("dflash_k7", 7, (8, 16, 32, 64, 128)),
    "dflash_k9": Variant("dflash_k9", 9, (10, 20, 40, 80, 160)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch Qwen3-30B-A3B full NeMo-Gym SWE GRPO on Lyris"
    )
    parser.add_argument(
        "--mode", choices=("dry-run", "test-only", "submit"), required=True
    )
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--repo-dir", type=Path, default=DEFAULT_REMOTE_REPO)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--container", type=Path, default=DEFAULT_CONTAINER)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--run-tag")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--num-prompts", type=int, default=2)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--time-limit", default="05:00:00")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    if args.num_prompts < 1:
        parser.error("num-prompts must be positive")
    if args.num_generations < 2:
        parser.error("num-generations must be at least 2 for GRPO")
    return args


def _capture_sizes_override(capture_sizes: tuple[int, ...]) -> str:
    values = ",".join(str(value) for value in capture_sizes)
    return (
        "++policy.generation.vllm_kwargs.compilation_config."
        f"cudagraph_capture_sizes=[{values}]"
    )


def build_plan(args: argparse.Namespace) -> RunPlan:
    variant = VARIANTS[args.variant]
    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = args.run_root or DEFAULT_RUN_ROOT
    run_dir = run_root / f"q30-swe-full-rl-{variant.name}-{run_tag}"
    config = (
        args.repo_dir / "examples/nemo_gym/grpo_qwen3_30ba3b_thinking_swe2_smoke.yaml"
    )
    entrypoint = args.repo_dir / "examples/nemo_gym/run_grpo_nemo_gym.py"
    global_batch_size = args.num_prompts * args.num_generations
    wandb_run_name = f"q30-swe-full-rl-{variant.name}-{run_tag}"
    gym_revision = _git_output(
        args.repo_dir, "rev-parse", "HEAD:3rdparty/Gym-workspace/Gym"
    )
    gym_venv_dir = DEFAULT_GYM_VENV_ROOT / (f"{gym_revision}-py312-openai2.7.2")

    overrides = [
        f"data.train.data_path={args.dataset}",
        f"data.validation.data_path={args.dataset}",
        "cluster.num_nodes=9",
        "cluster.gpus_per_node=4",
        "++cluster.segment_size=8",
        "++env.nemo_gym.is_trajectory_collection=false",
        "++env.nemo_gym.subprocess_openai_version=2.7.2",
        f"grpo.max_num_steps={args.steps}",
        f"grpo.num_prompts_per_step={args.num_prompts}",
        f"grpo.num_generations_per_prompt={args.num_generations}",
        "grpo.val_at_start=false",
        "grpo.val_period=1000",
        "grpo.async_grpo.enabled=true",
        "grpo.async_grpo.in_flight_weight_updates=true",
        f"policy.train_global_batch_size={global_batch_size}",
        f"policy.generation_batch_size={global_batch_size}",
        "policy.train_micro_batch_size=1",
        "policy.logprob_batch_size=1",
        "policy.max_total_sequence_length=32768",
        "policy.generation.max_new_tokens=32768",
        "policy.generation.temperature=1.0",
        "policy.generation.top_p=1.0",
        "policy.megatron_cfg.tensor_model_parallel_size=4",
        "policy.megatron_cfg.pipeline_model_parallel_size=2",
        "policy.megatron_cfg.context_parallel_size=4",
        "policy.megatron_cfg.expert_model_parallel_size=8",
        "policy.generation.colocated.enabled=false",
        "policy.generation.colocated.resources.num_nodes=1",
        "policy.generation.colocated.resources.gpus_per_node=4",
        "policy.generation.vllm_cfg.tensor_parallel_size=2",
        "policy.generation.vllm_cfg.enforce_eager=false",
        "policy.generation.vllm_cfg.enable_vllm_metrics_logger=true",
        "policy.generation.vllm_cfg.vllm_metrics_logger_interval=0.5",
        "policy.generation.vllm_kwargs.moe_backend=triton",
        "++policy.generation.vllm_kwargs.cudagraph_metrics=true",
        "checkpointing.enabled=false",
        f"logger.log_dir={run_dir / 'logs'}",
        "logger.wandb_enabled=true",
        "logger.tensorboard_enabled=false",
        f"logger.wandb.project={WANDB_PROJECT}",
        f"logger.wandb.name={wandb_run_name}",
        "env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.agent_max_turns=8",
        "env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.concurrency=4",
        "env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.swebench_agent_timeout=900",
    ]
    if variant.speculative_tokens is None:
        overrides.append(
            "++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode="
            "FULL_AND_PIECEWISE"
        )
    else:
        overrides.extend(
            [
                "++policy.generation.vllm_kwargs.speculative_config.method=dflash",
                f"++policy.generation.vllm_kwargs.speculative_config.model={DFLASH_SNAPSHOT}",
                "++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1",
                "++policy.generation.vllm_kwargs.speculative_config.max_model_len=4096",
                "++policy.generation.vllm_kwargs.speculative_config.attention_backend=FLASH_ATTN",
                "++policy.generation.vllm_kwargs.kernel_config.enable_flashinfer_autotune=false",
                "++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=FULL",
                (
                    "++policy.generation.vllm_kwargs.speculative_config."
                    f"num_speculative_tokens={variant.speculative_tokens}"
                ),
                _capture_sizes_override(variant.capture_sizes),
            ]
        )

    unset_environment = [
        item for name in INCOMPATIBLE_INHERITED_ENVIRONMENT for item in ("-u", name)
    ]
    command = [
        "env",
        *unset_environment,
        "VLLM_USE_V2_MODEL_RUNNER=1",
        f"HF_HOME={args.hf_home}",
        f"NEMO_GYM_VENV_DIR={gym_venv_dir}",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        f"PYTHONPATH={args.repo_dir}",
        f"NEMO_RL_VENV_DIR=/tmp/nemorl-swe-full-{variant.name}-{run_tag}",
        "NRL_FORCE_REBUILD_VENVS=true",
        f"TRITON_CACHE_DIR=/tmp/nemorl-triton-{variant.name}-{run_tag}",
        f"TORCHINDUCTOR_CACHE_DIR=/tmp/nemorl-inductor-{variant.name}-{run_tag}",
        "UV_LOCK_TIMEOUT=3600",
        "PYTHONFAULTHANDLER=1",
        "RAY_DEDUP_LOGS=0",
        "NRL_OH_DEBUG=1",
        "NRL_SWE_UTIL_SYNTH=/lustre/fsw/coreai_dlalgo_llm/users/sna/swe_util_synth",
        "APPTAINER_CACHEDIR=/lustre/fsw/coreai_dlalgo_llm/users/sna/apptainer_cache",
        f"APPTAINER_TMPDIR=/tmp/apptainer-swe-full-{variant.name}-{run_tag}",
        "/opt/nemo_rl_venv/bin/python",
        str(entrypoint),
        "--config",
        str(config),
        *overrides,
    ]
    sbatch_args = [
        "--account=coreai_dlalgo_llm",
        "--partition=gb200",
        "--nodes=9",
        "--ntasks-per-node=1",
        "--exclusive",
        "--segment=9",
        f"--time={args.time_limit}",
        "--dependency=",
        f"--job-name=coreai_dlalgo_llm-nemorl.swe-full-{variant.name}",
        f"--output={run_dir / 'slurm-%j.out'}",
    ]
    return RunPlan(
        variant=variant.name,
        run_tag=run_tag,
        run_dir=str(run_dir),
        config=str(config),
        dataset=str(args.dataset),
        container=str(args.container),
        gym_venv_dir=str(gym_venv_dir),
        wandb_project=WANDB_PROJECT,
        wandb_run_name=wandb_run_name,
        command=tuple(command),
        overrides=tuple(overrides),
        sbatch_args=tuple(sbatch_args),
        submission_unset_environment=INCOMPATIBLE_INHERITED_ENVIRONMENT,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(repo_dir: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_dir), *args],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def validate_remote_inputs(plan: RunPlan, repo_dir: Path) -> None:
    required_paths = [
        repo_dir / "ray.sub",
        Path(plan.config),
        Path(plan.dataset),
        Path(plan.container),
    ]
    if plan.variant != "baseline":
        required_paths.append(DFLASH_SNAPSHOT)
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths: " + ", ".join(missing))
    status = _git_output(repo_dir, "status", "--porcelain")
    if status:
        raise RuntimeError("Refusing to submit from a dirty worktree:\n" + status)


def write_provenance(plan: RunPlan, repo_dir: Path) -> Path:
    run_dir = Path(plan.run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    gym_app_path, _, gym_app_sha256 = apply_gym_openhands_tmux_fix(repo_dir)
    provenance = {
        **asdict(plan),
        "repo_head": _git_output(repo_dir, "rev-parse", "HEAD"),
        "repo_branch": _git_output(repo_dir, "branch", "--show-current"),
        "submodules": _git_output(
            repo_dir, "submodule", "status", "--recursive"
        ).splitlines(),
        "dataset_sha256": _sha256(Path(plan.dataset)),
        "container_size_bytes": Path(plan.container).stat().st_size,
        "gym_openhands_tmux_fix": {
            "path": str(gym_app_path),
            "sha256": gym_app_sha256,
        },
    }
    path = run_dir / "provenance.json"
    path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return path


def execute(plan: RunPlan, repo_dir: Path, mode: str) -> int:
    validate_remote_inputs(plan, repo_dir)
    ray_sub = repo_dir / "ray.sub"
    env = os.environ.copy()
    for name in plan.submission_unset_environment:
        env.pop(name, None)
    env.update(
        {
            "COMMAND": shlex.join(plan.command),
            "CONTAINER": plan.container,
            "MOUNTS": "/lustre:/lustre,/dev/fuse:/dev/fuse",
            "GPUS_PER_NODE": "4",
            "BASE_LOG_DIR": plan.run_dir,
        }
    )
    if mode == "test-only":
        command = ["sbatch", "--test-only", *plan.sbatch_args, str(ray_sub)]
    else:
        apply_gym_openhands_tmux_fix(repo_dir)
        provenance_path = write_provenance(plan, repo_dir)
        command = ["sbatch", "--parsable", *plan.sbatch_args, str(ray_sub)]
        print(f"provenance={provenance_path}", file=sys.stderr)
    result = subprocess.run(
        command, check=False, env=env, text=True, capture_output=True
    )
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip(), file=sys.stderr)
    return result.returncode


def main() -> int:
    args = parse_args()
    plan = build_plan(args)
    if args.mode == "dry-run":
        if args.json:
            print(json.dumps(asdict(plan), indent=2, sort_keys=True))
        else:
            print("COMMAND=" + shlex.join(plan.command))
            print(
                "SBATCH="
                + shlex.join(
                    ("sbatch", *plan.sbatch_args, str(args.repo_dir / "ray.sub"))
                )
            )
        return 0
    if args.mode == "submit" and not os.environ.get("WANDB_API_KEY"):
        print(
            "WANDB_API_KEY must be set in the submission environment",
            file=sys.stderr,
        )
        return 2
    return execute(plan, args.repo_dir, args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
