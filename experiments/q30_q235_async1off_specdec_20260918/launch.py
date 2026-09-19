"""Render matched Qwen3 Async-1off SpecDec benchmark jobs on OCI-HSG."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess


SOURCE = Path(
    os.environ.get(
        "ASYNC1OFF_SOURCE",
        "/home/sna/nemorl-q235-dspark-rp25-s44000-20260918",
    )
)
BASE = Path("/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna")
CONTAINER = BASE / "containers/nemo_rl_nightly_20260917_7214802.sqsh"
ARTIFACTS = BASE / "experiments/q30-q235-async1off-specdec-20260918"
MCORE_SOURCE = (
    SOURCE / "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
)
OVERLAY_BUILDER = (
    SOURCE / "experiments/qwen3_30ba3b_bf16_flashinfer_specdec_latest_main_20260909/"
    "prepare_vllm_dspark_fap_overlay.py"
)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Immutable official-recipe and resource contract for one target model."""

    label: str
    recipe: Path
    target: Path
    nodes: int
    gpus_per_node: int
    segment_size: int
    walltime: str


@dataclass(frozen=True, slots=True)
class ArmSpec:
    """One frozen speculative-decoding arm."""

    label: str
    method: str | None
    k: int
    models: frozenset[str]
    drafter: Path | None = None


MODELS = {
    "q30": ModelSpec(
        label="Qwen3-30BA3B",
        recipe=Path(
            "examples/configs/recipes/llm/performance/"
            "grpo-qwen3-30ba3b-4n4g-async-1off.yaml"
        ),
        target=(
            BASE / "hf_home/hub/models--Qwen--Qwen3-30B-A3B/snapshots/"
            "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
        ),
        nodes=4,
        gpus_per_node=4,
        segment_size=4,
        walltime="04:00:00",
    ),
    "q235": ModelSpec(
        label="Qwen3-235B",
        recipe=Path(
            "examples/configs/recipes/llm/performance/"
            "grpo-qwen3-235b-32n4g-async-1off.yaml"
        ),
        target=(
            BASE / "hf_home/hub/models--Qwen--Qwen3-235B-A22B/snapshots/"
            "8efa61729e24bd65b1d152b5ab5409052aa80e65"
        ),
        nodes=32,
        gpus_per_node=4,
        segment_size=16,
        walltime="05:00:00",
    ),
}

ARMS = {
    "baseline": ArmSpec(
        label="Baseline-S64",
        method=None,
        k=0,
        models=frozenset(MODELS),
    ),
    "dflash_k5": ArmSpec(
        label="DFlashK5-S64",
        method="dflash",
        k=5,
        models=frozenset({"q30"}),
        drafter=(
            BASE / "specdec_ptv23/ptv3_swa/"
            "sd2p3swa-q30-base-ptv3swe-dflash-b8-16n/"
            "exported-checkpoint-44000"
        ),
    ),
    "dspark_k5": ArmSpec(
        label="DSparkK5-S64",
        method="dspark",
        k=5,
        models=frozenset({"q30"}),
        drafter=(
            BASE / "specdec_ptv23/ptv3_swa/"
            "sd2p3swa-q30-base-ptv3swe-dspark-b8-16n/"
            "exported-checkpoint-44000"
        ),
    ),
    "dflash_k7": ArmSpec(
        label="DFlashK7-S64",
        method="dflash",
        k=7,
        models=frozenset({"q235"}),
        drafter=(
            BASE / "specdec_ptv23/drafters_ptv2en/"
            "sd2en-q235-base-ptv2en-dflash-b8-16n/"
            "exported-checkpoint-25391"
        ),
    ),
    "dspark_k7": ArmSpec(
        label="DSparkK7-S64",
        method="dspark",
        k=7,
        models=frozenset({"q235"}),
        drafter=(
            BASE / "drafters/specdec_ptv23_s44000/"
            "sd2p3rp-q235-base-ptv3rp25-dspark-b8-16n/"
            "exported-checkpoint-44000"
        ),
    ),
}


def capture_sizes(arm: ArmSpec, max_num_seqs: int = 64) -> str:
    """Return FAP shapes covering request and method-specific verification widths."""
    request_buckets = tuple(
        requests for requests in (1, 2, 4, 8, 16, 32, 64) if requests <= max_num_seqs
    )
    if arm.method is None:
        return "[" + ",".join(str(value) for value in request_buckets) + "]"

    target_width = arm.k + 1
    values = {request for request in request_buckets if request < target_width}
    values.update(target_width * request for request in request_buckets)
    if arm.method == "dspark":
        values.update({arm.k, arm.k * max_num_seqs})
        values.update(
            target_width * ((arm.k * request) // target_width)
            for request in request_buckets
            if (arm.k * request) // target_width > 0
        )
    return "[" + ",".join(str(value) for value in sorted(values)) + "]"


def configuration(model: str, arm: str, steps: int) -> dict[str, str]:
    """Return controlled overrides on top of an official Async-1off recipe."""
    if model not in MODELS:
        raise ValueError(f"unknown model: {model}")
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if steps not in {1, 20}:
        raise ValueError("the Async-1off cohort supports 1-step and 20-step runs")
    arm_spec = ARMS[arm]
    if model not in arm_spec.models:
        raise ValueError(f"arm {arm} is not valid for model {model}")

    model_spec = MODELS[model]
    values = {
        "grpo.max_num_steps": str(steps),
        "checkpointing.enabled": "false",
        "policy.model_name": str(model_spec.target),
        "policy.tokenizer.name": str(model_spec.target),
        "policy.precision": "bfloat16",
        "policy.draft.enabled": "false",
        "policy.generation.vllm_cfg.enforce_eager": "false",
        "policy.generation.vllm_kwargs.max_num_seqs": "64",
        "policy.generation.vllm_kwargs.moe_backend": "flashinfer_trtllm",
        "policy.generation.vllm_kwargs.compilation_config.cudagraph_mode": (
            "FULL_AND_PIECEWISE"
        ),
        (
            "policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes"
        ): capture_sizes(arm_spec),
        "logger.wandb_enabled": "true",
        "logger.tensorboard_enabled": "false",
        "logger.wandb.project": "sna-specdec",
        "logger.wandb.group": "q30-q235-async1off-specdec-20260918",
    }
    if arm_spec.method is None:
        values["policy.generation.vllm_kwargs.speculative_config"] = "null"
        return values

    if arm_spec.drafter is None:
        raise ValueError(f"arm {arm} has no drafter")
    values.update(
        {
            "policy.generation.refit_cfg.memory_lifecycle.mode": ("specdec_deep_refit"),
            "policy.generation.vllm_kwargs.speculative_config.method": (
                arm_spec.method
            ),
            "policy.generation.vllm_kwargs.speculative_config.model": str(
                arm_spec.drafter
            ),
            (
                "policy.generation.vllm_kwargs.speculative_config."
                "num_speculative_tokens"
            ): str(arm_spec.k),
            (
                "policy.generation.vllm_kwargs.speculative_config."
                "draft_tensor_parallel_size"
            ): "1",
            (
                "policy.generation.vllm_kwargs.speculative_config.attention_backend"
            ): "FLASH_ATTN",
            "policy.generation.vllm_kwargs.kernel_config.enable_flashinfer_autotune": (
                "false"
            ),
        }
    )
    return values


def required_inputs(model: str, arm: str) -> list[Path]:
    """Return all immutable files needed by a job."""
    if model not in MODELS or arm not in ARMS:
        raise ValueError("unknown model or arm")
    model_spec = MODELS[model]
    arm_spec = ARMS[arm]
    if model not in arm_spec.models:
        raise ValueError(f"arm {arm} is not valid for model {model}")
    required = [
        SOURCE / "ray.sub",
        SOURCE / model_spec.recipe,
        CONTAINER,
        model_spec.target / "config.json",
        model_spec.target / "model.safetensors.index.json",
        MCORE_SOURCE / "megatron/core/datasets/helpers.cpp",
    ]
    if arm_spec.drafter is not None:
        required.extend(
            [arm_spec.drafter / "config.json", arm_spec.drafter / "model.safetensors"]
        )
    if arm_spec.method == "dspark":
        required.extend(
            [
                OVERLAY_BUILDER,
                OVERLAY_BUILDER.parent / "patches/vllm-0.25.1-pr48167-runtime.patch",
                OVERLAY_BUILDER.parent
                / "patches/vllm-0.25.1-pr48167-group-causality-followup.patch",
            ]
        )
    return required


def render(
    account: str,
    run_name: str,
    *,
    model: str,
    arm: str,
    steps: int,
    directory: Path,
) -> str:
    """Render one reproducible OCI-HSG batch job."""
    if not re.fullmatch(r"[a-z0-9_]+", account):
        raise ValueError("invalid account")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_name):
        raise ValueError("invalid run name")
    model_spec = MODELS[model]
    arm_spec = ARMS[arm]
    values = configuration(model=model, arm=arm, steps=steps)
    values["logger.wandb.name"] = run_name
    values["logger.log_dir"] = str(directory / "metrics")
    command = shlex.join(
        [
            "/opt/nemo_rl_venv/bin/python",
            "examples/run_grpo.py",
            "--config",
            str(model_spec.recipe),
            *[f"++{key}={value}" for key, value in values.items()],
        ]
    )
    drafter_checks = ""
    if arm_spec.drafter is not None:
        drafter_checks = (
            f'test -f "{arm_spec.drafter}/config.json"\n'
            f'test -f "{arm_spec.drafter}/model.safetensors"\n'
        )
    dspark_setup = ""
    if arm_spec.method == "dspark":
        dspark_setup = (
            f"; /usr/local/bin/python-VllmGenerationWorker {OVERLAY_BUILDER}"
            ' --overlay-root "$ASYNC_VLLM_OVERLAY"'
        )
    walltime = "01:30:00" if steps == 1 else model_spec.walltime
    return f"""#!/usr/bin/env bash
#SBATCH --job-name={account}-specdec.{run_name}
#SBATCH --account={account}
#SBATCH --partition=batch
#SBATCH --time={walltime}
#SBATCH --nodes={model_spec.nodes}
#SBATCH --gpus-per-node={model_spec.gpus_per_node}
#SBATCH --segment={model_spec.segment_size}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --output={directory}/slurm-%j.out
#SBATCH --error={directory}/slurm-%j.err
set -euo pipefail
export PATH=/cm/local/apps/slurm/25.11/bin:$PATH
export NCCL_NVLS_ENABLE=0
export NRL_DISABLE_NUMA_MEMBIND=1
export RAY_memory_usage_threshold=0.98
test -n "${{WANDB_API_KEY:-}}"
test -z "$(git -C {SOURCE} status --porcelain=v1 --untracked-files=all)"
test -f "{model_spec.target}/config.json"
test -f "{model_spec.target}/model.safetensors.index.json"
{drafter_checks}mkdir -p {directory}
git -C {SOURCE} rev-parse HEAD >{directory}/source_sha.txt
git -C {SOURCE} submodule status --recursive >{directory}/submodules.txt
export CONTAINER={CONTAINER}
export MOUNTS=/home:/home,/lustre:/lustre,/raid:/raid
export GPUS_PER_NODE={model_spec.gpus_per_node} DEDICATED_RAY_HEAD=0
export BASE_LOG_DIR={directory}
export NETRC=/home/sna/.netrc
export HF_HOME={BASE}/hf_home
export HF_DATASETS_CACHE=${{HF_HOME}}/datasets
export WANDB_DIR={directory}
export ASYNC_NODE_ROOT=/raid/scratch/sna/async1off-${{SLURM_JOB_ID}}
export ASYNC_MCORE_OVERLAY=${{ASYNC_NODE_ROOT}}/mcore-overlay
export ASYNC_VLLM_OVERLAY=${{ASYNC_NODE_ROOT}}/vllm-overlay
export XDG_CACHE_HOME=${{ASYNC_NODE_ROOT}}/cache
export TRITON_CACHE_DIR=${{XDG_CACHE_HOME}}/triton
export TORCH_EXTENSIONS_DIR=${{XDG_CACHE_HOME}}/torch-extensions
export WANDB_CACHE_DIR=${{XDG_CACHE_HOME}}/wandb
export WANDB_CONFIG_DIR=${{XDG_CACHE_HOME}}/wandb-config
export RAY_TMPDIR=${{ASYNC_NODE_ROOT}}/ray
export NRL_NATIVE_TMP=${{ASYNC_NODE_ROOT}}/tmp
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset PYTHONOPTIMIZE NRL_FORCE_REBUILD_VENVS TMPDIR
export PYTHONPATH=${{ASYNC_VLLM_OVERLAY}}:${{ASYNC_MCORE_OVERLAY}}:{SOURCE}
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=PYTHONPATH
export SETUP_COMMAND='set -euo pipefail
mkdir -p "$ASYNC_MCORE_OVERLAY" "$ASYNC_VLLM_OVERLAY" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$RAY_TMPDIR" "$NRL_NATIVE_TMP"
cp -R "{MCORE_SOURCE}/megatron" "$ASYNC_MCORE_OVERLAY/"
test -f "$ASYNC_MCORE_OVERLAY/megatron/core/datasets/helpers.cpp"
test -x /opt/nemo_rl_venv/bin/python
test -x /usr/local/bin/python-MegatronPolicyWorker
test -x /usr/local/bin/python-VllmGenerationWorker{dspark_setup}'
export COMMAND={
        shlex.quote(f'''set -euo pipefail
cd {SOURCE}
export TMPDIR=$NRL_NATIVE_TMP
/opt/nemo_rl_venv/bin/python -c 'from pathlib import Path; import nemo_rl; p=Path(nemo_rl.__file__).resolve(); root=Path("{SOURCE}").resolve(); assert p.is_relative_to(root), (p, root)'
exec {command}''')
    }
exec bash {SOURCE}/ray.sub
"""


def sbatch_arguments(job: Path, *, test_only: bool) -> list[str]:
    """Return an independent scheduler invocation."""
    return ["sbatch", "--test-only" if test_only else "--parsable", str(job)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--render", action="store_true")
    mode.add_argument("--test-only", action="store_true")
    mode.add_argument("--submit", action="store_true")
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--steps", type=int, choices=(1, 20), required=True)
    parser.add_argument("--account", default="coreai_dlalgo_nemorl")
    args = parser.parse_args()

    arm_spec = ARMS[args.arm]
    if args.model not in arm_spec.models:
        parser.error(f"arm {args.arm} is not valid for model {args.model}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_spec = MODELS[args.model]
    run_name = f"{model_spec.label}-Async1off-{arm_spec.label}-{args.steps}step-{stamp}"
    directory = ARTIFACTS / run_name
    script = render(
        account=args.account,
        run_name=run_name,
        model=args.model,
        arm=args.arm,
        steps=args.steps,
        directory=directory,
    )
    if args.render:
        print(script)
        return

    missing = [
        str(path)
        for path in required_inputs(args.model, args.arm)
        if not path.is_file()
    ]
    if missing:
        parser.error("missing required input: " + ", ".join(missing))
    if not os.environ.get("WANDB_API_KEY"):
        parser.error("WANDB_API_KEY is not exported")
    status = subprocess.check_output(
        ["git", "-C", str(SOURCE), "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
    )
    if status:
        parser.error("source checkout is dirty")

    directory.mkdir(parents=True, exist_ok=False)
    (directory / "overrides.json").write_text(
        json.dumps(configuration(args.model, args.arm, args.steps), indent=2) + "\n"
    )
    job = directory / "job.sbatch"
    job.write_text(script)
    job.chmod(0o700)
    test_result = subprocess.run(
        sbatch_arguments(job, test_only=True),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (directory / "test-only.txt").write_text(test_result.stdout)
    print(test_result.stdout, end="", flush=True)
    test_result.check_returncode()
    if args.submit:
        submit_result = subprocess.run(
            sbatch_arguments(job, test_only=False),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (directory / "submission.txt").write_text(submit_result.stdout)
        print(submit_result.stdout, end="", flush=True)
        submit_result.check_returncode()
    print(f"Artifacts: {directory}")


if __name__ == "__main__":
    main()
