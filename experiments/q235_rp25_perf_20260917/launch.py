"""Render and submit the Qwen3-235B no-SpecDec performance baseline."""

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


SOURCE = Path("/home/sna/nemorl-q235-specdec-matrix-20260917")
INITIALIZED_SOURCE = Path("/home/sna/nemorl-q235-rp25-perf-20260917")
RECIPE = Path("examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g.yaml")
BASE = Path("/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna")
TARGET_REVISION = "8efa61729e24bd65b1d152b5ab5409052aa80e65"
TARGET = BASE / "hf_home/hub/models--Qwen--Qwen3-235B-A22B/snapshots" / TARGET_REVISION
CONTAINER = BASE / "containers/nemo_rl_nightly_20260909_7023221.sqsh"
ARTIFACTS = BASE / "experiments/q235-rp25-perf-20260917"
SHARED_MEGATRON_CHECKPOINT = ARTIFACTS / "shared-megatron-initial-checkpoint"
MCORE_SOURCE = (
    INITIALIZED_SOURCE
    / "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
)
VLLM_WORKER_PYTHON = Path("/usr/local/bin/python-VllmGenerationWorker")
DSPARK_OVERLAY_BUILDER = (
    SOURCE / "experiments/qwen3_30ba3b_bf16_flashinfer_specdec_latest_main_20260909/"
    "prepare_vllm_dspark_fap_overlay.py"
)


@dataclass(frozen=True, slots=True)
class SiteSpec:
    """Paths and scheduler limits for one controlled cluster."""

    base: Path
    target: Path
    container: Path
    artifacts: Path
    partition: str
    gpu_directive: str | None
    walltime: str
    default_account: str


@dataclass(frozen=True, slots=True)
class ArmSpec:
    """One frozen speculative-decoding measurement arm."""

    label: str
    method: str | None
    k: int
    block_size: int


PTYCHE_BASE = Path("/lustre/fsw/coreai_dlalgo_llm/users/sna")
LYRIS_BASE = Path("/lustre/fsw/coreai_dlalgo_llm/users/sna")
SITES = {
    "oci": SiteSpec(
        base=BASE,
        target=TARGET,
        container=CONTAINER,
        artifacts=ARTIFACTS,
        partition="batch",
        gpu_directive="gpus-per-node=4",
        walltime="04:00:00",
        default_account="coreai_dlalgo_nemorl",
    ),
    "ptyche": SiteSpec(
        base=PTYCHE_BASE,
        target=(PTYCHE_BASE / f"models/Qwen3-235B-A22B-{TARGET_REVISION}"),
        container=(
            PTYCHE_BASE / "containers/nemo-rl-20260916/"
            "nemo_rl_nightly_20260916_2837270.sqsh"
        ),
        artifacts=PTYCHE_BASE / "experiments/q235-rp25-perf-20260917",
        partition="36x2-a01r",
        gpu_directive=None,
        walltime="03:00:00",
        default_account="coreai_dlalgo_llm",
    ),
    "lyris": SiteSpec(
        base=LYRIS_BASE,
        target=(
            LYRIS_BASE
            / "hf_home/hub/models--Qwen--Qwen3-235B-A22B/snapshots"
            / TARGET_REVISION
        ),
        container=(
            LYRIS_BASE / "containers/nemo-rl-20260916/"
            "nemo_rl_nightly_20260916_3078480.sqsh"
        ),
        artifacts=LYRIS_BASE / "experiments/q235-rp25-perf-20260917",
        partition="gb200",
        gpu_directive=None,
        walltime="03:00:00",
        default_account="coreai_dlalgo_llm",
    ),
}

ARMS = {
    "baseline": ArmSpec("Baseline", None, 0, 0),
    "dflash_k5": ArmSpec("DFlashK5-B8", "dflash", 5, 8),
    "dflash_k7": ArmSpec("DFlashK7-B8", "dflash", 7, 8),
    "dspark_k5": ArmSpec("DSparkK5-B8", "dspark", 5, 8),
    "dspark_k7": ArmSpec("DSparkK7-B8", "dspark", 7, 8),
    "dflash_b16_k11": ArmSpec("DFlashK11-B16", "dflash", 11, 16),
    "dflash_b16_k13": ArmSpec("DFlashK13-B16", "dflash", 13, 16),
    "dspark_b16_k11": ArmSpec("DSparkK11-B16", "dspark", 11, 16),
    "dspark_b16_k13": ArmSpec("DSparkK13-B16", "dspark", 13, 16),
    "eagle3_k3": ArmSpec("Eagle3K3", "eagle3", 3, 0),
    "eagle3_k5": ArmSpec("Eagle3K5", "eagle3", 5, 0),
}


def _drafter_path(site: str, arm: ArmSpec) -> Path:
    if arm.method == "eagle3":
        if site == "oci":
            root = BASE
        else:
            root = SITES[site].base
        return (
            root / "hf_home/hub/"
            "models--RedHatAI--Qwen3-235B-A22B-speculator.eagle3/snapshots/"
            "6b3306ff92023cc3b393ec9bb26a168ad88096fe"
        )
    if arm.method is None:
        raise ValueError("baseline has no drafter")
    if site == "oci":
        return (
            BASE
            / "specdec_ptv23/drafters_ptv2en"
            / f"sd2en-q235-base-ptv2en-{arm.method}-b{arm.block_size}-16n"
            / "exported-checkpoint-25391"
        )
    return (
        SITES[site].base
        / "drafters/q235-base-ptv2en-s25391-20260917"
        / f"{arm.method}-b{arm.block_size}"
    )


def _capture_sizes(arm: ArmSpec, max_num_seqs: int = 64) -> str:
    widths = {1, arm.k + 1}
    if arm.method == "dspark":
        widths.add(arm.k)
    values = {
        requests * width
        for width in widths
        for requests in (1, 2, 4, 8, 16, 32, 64)
        if requests <= max_num_seqs
    }
    return "[" + ",".join(str(value) for value in sorted(values)) + "]"


def configuration(
    steps: int = 20, site: str = "oci", arm: str = "baseline"
) -> dict[str, str]:
    """Return the minimal overrides on top of the official 16n4g recipe."""
    if steps not in {1, 20}:
        raise ValueError("this controlled baseline supports 1-step gates or 20 steps")
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    spec = SITES[site]
    arm_spec = ARMS[arm]
    values = {
        "grpo.max_num_steps": str(steps),
        "checkpointing.enabled": "false",
        "policy.model_name": str(spec.target),
        "policy.tokenizer.name": str(spec.target),
        "policy.precision": "bfloat16",
        "policy.megatron_cfg.distributed_timeout_seconds": "2400",
        "policy.draft.enabled": "false",
        "policy.generation.vllm_cfg.enforce_eager": "false",
        "policy.generation.vllm_kwargs.compilation_config.cudagraph_mode": (
            "FULL_AND_PIECEWISE"
        ),
        (
            "policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes"
        ): "[1,2,4,8,16,32,64]",
        "logger.wandb_enabled": "true",
        "logger.tensorboard_enabled": "false",
        "logger.wandb.project": "sna-specdec",
        "logger.wandb.group": "q235-rp25-frozen-perf-20260917",
    }
    if arm_spec.method is None:
        values["policy.generation.vllm_kwargs.speculative_config"] = "null"
        return values

    values.update(
        {
            "policy.generation.vllm_kwargs.max_num_seqs": "64",
            (
                "policy.generation.vllm_kwargs.compilation_config."
                "cudagraph_capture_sizes"
            ): _capture_sizes(arm_spec),
            "policy.generation.vllm_kwargs.speculative_config.method": (
                arm_spec.method
            ),
            "policy.generation.vllm_kwargs.speculative_config.model": str(
                _drafter_path(site, arm_spec)
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
        }
    )
    if arm_spec.method in {"dflash", "dspark"}:
        values[
            "policy.generation.vllm_kwargs.kernel_config.enable_flashinfer_autotune"
        ] = "false"
    return values


def required_inputs(site: str, arm: str = "baseline") -> list[Path]:
    """Return every immutable file required before an ungated submission."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    spec = SITES[site]
    inputs = [
        SOURCE / "ray.sub",
        spec.container,
        spec.target / "config.json",
        spec.target / "model.safetensors.index.json",
        MCORE_SOURCE / "megatron/core/datasets/helpers.cpp",
    ]
    arm_spec = ARMS[arm]
    if arm_spec.method is not None:
        drafter = _drafter_path(site, arm_spec)
        inputs.extend([drafter / "config.json", drafter / "model.safetensors"])
    if arm_spec.method == "dspark":
        inputs.extend(
            [
                DSPARK_OVERLAY_BUILDER,
                DSPARK_OVERLAY_BUILDER.parent
                / "patches/vllm-0.25.1-pr48167-runtime.patch",
                DSPARK_OVERLAY_BUILDER.parent
                / "patches/vllm-0.25.1-pr48167-group-causality-followup.patch",
            ]
        )
    return inputs


def render(
    account: str,
    run_name: str,
    *,
    steps: int = 20,
    site: str = "oci",
    arm: str = "baseline",
    directory: Path | None = None,
) -> str:
    """Render one controlled batch job without mutating scheduler state."""
    if not re.fullmatch(r"[a-z0-9_]+", account):
        raise ValueError("invalid account")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_name):
        raise ValueError("invalid run name")
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    spec = SITES[site]
    arm_spec = ARMS[arm]
    walltime = "01:00:00" if steps == 1 else spec.walltime
    gpu_directive = f"#SBATCH --{spec.gpu_directive}\n" if spec.gpu_directive else ""
    run_dir = directory or spec.artifacts / run_name
    shared_megatron_checkpoint = spec.artifacts / "shared-megatron-initial-checkpoint"
    values = configuration(steps, site, arm)
    values["logger.wandb.name"] = run_name
    values["logger.log_dir"] = str(run_dir / "metrics")
    command = shlex.join(
        [
            "/opt/nemo_rl_venv/bin/python",
            "examples/run_grpo.py",
            "--config",
            str(RECIPE),
            *[f"++{key}={value}" for key, value in values.items()],
        ]
    )
    drafter_checks = ""
    if arm_spec.method is not None:
        drafter = _drafter_path(site, arm_spec)
        drafter_checks = (
            f'test -f "{drafter}/config.json"\ntest -f "{drafter}/model.safetensors"\n'
        )
    dspark_setup = ""
    if arm_spec.method == "dspark":
        dspark_setup = (
            f"; {VLLM_WORKER_PYTHON} {DSPARK_OVERLAY_BUILDER}"
            ' --overlay-root "$Q235_VLLM_OVERLAY"'
        )
    return f"""#!/usr/bin/env bash
#SBATCH --job-name={account}-specdec.{run_name}
#SBATCH --account={account}
#SBATCH --partition={spec.partition}
#SBATCH --time={walltime}
#SBATCH --nodes=16
#SBATCH --segment=16
{gpu_directive}#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --output={run_dir}/slurm-%j.out
#SBATCH --error={run_dir}/slurm-%j.err
set -euo pipefail
export PATH=/cm/local/apps/slurm/25.11/bin:$PATH
export NCCL_NVLS_ENABLE=0
export NRL_DISABLE_NUMA_MEMBIND=1
test -n "${{WANDB_API_KEY:-}}"
test -z "$(git -C {SOURCE} status --porcelain=v1 --untracked-files=all)"
{drafter_checks}test -f "{spec.target}/config.json"
test -f "{spec.target}/model.safetensors.index.json"
git -C {SOURCE} rev-parse HEAD >{run_dir}/source_sha.txt
git -C {SOURCE} submodule status --recursive >{run_dir}/submodules.txt
export CONTAINER={spec.container}
export MOUNTS=/home:/home,/lustre:/lustre,/raid:/raid
export GPUS_PER_NODE=4 DEDICATED_RAY_HEAD=0
export BASE_LOG_DIR={run_dir}
export NETRC=/home/sna/.netrc
export HF_HOME={spec.base}/hf_home
export HF_DATASETS_CACHE=${{HF_HOME}}/datasets
export NRL_MEGATRON_CHECKPOINT_DIR={shared_megatron_checkpoint}
export NRL_MOUNT_CHECK_ID={run_name}
export Q235_NODE_ROOT=/raid/scratch/sna/q235-${{SLURM_JOB_ID}}
export Q235_MCORE_SOURCE={MCORE_SOURCE}
export Q235_MCORE_OVERLAY=${{Q235_NODE_ROOT}}/mcore-overlay
export Q235_VLLM_OVERLAY=${{Q235_NODE_ROOT}}/vllm-overlay
export XDG_CACHE_HOME=${{Q235_NODE_ROOT}}/cache
export TRITON_CACHE_DIR=${{XDG_CACHE_HOME}}/triton
export TORCH_EXTENSIONS_DIR=${{XDG_CACHE_HOME}}/torch-extensions
export WANDB_DIR={run_dir}
export WANDB_CACHE_DIR=${{XDG_CACHE_HOME}}/wandb
export WANDB_CONFIG_DIR=${{XDG_CACHE_HOME}}/wandb-config
export RAY_TMPDIR=/raid/scratch/sna/r${{SLURM_JOB_ID}}
export NRL_NATIVE_TMP=/raid/scratch/sna/q235-${{SLURM_JOB_ID}}/tmp
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset PYTHONOPTIMIZE NRL_FORCE_REBUILD_VENVS TMPDIR
export PYTHONPATH=${{Q235_VLLM_OVERLAY}}:${{Q235_MCORE_OVERLAY}}:{SOURCE}
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=PYTHONPATH
export SETUP_COMMAND='set -euo pipefail
mkdir -p "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$RAY_TMPDIR" "$NRL_NATIVE_TMP" "$Q235_MCORE_OVERLAY"
cp -R "$Q235_MCORE_SOURCE/megatron" "$Q235_MCORE_OVERLAY/"
test -f "$Q235_MCORE_OVERLAY/megatron/core/datasets/helpers.cpp"
test -d "$NRL_MEGATRON_CHECKPOINT_DIR"
test -w "$NRL_MEGATRON_CHECKPOINT_DIR"
touch "$NRL_MEGATRON_CHECKPOINT_DIR/.mount-check-${{NRL_MOUNT_CHECK_ID}}-$(hostname)"
test -x /opt/nemo_rl_venv/bin/python
test -x /usr/local/bin/python-MegatronPolicyWorker
test -x /usr/local/bin/python-VllmGenerationWorker{dspark_setup}'
export COMMAND={
        shlex.quote(f'''set -euo pipefail
cd {SOURCE}
unset PYTHONOPTIMIZE NRL_FORCE_REBUILD_VENVS
export TMPDIR=$NRL_NATIVE_TMP
echo SOURCE_MODE=mounted-worktree
git rev-parse HEAD || true
sha256sum {RECIPE} >{run_dir}/container_recipe_sha256.txt
/opt/nemo_rl_venv/bin/python -c 'import sys; from pathlib import Path; import nemo_rl; p = Path(nemo_rl.__file__).resolve(); root = Path("{SOURCE}").resolve(); assert p.is_relative_to(root), (p, root); print("SOURCE_IMPORT_OK", sys.executable, p)'
/usr/local/bin/python-MegatronPolicyWorker -c 'import os; from pathlib import Path; from nemo_rl.models.policy.utils import get_megatron_checkpoint_dir; p = Path(get_megatron_checkpoint_dir()); assert str(p) == os.environ["NRL_MEGATRON_CHECKPOINT_DIR"]; run_id = os.environ["NRL_MOUNT_CHECK_ID"]; markers = list(p.glob(f".mount-check-{{run_id}}-*")); assert len(markers) == 16, markers; print("SHARED_CHECKPOINT_PREFLIGHT_OK", p, len(markers))'
exec {command}''')
    }
exec bash {SOURCE}/ray.sub
"""


def sbatch_arguments(
    job: Path, *, test_only: bool, afterok: int | None = None
) -> list[str]:
    """Build scheduler arguments, optionally gating launch on staged inputs."""
    if afterok is not None and afterok <= 0:
        raise ValueError("afterok must be a positive SLURM job ID")
    arguments = ["sbatch", "--test-only" if test_only else "--parsable"]
    if afterok is not None:
        arguments.append(f"--dependency=afterok:{afterok}")
    return [*arguments, str(job)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--render", action="store_true")
    mode.add_argument("--test-only", action="store_true")
    mode.add_argument("--submit", action="store_true")
    parser.add_argument("--site", choices=tuple(SITES), default="oci")
    parser.add_argument("--arm", choices=tuple(ARMS), default="baseline")
    parser.add_argument("--account")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--afterok", type=int)
    args = parser.parse_args()
    spec = SITES[args.site]
    account = args.account or spec.default_account

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"Qwen3-235B-{ARMS[args.arm].label}-{args.steps}step-{stamp}"
    directory = spec.artifacts / run_name
    script = render(
        account=account,
        run_name=run_name,
        steps=args.steps,
        site=args.site,
        arm=args.arm,
        directory=directory,
    )
    if args.render:
        print(script)
        return

    required = required_inputs(args.site, args.arm)
    if args.afterok is not None and ARMS[args.arm].method is not None:
        drafter = _drafter_path(args.site, ARMS[args.arm])
        required = [
            path
            for path in required
            if path not in {drafter / "config.json", drafter / "model.safetensors"}
        ]
    missing = [str(path) for path in required if not path.is_file()]
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
    (spec.artifacts / "shared-megatron-initial-checkpoint").mkdir(
        parents=True, exist_ok=True
    )
    (directory / "overrides.json").write_text(
        json.dumps(configuration(args.steps, args.site, args.arm), indent=2) + "\n"
    )
    if args.afterok is not None:
        (directory / "dependency.json").write_text(
            json.dumps({"afterok": args.afterok}, indent=2) + "\n"
        )
    job = directory / "job.sbatch"
    job.write_text(script)
    job.chmod(0o700)
    result = subprocess.run(
        sbatch_arguments(job, test_only=True, afterok=args.afterok),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (directory / "test-only.txt").write_text(result.stdout)
    print(result.stdout, end="", flush=True)
    result.check_returncode()
    if args.submit:
        result = subprocess.run(
            sbatch_arguments(job, test_only=False, afterok=args.afterok),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (directory / "submission.txt").write_text(result.stdout)
        print(result.stdout, end="", flush=True)
        result.check_returncode()
    print(f"Artifacts: {directory}")


if __name__ == "__main__":
    main()
