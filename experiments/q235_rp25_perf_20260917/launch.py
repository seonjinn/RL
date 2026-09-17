"""Render and submit the Qwen3-235B no-SpecDec performance baseline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess


SOURCE = Path("/home/sna/nemorl-q235-rp25-perf-20260917")
RECIPE = Path(
    "examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g.yaml"
)
BASE = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna"
)
TARGET_REVISION = "8efa61729e24bd65b1d152b5ab5409052aa80e65"
TARGET = (
    BASE
    / "hf_home/hub/models--Qwen--Qwen3-235B-A22B/snapshots"
    / TARGET_REVISION
)
CONTAINER = BASE / "containers/nemo_rl_nightly_20260909_7023221.sqsh"
ARTIFACTS = BASE / "experiments/q235-rp25-perf-20260917"
SHARED_MEGATRON_CHECKPOINT = ARTIFACTS / "shared-megatron-initial-checkpoint"


def configuration(steps: int = 20) -> dict[str, str]:
    """Return the minimal overrides on top of the official 16n4g recipe."""
    if steps != 20:
        raise ValueError("this controlled baseline supports exactly 20 steps")
    return {
        "grpo.max_num_steps": str(steps),
        "checkpointing.enabled": "false",
        "policy.model_name": str(TARGET),
        "policy.tokenizer.name": str(TARGET),
        "policy.precision": "bfloat16",
        "policy.draft.enabled": "false",
        "policy.generation.vllm_cfg.enforce_eager": "false",
        "policy.generation.vllm_kwargs.moe_backend": "flashinfer_trtllm",
        "policy.generation.vllm_kwargs.compilation_config.cudagraph_mode": (
            "FULL_AND_PIECEWISE"
        ),
        (
            "policy.generation.vllm_kwargs.compilation_config."
            "cudagraph_capture_sizes"
        ): "[1,2,4,8,16,32,64]",
        "policy.generation.vllm_kwargs.speculative_config": "null",
        "logger.wandb_enabled": "true",
        "logger.tensorboard_enabled": "false",
        "logger.wandb.project": "sna-specdec",
        "logger.wandb.group": "q235-rp25-frozen-perf-20260917",
    }


def render(
    account: str,
    run_name: str,
    *,
    steps: int = 20,
    directory: Path | None = None,
) -> str:
    """Render one OCI-HSG batch job without mutating scheduler state."""
    if not re.fullmatch(r"[a-z0-9_]+", account):
        raise ValueError("invalid account")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_name):
        raise ValueError("invalid run name")
    run_dir = directory or ARTIFACTS / run_name
    values = configuration(steps)
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
    return f"""#!/usr/bin/env bash
#SBATCH --job-name={account}.{run_name}
#SBATCH --account={account}
#SBATCH --partition=batch
#SBATCH --time=05:00:00
#SBATCH --nodes=16
#SBATCH --segment=16
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --output={run_dir}/slurm-%j.out
#SBATCH --error={run_dir}/slurm-%j.err
set -euo pipefail
export PATH=/cm/local/apps/slurm/25.11/bin:$PATH
test -n "${{WANDB_API_KEY:-}}"
test -z "$(git -C {SOURCE} status --porcelain=v1 --untracked-files=all)"
git -C {SOURCE} rev-parse HEAD >{run_dir}/source_sha.txt
git -C {SOURCE} submodule status --recursive >{run_dir}/submodules.txt
export CONTAINER={CONTAINER}
export MOUNTS=/home:/home,/lustre:/lustre,/raid:/raid
export GPUS_PER_NODE=4 DEDICATED_RAY_HEAD=0
export BASE_LOG_DIR={run_dir}
export NETRC=/home/sna/.netrc
export HF_HOME={BASE}/hf_home
export HF_DATASETS_CACHE=${{HF_HOME}}/datasets
export NRL_MEGATRON_CHECKPOINT_DIR={SHARED_MEGATRON_CHECKPOINT}
export XDG_CACHE_HOME=/raid/scratch/sna/q235-rp25/cache
export TRITON_CACHE_DIR=${{XDG_CACHE_HOME}}/triton
export TORCH_EXTENSIONS_DIR=${{XDG_CACHE_HOME}}/torch-extensions
export WANDB_DIR={run_dir}
export WANDB_CACHE_DIR=${{XDG_CACHE_HOME}}/wandb
export WANDB_CONFIG_DIR=${{XDG_CACHE_HOME}}/wandb-config
export RAY_TMPDIR=/raid/scratch/sna/r${{SLURM_JOB_ID}}
export NRL_NATIVE_TMP=/raid/scratch/sna/q235-${{SLURM_JOB_ID}}/tmp
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset PYTHONPATH PYTHONOPTIMIZE NRL_FORCE_REBUILD_VENVS TMPDIR
export SETUP_COMMAND='set -euo pipefail
mkdir -p "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$RAY_TMPDIR" "$NRL_NATIVE_TMP"
test -d "$NRL_MEGATRON_CHECKPOINT_DIR"
test -w "$NRL_MEGATRON_CHECKPOINT_DIR"
touch "$NRL_MEGATRON_CHECKPOINT_DIR/.mount-check-$(hostname)"
test -x /opt/nemo_rl_venv/bin/python
test -x /usr/local/bin/python-MegatronPolicyWorker
test -x /usr/local/bin/python-VllmGenerationWorker'
export COMMAND={shlex.quote(f'''set -euo pipefail
cd /opt/nemo-rl
unset PYTHONPATH PYTHONOPTIMIZE NRL_FORCE_REBUILD_VENVS
export TMPDIR=$NRL_NATIVE_TMP
echo SOURCE_MODE=container-native
git rev-parse HEAD || true
sha256sum {RECIPE} >{run_dir}/container_recipe_sha256.txt
/opt/nemo_rl_venv/bin/python -c 'import sys,nemo_rl; print(sys.executable, nemo_rl.__file__)'
/usr/local/bin/python-MegatronPolicyWorker -c 'import os; from pathlib import Path; from nemo_rl.models.policy.utils import get_megatron_checkpoint_dir; p = Path(get_megatron_checkpoint_dir()); assert str(p) == os.environ["NRL_MEGATRON_CHECKPOINT_DIR"]; markers = list(p.glob(".mount-check-*")); assert len(markers) == 16, markers; print("SHARED_CHECKPOINT_PREFLIGHT_OK", p, len(markers))'
exec {command}''')}
exec bash {SOURCE}/ray.sub
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--render", action="store_true")
    mode.add_argument("--test-only", action="store_true")
    mode.add_argument("--submit", action="store_true")
    parser.add_argument("--account", default="coreai_dlalgo_nemorl")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"Qwen3-235B-Baseline-20step-{stamp}"
    directory = ARTIFACTS / run_name
    script = render(
        account=args.account,
        run_name=run_name,
        steps=args.steps,
        directory=directory,
    )
    if args.render:
        print(script)
        return

    required = (
        SOURCE / "ray.sub",
        CONTAINER,
        TARGET / "config.json",
        TARGET / "model.safetensors.index.json",
    )
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
    SHARED_MEGATRON_CHECKPOINT.mkdir(parents=True, exist_ok=True)
    (directory / "overrides.json").write_text(
        json.dumps(configuration(args.steps), indent=2) + "\n"
    )
    job = directory / "job.sbatch"
    job.write_text(script)
    job.chmod(0o700)
    result = subprocess.run(
        ["sbatch", "--test-only", str(job)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (directory / "test-only.txt").write_text(result.stdout)
    print(result.stdout, end="", flush=True)
    result.check_returncode()
    if args.submit:
        result = subprocess.run(
            ["sbatch", "--parsable", str(job)],
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
