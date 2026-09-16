"""Render or submit the isolated Qwen3.5 BF16/graph Math gate and cohort."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

SOURCE = Path("/home/sna/nemorl-q35-math-specdec-20260916")
EXPERIMENT = Path("experiments/q35_math_specdec_20260916")
RECIPE = Path(
    "examples/configs/recipes/llm/grpo-qwen3.5-35ba3b-2n8g-megatron-ep16tp2cp2.yaml"
)
BASE = Path("/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna")
TARGET = (
    BASE
    / "hf_home/hub/models--Qwen--Qwen3.5-35B-A3B-Base/snapshots/0f0813072d2358973511097385626f21fcb6d422"
)
CONTAINER = BASE / "containers/nemo_rl_nightly_20260909_7023221.sqsh"
ARTIFACTS = BASE / "experiments/q35-math-specdec-20260916"
VLLM_PYTHON = "/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python"


def configuration(arm: str, concurrency: str, steps: int) -> dict[str, str]:
    widths = (1,) if arm == "baseline" else ((1, 6) if arm == "dflash" else (1, 5, 6))
    sizes = sorted({w * r for w in widths for r in (1, 2, 4, 8, 16, 32, 64, 128)})
    values = {
        "grpo.max_num_steps": str(steps),
        "grpo.val_at_start": "false",
        "grpo.val_at_end": "false",
        "grpo.val_period": "0",
        "checkpointing.enabled": "false",
        "policy.model_name": str(TARGET),
        "policy.tokenizer.name": str(TARGET),
        "policy.precision": "bfloat16",
        "policy.draft.enabled": "false",
        "policy.generation.vllm_cfg.enforce_eager": "false",
        "policy.generation.vllm_kwargs.moe_backend": "flashinfer_trtllm",
        "policy.generation.vllm_kwargs.compilation_config.cudagraph_mode": "FULL_AND_PIECEWISE",
        "policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes": json.dumps(
            sizes, separators=(",", ":")
        ),
        "policy.generation.vllm_kwargs.kernel_config.enable_flashinfer_autotune": "false",
        "cluster.num_nodes": "4",
        "cluster.gpus_per_node": "4",
        "cluster.segment_size": "4",
        "logger.wandb_enabled": "true",
        "logger.wandb.project": "sna-specdec",
        "logger.wandb.group": "q35-math-bf16-fap-20260916",
    }
    if concurrency != "default":
        values["policy.generation.vllm_kwargs.max_num_seqs"] = concurrency
    prefix = "policy.generation.vllm_kwargs.speculative_config"
    if arm == "baseline":
        values[prefix] = "null"
    else:
        draft = (
            BASE
            / f"specdec_ptv23/ptv3_swa/sd2p3rp-q35-a3b-ptv3rp25-{arm}-b8-16n/exported-checkpoint-44000"
        )
        values.update(
            {
                prefix + ".method": arm,
                prefix + ".model": str(draft),
                prefix + ".num_speculative_tokens": "5",
                prefix + ".draft_tensor_parallel_size": "1",
                prefix + ".attention_backend": "FLASH_ATTN",
            }
        )
    return values


def render(
    arm: str, concurrency: str, steps: int, account: str, directory: Path
) -> str:
    values = configuration(arm, concurrency, steps)
    values["logger.wandb.name"] = directory.name
    values["logger.log_dir"] = str(directory / "logs")
    # Resolve only the staged paths from the job environment, not from the host.
    for key in ("policy.model_name", "policy.tokenizer.name"):
        values[key] = "${oc.env:Q35_NODE_ROOT}/target"
    if arm != "baseline":
        values["policy.generation.vllm_kwargs.speculative_config.model"] = (
            "${oc.env:Q35_NODE_ROOT}/draft"
        )
    command = shlex.join(
        [
            "/opt/nemo_rl_venv/bin/python",
            "examples/run_grpo.py",
            "--config",
            str(SOURCE / RECIPE),
            *[f"++{k}={v}" for k, v in values.items()],
        ]
    )
    walltime = "04:00:00" if steps <= 3 else "08:00:00"
    partition = "batch" if steps <= 3 else "batch_long"
    return f"""#!/usr/bin/env bash
#SBATCH --job-name={account}.{directory.name}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --time={walltime}
#SBATCH --nodes=4
#SBATCH --segment=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --output={directory}/slurm-%j.out
#SBATCH --error={directory}/slurm-%j.err
set -euo pipefail
export PATH=/cm/local/apps/slurm/25.11/bin:$PATH
test -n "${{WANDB_API_KEY:-}}"
test -z "$(git -C {SOURCE} status --porcelain=v1 --untracked-files=all)"
git -C {SOURCE} rev-parse HEAD >{directory}/source_sha.txt
git -C {SOURCE} submodule status --recursive >{directory}/submodules.txt
export CONTAINER={CONTAINER}
export MOUNTS=/lustre:/lustre,/home:/home,/raid:/raid
export GPUS_PER_NODE=4 CPUS_PER_WORKER=64
export BASE_LOG_DIR={directory}
export RAY_TMPDIR=/raid/scratch/sna/r${{SLURM_JOB_ID}}
export Q35_NODE_ROOT=/raid/scratch/sna/q35-${{SLURM_JOB_ID}}
export Q35_SOURCE={SOURCE} Q35_TARGET={TARGET} Q35_ARM={arm}
export Q35_ARTIFACT_DIR={directory}
export HF_HOME="${{Q35_NODE_ROOT}}/hf" XDG_CACHE_HOME="${{Q35_NODE_ROOT}}/cache"
export TMPDIR="${{Q35_NODE_ROOT}}/tmp"
export PYTHONPATH="${{Q35_NODE_ROOT}}/vllm-overlay:${{Q35_NODE_ROOT}}/mcore-overlay:{SOURCE}:${{PYTHONPATH:-}}"
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=PYTHONPATH,Q35_NODE_ROOT,HF_HOME,XDG_CACHE_HOME,TMPDIR
export SETUP_COMMAND='bash {SOURCE / EXPERIMENT}/setup_node.sh'
export COMMAND={shlex.quote(f"cd {SOURCE} && " + command)}
exec bash {SOURCE}/ray.sub
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("config-json", "render", "test-only", "submit"):
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("arm", choices=("baseline", "dflash", "dspark"))
    parser.add_argument("concurrency", choices=("default", "64"))
    parser.add_argument("--steps", type=int, choices=(3, 20), default=3)
    parser.add_argument("--account", default="coreai_dlalgo_llm")
    parser.add_argument("--confirm-base-target-lineage", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9_]+", args.account):
        parser.error("invalid account")
    if (
        (args.submit or args.test_only)
        and args.arm != "baseline"
        and not args.confirm_base_target_lineage
    ):
        parser.error(
            "confirm drafter target lineage is Qwen3.5-35B-A3B-Base before submission"
        )
    config = configuration(args.arm, args.concurrency, args.steps)
    if args.config_json:
        print(json.dumps(config))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = {
        "baseline": "Baseline",
        "dflash": "DFlashK5-frozen",
        "dspark": "DSparkK5-frozen",
    }[args.arm]
    directory = (
        ARTIFACTS
        / f"Qwen3.5-35BA3B-{label}-S{args.concurrency}-{args.steps}step-{stamp}"
    )
    script = render(args.arm, args.concurrency, args.steps, args.account, directory)
    if args.render:
        print(script)
        return
    for path in (
        SOURCE / RECIPE,
        CONTAINER,
        TARGET / "config.json",
        TARGET / "model.safetensors.index.json",
    ):
        if not path.is_file():
            parser.error(f"missing required input: {path}")
    status = subprocess.check_output(
        ["git", "-C", str(SOURCE), "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
    )
    if status:
        parser.error("source checkout is dirty")
    if not os.environ.get("WANDB_API_KEY"):
        parser.error("WANDB_API_KEY is not exported")
    if args.arm != "baseline":
        draft = Path(config["policy.generation.vllm_kwargs.speculative_config.model"])
        if not (draft / "model.safetensors").is_file():
            parser.error("drafter weight file is missing")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "overrides.json").write_text(json.dumps(config, indent=2) + "\n")
    job = directory / "job.sbatch"
    job.write_text(script)
    job.chmod(0o700)
    for flags, filename in ((["--test-only"], "test-only.txt"), ([], "submission.txt")):
        if not flags and not args.submit:
            break
        result = subprocess.run(
            ["sbatch", *flags, str(job)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (directory / filename).write_text(result.stdout)
        print(result.stdout, end="", flush=True)
        result.check_returncode()
    print(f"Artifacts: {directory}")


if __name__ == "__main__":
    main()
