#!/usr/bin/env python3
"""Submit the Qwen3-30B-A3B mixed-math Speculators chain past 200k.

The 50k->200k jobs have several fallback branches.  This helper is meant to be
submitted with afterok dependencies on each possible upstream branch.  The first
successful upstream branch that reaches this script takes an atomic lock, submits
the next 50k GPU training job plus an afternotok continuation, and then submits
the same helper for the following stage behind both possible success paths.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(
    os.environ.get(
        "REPO_ROOT",
        "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/Nemo-RL_Qwen3_Roadmap",
    )
)
ARTIFACT_ROOT = Path(
    os.environ.get(
        "ARTIFACT_ROOT",
        "/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3",
    )
)

PIPELINE_SCRIPT = (
    REPO_ROOT / "experiments/eagle3_qwen3_235b/slurm_speculators_offline_pipeline.sbatch"
)
THIS_SCRIPT = REPO_ROOT / "experiments/eagle3_qwen3_235b/submit_30ba3b_mixed_stage_chain.py"

ACCOUNT = os.environ.get("ACCOUNT", "coreai_dlalgo_nemorl")
PARTITION = os.environ.get("PARTITION", "batch")
TIME_LIMIT = os.environ.get("TIME_LIMIT", "04:00:00")
GPUS_PER_NODE = os.environ.get("GPUS_PER_NODE", "4")

MODEL = "Qwen/Qwen3-30B-A3B"
SEQ_LENGTH = "8192"
TARGET_LAYERS = "1 23 44"
LAYERS_TAG = "layers48_mlen8193"


def run(cmd: list[str], *, env: dict[str, str] | None = None, dry_run: bool = False) -> str:
    print("+", " ".join(cmd), flush=True)
    if dry_run:
        return "Submitted batch job DRYRUN"
    out = subprocess.check_output(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stderr=subprocess.STDOUT,
    )
    print(out.strip(), flush=True)
    return out


def parse_job_id(sbatch_output: str) -> str:
    for line in sbatch_output.splitlines():
        parts = line.strip().split()
        if len(parts) >= 4 and parts[:3] == ["Submitted", "batch", "job"]:
            return parts[3]
    raise RuntimeError(f"could not parse sbatch job id from: {sbatch_output!r}")


def line_count(path: Path) -> int:
    count = 0
    with path.open("rb") as f:
        for _ in f:
            count += 1
    return count


def source_for_offset(offset: int) -> Path:
    if offset == 0:
        return ARTIFACT_ROOT / "data/mixed_math_nonopenmath_qwen3_30ba3b_conversations_50k.jsonl"
    return (
        ARTIFACT_ROOT
        / f"data/mixed_math_nonopenmath_qwen3_30ba3b_conversations_50k_offset{offset}.jsonl"
    )


def stage_root(seen: int) -> Path:
    return ARTIFACT_ROOT / f"speculators/eagle3_qwen3_30ba3b_mixed_math_nonopenmath_{seen // 1000}k_seen"


def checkpoint_for_seen(seen: int) -> Path:
    if seen == 50_000:
        return (
            ARTIFACT_ROOT
            / "speculators/eagle3_qwen3_30ba3b_mixed_math_nonopenmath_50k/checkpoints_train_50k_layers48_mlen8193/0"
        )
    previous_seen = seen - 50_000
    return (
        stage_root(seen)
        / f"checkpoints_from{previous_seen // 1000}k_plus50k_layers48_mlen8193/0"
    )


def denylist_for_offset(offset: int) -> list[Path]:
    denylist = [ARTIFACT_ROOT / "data/openmath_reasoning_cot_conversations_50k.jsonl"]
    denylist.append(source_for_offset(0))
    for earlier in range(50_000, offset, 50_000):
        denylist.append(source_for_offset(earlier))
    return denylist


def validate_ready(stage_seen: int) -> tuple[Path, Path, list[Path]]:
    if stage_seen < 200_000 or stage_seen > 500_000 or stage_seen % 50_000:
        raise ValueError("--stage-seen must be one of 200000, 250000, ..., 500000")

    source_offset = stage_seen - 50_000
    previous_checkpoint = checkpoint_for_seen(stage_seen - 50_000)
    source = source_for_offset(source_offset)
    denylist = denylist_for_offset(source_offset)

    missing = [p for p in [previous_checkpoint, source, *denylist] if not p.exists()]
    if missing:
        raise FileNotFoundError("missing prerequisites:\n" + "\n".join(str(p) for p in missing))

    rows = line_count(source)
    if rows != 50_000:
        raise RuntimeError(f"source slice is not exactly 50000 rows: {source} rows={rows}")

    return source, previous_checkpoint, denylist


def acquire_lock(stage_seen: int, trigger_job: str) -> Path | None:
    lock_root = ARTIFACT_ROOT / "stage_submit_locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_dir = lock_root / f"qwen3_30ba3b_mixed_{stage_seen // 1000}k"
    try:
        lock_dir.mkdir()
    except FileExistsError:
        print(f"stage {stage_seen} already claimed by {lock_dir}; exiting", flush=True)
        return None
    (lock_dir / "claimed_by.txt").write_text(
        f"trigger_job={trigger_job}\ntime={time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
    )
    return lock_dir


def stage_env(stage_seen: int, source: Path, previous_checkpoint: Path, denylist: list[Path]) -> dict[str, str]:
    source_offset = stage_seen - 50_000
    previous_seen = stage_seen - 50_000
    out_dir = stage_root(stage_seen)
    return {
        **os.environ,
        "REPO_ROOT": str(REPO_ROOT),
        "ARTIFACT_ROOT": str(ARTIFACT_ROOT),
        "MODEL": MODEL,
        "SOURCE_CONVERSATIONS": str(source),
        "SPECULATORS_JSONL": str(
            ARTIFACT_ROOT
            / f"data/mixed_math_nonopenmath_qwen3_30ba3b_conversations_50k_offset{source_offset}_speculators.jsonl"
        ),
        "OUTPUT_DIR": str(out_dir),
        "HIDDEN_STATES_DIR": str(out_dir / f"hidden_states_{LAYERS_TAG}_offset{source_offset // 1000}k"),
        "CHECKPOINT_DIR": str(
            out_dir / f"checkpoints_from{previous_seen // 1000}k_plus50k_layers48_mlen8193"
        ),
        "VLLM_TMP_HIDDEN_STATES": str(
            out_dir / f"vllm_tmp_hidden_states_{LAYERS_TAG}_offset{source_offset // 1000}k"
        ),
        "FROM_PRETRAINED": str(previous_checkpoint),
        "SEQ_LENGTH": SEQ_LENGTH,
        "VLLM_MAX_MODEL_LEN": SEQ_LENGTH,
        "MAX_SAMPLES": "50000",
        "SAMPLE_OFFSET": "0",
        "MIN_HIDDEN_STATES": "50000",
        "MINIMUM_VALID_TOKENS": "1",
        "TARGET_LAYER_IDS": TARGET_LAYERS,
        "DRAFT_VOCAB_SIZE": "32000",
        "SPECULATOR_TYPE": "eagle3",
        "EPOCHS": "1",
        "LR": "5e-5",
        "TTT_STEPS": "3",
        "TTT_STEP_LOSS_DECAY": "1.0",
        "BLOCK_SIZE": "8",
        "MAX_ANCHORS": "256",
        "NUM_LAYERS": "1",
        "DRAFT_ARCH": "llama",
        "NUM_TRAIN_GPUS": "4",
        "VLLM_TP": "4",
        "VLLM_DP": "1",
        "VLLM_GPU_UTIL": "0.85",
        "CONCURRENCY": "16",
        "REQUEST_TIMEOUT": "240",
        "MAX_RETRIES": "3",
        "VLLM_EXTRA_ARGS": "--attention-backend TRITON_ATTN --max-num-seqs 1 --max-cudagraph-capture-size 1 --disable-custom-all-reduce",
        "RUN_CLONE": "false",
        "RUN_CONVERT": "true",
        "RUN_PREPARE": "true",
        "RUN_DATAGEN": "true",
        "RUN_TRAIN": "true",
        "VALIDATE_OUTPUTS": "true",
        "VALIDATE_SOURCE_CONVERSATIONS": "true",
        "FAIL_ON_DUPLICATE_PROMPTS": "true",
        "DENYLIST_PROMPTS_FROM": " ".join(str(p) for p in denylist),
        "INSTALL_SPECULATORS": "true",
        "APPLY_COMPAT_PATCHES": "true",
        "SPECULATORS_DISABLE_TORCH_COMPILE": "false",
        "SPECULATORS_FSDP_WRAP_LAYERS": "true",
    }


def submit_gpu_stage(stage_seen: int, env: dict[str, str], dry_run: bool) -> tuple[str, str]:
    name = f"qwen3_30ba3b-speculators-mixed-nonopenmath-{stage_seen // 1000}k-auto"
    common = [
        "sbatch",
        "--nodes=1",
        f"--account={ACCOUNT}",
        f"--partition={PARTITION}",
        f"--gres=gpu:{GPUS_PER_NODE}",
        f"--time={TIME_LIMIT}",
        "--mem=0",
    ]
    main_out = run(common + [f"--job-name={name}", str(PIPELINE_SCRIPT)], env=env, dry_run=dry_run)
    main_id = parse_job_id(main_out)
    cont_out = run(
        common
        + [
            f"--job-name={name}-cont",
            f"--dependency=afternotok:{main_id}",
            str(PIPELINE_SCRIPT),
        ],
        env=env,
        dry_run=dry_run,
    )
    cont_id = parse_job_id(cont_out)
    return main_id, cont_id


def submit_next_selectors(stage_seen: int, upstream_ids: tuple[str, str], dry_run: bool) -> list[str]:
    next_stage = stage_seen + 50_000
    if next_stage > 500_000:
        return []

    selector_ids: list[str] = []
    for upstream in upstream_ids:
        name = f"qwen3_30ba3b-submit-{next_stage // 1000}k-after-{upstream}"
        wrap = (
            f"cd {REPO_ROOT} && "
            f"python3 {THIS_SCRIPT} --stage-seen {next_stage} --trigger-job {upstream}"
        )
        out = run(
            [
                "sbatch",
                "--nodes=1",
                "--ntasks=1",
                f"--account={ACCOUNT}",
                f"--partition={PARTITION}",
                "--time=00:10:00",
                f"--job-name={name}",
                f"--dependency=afterok:{upstream}",
                "--output=logs/%x_%j.out",
                "--error=logs/%x_%j.err",
                "--wrap",
                wrap,
            ],
            dry_run=dry_run,
        )
        selector_ids.append(parse_job_id(out))
    return selector_ids


def write_manifest(
    lock_dir: Path,
    stage_seen: int,
    trigger_job: str,
    source: Path,
    previous_checkpoint: Path,
    main_id: str,
    cont_id: str,
    selector_ids: list[str],
) -> None:
    manifest = {
        "stage_seen": stage_seen,
        "trigger_job": trigger_job,
        "source": str(source),
        "previous_checkpoint": str(previous_checkpoint),
        "main_job": main_id,
        "continuation_job": cont_id,
        "next_selector_jobs": selector_ids,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    (lock_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-seen", type=int, required=True)
    parser.add_argument("--trigger-job", default=os.environ.get("SLURM_JOB_ID", "manual"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source, previous_checkpoint, denylist = validate_ready(args.stage_seen)
    if args.dry_run:
        env = stage_env(args.stage_seen, source, previous_checkpoint, denylist)
        main_id, cont_id = submit_gpu_stage(args.stage_seen, env, args.dry_run)
        submit_next_selectors(args.stage_seen, (main_id, cont_id), args.dry_run)
        return 0

    lock_dir = acquire_lock(args.stage_seen, args.trigger_job)
    if lock_dir is None:
        return 0

    try:
        env = stage_env(args.stage_seen, source, previous_checkpoint, denylist)
        main_id, cont_id = submit_gpu_stage(args.stage_seen, env, args.dry_run)
        selector_ids = submit_next_selectors(args.stage_seen, (main_id, cont_id), args.dry_run)
        write_manifest(
            lock_dir,
            args.stage_seen,
            args.trigger_job,
            source,
            previous_checkpoint,
            main_id,
            cont_id,
            selector_ids,
        )
    except Exception:
        shutil.rmtree(lock_dir, ignore_errors=True)
        raise
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
