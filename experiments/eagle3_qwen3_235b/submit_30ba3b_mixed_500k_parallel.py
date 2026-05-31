#!/usr/bin/env python3
"""Submit a parallel 500K Speculators EAGLE3 build for Qwen3-30B-A3B.

This is separate from the conservative 50K continuation chain.  It builds one
500K prepared dataset, generates hidden states by global-index shards, then
trains once over the full 500K cache.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
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
UNIQUE_BUILDER = (
    REPO_ROOT / "experiments/eagle3_qwen3_235b/build_unique_training_conversations.py"
)

ACCOUNT = os.environ.get("ACCOUNT", "coreai_dlalgo_nemorl")
PARTITION = os.environ.get("PARTITION", "batch")
GPUS_PER_NODE = os.environ.get("GPUS_PER_NODE", "4")
MERGE_GPUS = os.environ.get("MERGE_GPUS", GPUS_PER_NODE)

MODEL = "Qwen/Qwen3-30B-A3B"
SEQ_LENGTH = "8192"
TARGET_LAYERS = "1 23 44"
LAYERS_TAG = "layers48_mlen8193"


def run(cmd: list[str], *, env: dict[str, str] | None = None, dry_run: bool = False) -> str:
    print("+", " ".join(cmd), flush=True)
    if dry_run:
        return "Submitted batch job DRYRUN"
    try:
        out = subprocess.check_output(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        if exc.output:
            print(exc.output, flush=True)
        raise
    print(out.strip(), flush=True)
    return out


def parse_job_id(sbatch_output: str) -> str:
    for line in sbatch_output.splitlines():
        parts = line.strip().split()
        if len(parts) >= 4 and parts[:3] == ["Submitted", "batch", "job"]:
            return parts[3]
    raise RuntimeError(f"could not parse sbatch job id from: {sbatch_output!r}")


def source_for_offset(offset: int) -> Path:
    if offset == 0:
        return ARTIFACT_ROOT / "data/mixed_math_nonopenmath_qwen3_30ba3b_conversations_50k.jsonl"
    return (
        ARTIFACT_ROOT
        / f"data/mixed_math_nonopenmath_qwen3_30ba3b_conversations_50k_offset{offset}.jsonl"
    )


def line_count(path: Path) -> int:
    count = 0
    with path.open("rb") as f:
        for _ in f:
            count += 1
    return count


def default_replacement_conversations() -> Path:
    return ARTIFACT_ROOT / "data/mixed_math_nonopenmath_qwen3_30ba3b_replacement_conversations_dapo100.jsonl"


def validate_sources(replacement_conversations: Path) -> list[Path]:
    sources = [source_for_offset(offset) for offset in range(0, 500_000, 50_000)]
    missing = [p for p in sources if not p.exists()]
    if missing:
        raise FileNotFoundError("missing 50K source slices:\n" + "\n".join(map(str, missing)))
    if not UNIQUE_BUILDER.exists():
        raise FileNotFoundError(f"missing unique merge builder: {UNIQUE_BUILDER}")
    if not replacement_conversations.exists() or replacement_conversations.stat().st_size == 0:
        raise FileNotFoundError(
            "missing replacement conversations needed to make exact unique 500K:\n"
            f"{replacement_conversations}"
        )
    bad = [(p, line_count(p)) for p in sources if line_count(p) != 50_000]
    if bad:
        raise RuntimeError(
            "source slices must each have exactly 50000 rows:\n"
            + "\n".join(f"{p}: {n}" for p, n in bad)
        )
    return sources


def submit_merge_job(sources: list[Path], replacement: Path, merged: Path, dry_run: bool) -> str:
    merged.parent.mkdir(parents=True, exist_ok=True)
    summary = ARTIFACT_ROOT / "reports/mixed_math_nonopenmath_qwen3_30ba3b_500k_unique_merge_summary.json"
    primary_args = " ".join(shlex.quote(str(p)) for p in sources)
    script = (
        "set -euo pipefail; "
        f"tmp='{merged}.tmp.${{SLURM_JOB_ID:-manual}}'; rm -f \"$tmp\"; "
        f"python3 {shlex.quote(str(UNIQUE_BUILDER))} "
        f"--output \"$tmp\" "
        f"--summary-json {shlex.quote(str(summary))} "
        "--expected-count 500000 "
        f"--denylist-prompts-from {shlex.quote(str(ARTIFACT_ROOT / 'data/openmath_reasoning_cot_conversations_50k.jsonl'))} "
        f"--primary {primary_args} "
        f"--replacement {shlex.quote(str(replacement))}; "
        f"if [[ -s \"$tmp\" ]]; then mv -f \"$tmp\" '{merged}'; fi; "
        f"test -s '{merged}'; "
        f"rows=$(wc -l < '{merged}' | tr -d ' '); "
        f"test \"$rows\" = 500000; "
        f"echo merged={merged} rows=$rows"
    )
    wrap = f"bash -lc {shlex.quote(script)}"
    out = run(
        [
            "sbatch",
            "--nodes=1",
            "--ntasks=1",
            f"--account={ACCOUNT}",
            f"--partition={PARTITION}",
            f"--gres=gpu:{MERGE_GPUS}",
            "--time=00:30:00",
            "--job-name=qwen3_30ba3b-merge-mixed-500k",
            "--output=logs/%x_%j.out",
            "--error=logs/%x_%j.err",
            "--wrap",
            wrap,
        ],
        dry_run=dry_run,
    )
    return parse_job_id(out)


def common_env(root: Path, merged: Path, spec_jsonl: Path) -> dict[str, str]:
    return {
        **os.environ,
        "REPO_ROOT": str(REPO_ROOT),
        "ARTIFACT_ROOT": str(ARTIFACT_ROOT),
        "MODEL": MODEL,
        "SOURCE_CONVERSATIONS": str(merged),
        "SPECULATORS_JSONL": str(spec_jsonl),
        "OUTPUT_DIR": str(root),
        "HIDDEN_STATES_DIR": str(root / f"hidden_states_{LAYERS_TAG}_500k_global"),
        "CHECKPOINT_DIR": str(root / f"checkpoints_train_500k_{LAYERS_TAG}"),
        "VLLM_TMP_HIDDEN_STATES": str(root / f"vllm_tmp_hidden_states_{LAYERS_TAG}_500k"),
        "SEQ_LENGTH": SEQ_LENGTH,
        "VLLM_MAX_MODEL_LEN": SEQ_LENGTH,
        "MAX_SAMPLES": "0",
        "MIN_HIDDEN_STATES": "500000",
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
        "VALIDATE_OUTPUTS": "true",
        "VALIDATE_SOURCE_CONVERSATIONS": "true",
        "FAIL_ON_DUPLICATE_PROMPTS": "true",
        "DENYLIST_PROMPTS_FROM": str(
            ARTIFACT_ROOT / "data/openmath_reasoning_cot_conversations_50k.jsonl"
        ),
        "INSTALL_SPECULATORS": "true",
        "APPLY_COMPAT_PATCHES": "false",
        "SPECULATORS_DISABLE_TORCH_COMPILE": "false",
        "SPECULATORS_FSDP_WRAP_LAYERS": "true",
    }


def submit_pipeline(
    *,
    name: str,
    dependency: str,
    time_limit: str,
    env: dict[str, str],
    dry_run: bool,
) -> str:
    cmd = [
        "sbatch",
        "--nodes=1",
        f"--account={ACCOUNT}",
        f"--partition={PARTITION}",
        f"--gres=gpu:{GPUS_PER_NODE}",
        f"--time={time_limit}",
        "--mem=0",
        f"--job-name={name}",
        "--output=logs/%x_%j.out",
        "--error=logs/%x_%j.err",
    ]
    if dependency:
        cmd.append(f"--dependency={dependency}")
    cmd.append(str(PIPELINE_SCRIPT))
    return parse_job_id(run(cmd, env=env, dry_run=dry_run))


def submit_parallel(args: argparse.Namespace) -> dict[str, object]:
    (REPO_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    replacement = Path(args.replacement_conversations)
    sources = validate_sources(replacement)
    root = ARTIFACT_ROOT / "speculators/eagle3_qwen3_30ba3b_mixed_math_nonopenmath_500k_parallel"
    merged = ARTIFACT_ROOT / "data/mixed_math_nonopenmath_qwen3_30ba3b_conversations_500k_unique.jsonl"
    spec_jsonl = ARTIFACT_ROOT / "data/mixed_math_nonopenmath_qwen3_30ba3b_conversations_500k_unique_speculators.jsonl"

    merge_id = ""
    prep_dependency = ""
    if args.skip_merge_if_present and merged.exists() and line_count(merged) == 500_000:
        merge_id = "verified_existing_500k_unique"
    else:
        merge_id = submit_merge_job(sources, replacement, merged, args.dry_run)
        prep_dependency = f"afterok:{merge_id}"
    base = common_env(root, merged, spec_jsonl)
    suffix = args.job_suffix

    prep_env = {
        **base,
        "RUN_CONVERT": "true",
        "RUN_PREPARE": "true",
        "RUN_DATAGEN": "false",
        "RUN_TRAIN": "false",
    }
    prep_id = submit_pipeline(
        name=f"qwen3_30ba3b-speculators-mixed-500k-prep{suffix}",
        dependency=prep_dependency,
        time_limit=args.prep_time,
        env=prep_env,
        dry_run=args.dry_run,
    )

    shard_ids: list[str] = []
    shard_size = 500_000 // args.shards
    for shard in range(args.shards):
        start = shard * shard_size
        end = 500_000 if shard == args.shards - 1 else (shard + 1) * shard_size
        shard_env = {
            **base,
            "RUN_CONVERT": "false",
            "RUN_PREPARE": "false",
            "RUN_DATAGEN": "true",
            "RUN_TRAIN": "false",
            "DATAGEN_START_INDEX": str(start),
            "DATAGEN_END_INDEX": str(end),
            "VLLM_TMP_HIDDEN_STATES": str(
                root / f"vllm_tmp_hidden_states_{LAYERS_TAG}_500k_shard{shard:02d}"
            ),
        }
        shard_ids.append(
            submit_pipeline(
                name=f"qwen3_30ba3b-speculators-mixed-500k-hs{suffix}-{shard:02d}",
                dependency=f"afterok:{prep_id}",
                time_limit=args.datagen_time,
                env=shard_env,
                dry_run=args.dry_run,
            )
        )

    train_env = {
        **base,
        "RUN_CONVERT": "false",
        "RUN_PREPARE": "false",
        "RUN_DATAGEN": "false",
        "RUN_TRAIN": "true",
    }
    if args.from_pretrained:
        train_env["FROM_PRETRAINED"] = args.from_pretrained
    train_dep = "afterok:" + ":".join(shard_ids)
    train_id = submit_pipeline(
        name=f"qwen3_30ba3b-speculators-mixed-500k-train{suffix}",
        dependency=train_dep,
        time_limit=args.train_time,
        env=train_env,
        dry_run=args.dry_run,
    )

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "mode": "parallel_500k_global_hidden_cache",
        "model": MODEL,
        "shards": args.shards,
        "source_slices": [str(p) for p in sources],
        "replacement_conversations": str(replacement),
        "merged_conversations": str(merged),
        "speculators_jsonl": str(spec_jsonl),
        "output_dir": str(root),
        "hidden_states_dir": train_env["HIDDEN_STATES_DIR"],
        "checkpoint_dir": train_env["CHECKPOINT_DIR"],
        "merge_job": merge_id,
        "prepare_job": prep_id,
        "hidden_state_jobs": shard_ids,
        "train_job": train_id,
        "from_pretrained": args.from_pretrained or "",
        "skip_merge_if_present": args.skip_merge_if_present,
    }
    report_path = ARTIFACT_ROOT / "reports/mixed_math_nonopenmath_qwen3_30ba3b_500k_parallel_submit_summary.json"
    if not args.dry_run:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=int, default=20)
    parser.add_argument("--prep-time", default="02:00:00")
    parser.add_argument("--datagen-time", default="04:00:00")
    parser.add_argument("--train-time", default="04:00:00")
    parser.add_argument("--from-pretrained", default="")
    parser.add_argument("--replacement-conversations", default=str(default_replacement_conversations()))
    parser.add_argument("--skip-merge-if-present", action="store_true")
    parser.add_argument("--job-suffix", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.shards <= 0 or 500_000 % args.shards != 0:
        raise ValueError("--shards must be a positive divisor of 500000")
    submit_parallel(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
