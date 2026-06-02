#!/usr/bin/env python3
"""Submit one-sample hidden-state repairs for Qwen3-235B Speculators 500K.

Use this after the 100-shard hidden-state array completes but coverage
validation reports a small set of missing ``hs_<index>.safetensors`` files.
Each missing id is submitted as a DATAGEN_START_INDEX=id, DATAGEN_END_INDEX=id+1
job, then a replacement train job is submitted after all repairs finish.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import submit_qwen235b_mixed_500k_speculators_after_finalize as base


def parse_missing_ids(values: list[str]) -> list[int]:
    ids: list[int] = []
    for value in values:
        for item in value.replace(",", " ").split():
            ids.append(int(item))
    unique = sorted(set(ids))
    if len(unique) != len(ids):
        raise ValueError(f"duplicate missing ids requested: {ids}")
    bad = [idx for idx in unique if idx < 0 or idx >= 500_000]
    if bad:
        raise ValueError(f"missing ids must be in [0, 500000): {bad}")
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("missing_ids", nargs="+", help="Missing numeric hidden-state ids, comma or space separated.")
    parser.add_argument("--datagen-time", default="01:00:00")
    parser.add_argument("--datagen-nodes", type=int, default=2)
    parser.add_argument("--train-time", default="04:00:00")
    parser.add_argument("--job-suffix", default="-missing4-repair")
    parser.add_argument("--vllm-site", default=str(base.ARTIFACT_ROOT / "python_site/vllm_0_17_0_extract_py312"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    missing_ids = parse_missing_ids(args.missing_ids)
    paths = base.paths()
    base.validate_inputs(paths, allow_pending_finalizer=False)
    if not args.dry_run:
        base.validate_prepared_outputs(paths)
        base.validate_hidden_state_manifest_reuse_safety(paths)
        (base.REPO_ROOT / "logs").mkdir(parents=True, exist_ok=True)
        paths["report"].parent.mkdir(parents=True, exist_ok=True)

    ray_sub = base.materialize_ray_submit_script(dry_run=args.dry_run)
    common = base.common_env(paths)

    repair_jobs: dict[str, str] = {}
    for idx in missing_ids:
        repair_jobs[str(idx)] = base.submit_ray_datagen(
            shard=idx,
            start=idx,
            end=idx + 1,
            dependency="",
            env=common,
            datagen_time=args.datagen_time,
            num_nodes=args.datagen_nodes,
            vllm_site=Path(args.vllm_site),
            ray_sub=ray_sub,
            dry_run=args.dry_run,
        )

    train_env = {
        **common,
        "RUN_CONVERT": "false",
        "RUN_PREPARE": "false",
        "RUN_DATAGEN": "false",
        "RUN_TRAIN": "true",
        "VALIDATE_SOURCE_CONVERSATIONS": "false",
        "VALIDATE_HIDDEN_STATE_COVERAGE": "true",
    }
    dependency = "afterok:" + ":".join(repair_jobs.values())
    train_id = base.submit_sbatch_pipeline(
        name=f"qwen3_235b-speculators-mixed-500k-train{args.job_suffix}",
        dependency=dependency,
        time_limit=args.train_time,
        env=train_env,
        dry_run=args.dry_run,
    )

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "mode": "qwen3_235b_missing_hidden_state_repair",
        "missing_ids": missing_ids,
        "repair_jobs": repair_jobs,
        "replacement_train_job": train_id,
        "replacement_train_dependency": dependency,
        "superseded_train_job": "3101473",
        "hidden_states_dir": str(paths["hidden_states_dir"]),
        "checkpoint_dir": str(paths["checkpoint_dir"]),
        "ray_sub": str(ray_sub),
        "dry_run": args.dry_run,
    }
    if not args.dry_run:
        report = base.ARTIFACT_ROOT / "reports/qwen3_235b_500k_hidden_state_missing4_repair_summary.json"
        report.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
