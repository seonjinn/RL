#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.qwen3_8b_dflash_online_cp1.resume_contract import (
    HORIZON_STEPS,
    ORACLE_RUN_ID,
    latest_step,
    validate_checkpoint,
    validate_progress,
)


def write_manifest(
    path: Path,
    *,
    git_sha: str,
    checkpoint_root: Path,
    wandb_run_id: str,
    target_revision: str,
    drafter_revision: str,
    container_sha256: str,
) -> None:
    payload = {
        "schema_version": 1,
        "git_sha": git_sha,
        "checkpoint_root": str(checkpoint_root.resolve()),
        "wandb_run_id": wandb_run_id,
        "oracle_run_id": ORACLE_RUN_ID,
        "target_revision": target_revision,
        "drafter_revision": drafter_revision,
        "container_sha256": container_sha256,
        "training_horizon_steps": HORIZON_STEPS,
        "dflash_k": 7,
        "draft_training_enabled": False,
        "draft_refit_enabled": False,
        "fixed_public_drafter": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_manifest(
    path: Path,
    *,
    git_sha: str,
    checkpoint_root: Path,
    target_revision: str,
    drafter_revision: str,
    container_sha256: str,
) -> dict[str, object]:
    payload = json.loads(path.read_text())
    expected = {
        "schema_version": 1,
        "git_sha": git_sha,
        "checkpoint_root": str(checkpoint_root.resolve()),
        "oracle_run_id": ORACLE_RUN_ID,
        "target_revision": target_revision,
        "drafter_revision": drafter_revision,
        "container_sha256": container_sha256,
        "training_horizon_steps": HORIZON_STEPS,
        "dflash_k": 7,
        "draft_training_enabled": False,
        "draft_refit_enabled": False,
        "fixed_public_drafter": True,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"manifest {key} mismatch")
    if not isinstance(payload.get("wandb_run_id"), str) or not payload["wandb_run_id"]:
        raise ValueError("manifest requires a fresh W&B run ID")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--previous-step", type=int)
    parser.add_argument("--required-min-step", type=int)
    parser.add_argument("--print-latest-step", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--git-sha")
    parser.add_argument("--wandb-run-id")
    parser.add_argument("--target-revision")
    parser.add_argument("--drafter-revision")
    parser.add_argument("--container-sha256")
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--validate-manifest", action="store_true")
    args = parser.parse_args()
    if args.print_latest_step:
        print(latest_step(args.checkpoint_dir))
        return
    if args.expected_step is not None:
        validate_checkpoint(args.checkpoint_dir, expected_step=args.expected_step)
    if args.previous_step is not None:
        if args.expected_step is None or args.required_min_step is None:
            raise ValueError("progress validation requires current and milestone")
        validate_progress(
            args.previous_step,
            args.expected_step,
            required_min_step=args.required_min_step,
        )
    identity = {
        "git_sha": args.git_sha,
        "checkpoint_root": args.checkpoint_dir,
        "target_revision": args.target_revision,
        "drafter_revision": args.drafter_revision,
        "container_sha256": args.container_sha256,
    }
    if args.write_manifest:
        if args.manifest is None or args.wandb_run_id is None:
            raise ValueError("manifest path and W&B run ID are required")
        write_manifest(args.manifest, wandb_run_id=args.wandb_run_id, **identity)
    if args.validate_manifest:
        if args.manifest is None:
            raise ValueError("manifest path is required")
        validate_manifest(args.manifest, **identity)


if __name__ == "__main__":
    main()
