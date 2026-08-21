#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


HORIZON_STEPS = 1000


def latest_step(root: Path) -> int:
    steps = sorted(
        int(match.group(1))
        for path in root.glob("step_*")
        if path.is_dir() and (match := re.fullmatch(r"step_(\d+)", path.name))
    )
    if not steps:
        raise ValueError(f"no checkpoint exists under {root}")
    return steps[-1]


def validate_checkpoint(root: Path, *, expected_step: int) -> Path:
    if list(root.glob("tmp_step_*")):
        raise ValueError(f"temporary checkpoint exists under {root}")
    if latest_step(root) != expected_step:
        raise ValueError(f"latest checkpoint is not step_{expected_step}")
    step_dir = root / f"step_{expected_step}"
    for path in (
        step_dir / "training_info.json",
        step_dir / "train_dataloader.pt",
        step_dir / "policy" / "weights" / "latest_train_state.pt",
        step_dir / "config.yaml",
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"checkpoint component is missing: {path}")
    info = json.loads((step_dir / "training_info.json").read_text())
    expected = {
        "current_step": expected_step,
        "total_steps": expected_step,
        "consumed_samples": 8 * expected_step,
    }
    for key, value in expected.items():
        if info.get(key) != value:
            raise ValueError(f"training_info.{key} must be {value}")
    match = re.search(
        r"(?m)^\s{2}max_num_steps:\s*(\d+)\s*$",
        (step_dir / "config.yaml").read_text(),
    )
    if match is None or int(match.group(1)) != HORIZON_STEPS:
        raise ValueError("checkpoint must preserve max_num_steps=1000")
    return step_dir


def validate_progress(previous_step: int, current_step: int, minimum: int) -> None:
    if not previous_step < current_step <= HORIZON_STEPS:
        raise ValueError(f"invalid progress {previous_step}->{current_step}")
    if current_step < minimum:
        raise ValueError(f"segment missed milestone {minimum}: reached {current_step}")


def manifest_identity(
    *,
    git_sha: str,
    checkpoint_root: Path,
    target_revision: str,
    drafter_revision: str,
    container_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "git_sha": git_sha,
        "checkpoint_root": str(checkpoint_root.resolve()),
        "target_revision": target_revision,
        "drafter_revision": drafter_revision,
        "container_sha256": container_sha256,
        "training_horizon_steps": HORIZON_STEPS,
        "speculator_type": "dspark",
        "num_speculative_tokens": 7,
        "draft_training_enabled": True,
        "draft_refit_enabled": True,
    }


def write_manifest(path: Path, *, wandb_run_id: str, **identity: object) -> None:
    payload = {**manifest_identity(**identity), "wandb_run_id": wandb_run_id}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_manifest(path: Path, **identity: object) -> dict[str, object]:
    payload = json.loads(path.read_text())
    for key, value in manifest_identity(**identity).items():
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
    parser.add_argument("--print-manifest-wandb-id", action="store_true")
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
            args.previous_step, args.expected_step, args.required_min_step
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
    if args.validate_manifest or args.print_manifest_wandb_id:
        if args.manifest is None:
            raise ValueError("manifest path is required")
        payload = validate_manifest(args.manifest, **identity)
        if args.print_manifest_wandb_id:
            print(payload["wandb_run_id"])


if __name__ == "__main__":
    main()
