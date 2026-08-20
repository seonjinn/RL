#!/usr/bin/env python3
"""Validate a time-bounded segment of one 1,000-step DFlash run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TRAINING_HORIZON_STEPS = 1000
WANDB_RUN_PATTERN = re.compile(r"wandb\.ai/[^\s]+/runs/([A-Za-z0-9_-]+)")
HORIZON_PATTERN = re.compile(r"(?m)^\s{2}max_num_steps:\s*(\d+)\s*$")


def validate_progress(
    previous_step: int,
    current_step: int,
    *,
    horizon_steps: int = TRAINING_HORIZON_STEPS,
) -> tuple[int, int]:
    """Require one segment to advance without crossing its fixed horizon."""
    if not 1 <= previous_step < current_step <= horizon_steps:
        raise ValueError(
            "resume progress must satisfy "
            f"1 <= previous < current <= {horizon_steps}; "
            f"got {previous_step} -> {current_step}"
        )
    return previous_step, current_step


def _checkpoint_steps(checkpoint_root: Path) -> list[int]:
    steps: list[int] = []
    for path in checkpoint_root.glob("step_*"):
        match = re.fullmatch(r"step_(\d+)", path.name)
        if match and path.is_dir():
            steps.append(int(match.group(1)))
    return sorted(steps)


def validate_checkpoint(
    checkpoint_root: Path,
    *,
    expected_step: int,
    expected_horizon_steps: int = TRAINING_HORIZON_STEPS,
) -> Path:
    """Validate the latest checkpoint needed by one serial resume segment."""
    if list(checkpoint_root.glob("tmp_step_*")):
        raise ValueError(f"temporary checkpoint exists under {checkpoint_root}")
    steps = _checkpoint_steps(checkpoint_root)
    if not steps or steps[-1] != expected_step:
        raise ValueError(
            f"latest checkpoint must be step_{expected_step}; found {steps or 'none'}"
        )

    step_dir = checkpoint_root / f"step_{expected_step}"
    info_path = step_dir / "training_info.json"
    dataloader_path = step_dir / "train_dataloader.pt"
    weights_path = step_dir / "policy" / "weights"
    config_path = step_dir / "config.yaml"
    for path in (info_path, dataloader_path, weights_path, config_path):
        if not path.exists():
            raise ValueError(f"checkpoint component is missing: {path}")

    optimizer_path = step_dir / "policy" / "optimizer"
    if not optimizer_path.exists():
        iteration_dir = weights_path / "iter_0000000"
        embedded_components = (
            iteration_dir / "metadata.json",
            iteration_dir / ".metadata",
            weights_path / "latest_checkpointed_iteration.txt",
            weights_path / "latest_train_state.pt",
        )
        for path in embedded_components:
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(
                    f"embedded MCore checkpoint component is missing: {path}"
                )
        shards = list(iteration_dir.glob("*.distcp"))
        if not shards or any(path.stat().st_size == 0 for path in shards):
            raise ValueError(
                f"embedded MCore checkpoint shards are incomplete: {iteration_dir}"
            )
        if (
            weights_path / "latest_checkpointed_iteration.txt"
        ).read_text().strip() != "0":
            raise ValueError("MCore checkpoint iteration marker must be 0")

    info = json.loads(info_path.read_text())
    expected_info = {
        "current_step": expected_step,
        "total_steps": expected_step,
        "consumed_samples": 8 * expected_step,
    }
    for name, expected in expected_info.items():
        actual = info.get(name)
        if actual != expected:
            raise ValueError(f"training_info.{name} must be {expected}; got {actual!r}")
    horizon_match = HORIZON_PATTERN.search(config_path.read_text())
    if horizon_match is None:
        raise ValueError(f"checkpoint has no grpo.max_num_steps: {config_path}")
    actual_horizon = int(horizon_match.group(1))
    if actual_horizon != expected_horizon_steps:
        raise ValueError(
            "checkpoint grpo.max_num_steps must remain "
            f"{expected_horizon_steps}; got {actual_horizon}"
        )
    return step_dir


def extract_wandb_run_id(train_log: Path) -> str:
    """Recover the successful gate's W&B identity for resume=must."""
    matches = WANDB_RUN_PATTERN.findall(train_log.read_text(errors="replace"))
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise ValueError(f"expected one W&B run id in {train_log}; found {unique}")
    return unique[0]


def write_gate_manifest(
    manifest_path: Path,
    *,
    dflash_k: int,
    git_sha: str,
    checkpoint_root: Path,
    wandb_run_id: str,
    target_revision: str,
    drafter_revision: str,
    container_sha256: str,
    training_horizon_steps: int = TRAINING_HORIZON_STEPS,
) -> None:
    """Persist the immutable identity needed by every resume segment."""
    payload = {
        "schema_version": 2,
        "dflash_k": dflash_k,
        "git_sha": git_sha,
        "checkpoint_root": str(checkpoint_root.resolve()),
        "wandb_run_id": wandb_run_id,
        "target_revision": target_revision,
        "drafter_revision": drafter_revision,
        "container_sha256": container_sha256,
        "training_horizon_steps": training_horizon_steps,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(manifest_path)


def validate_gate_manifest(
    manifest_path: Path,
    *,
    dflash_k: int,
    git_sha: str,
    checkpoint_root: Path,
    target_revision: str,
    drafter_revision: str,
    container_sha256: str,
    training_horizon_steps: int = TRAINING_HORIZON_STEPS,
) -> dict[str, object]:
    """Reject cross-arm or cross-provenance checkpoint reuse."""
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "schema_version": 2,
        "dflash_k": dflash_k,
        "git_sha": git_sha,
        "checkpoint_root": str(checkpoint_root.resolve()),
        "target_revision": target_revision,
        "drafter_revision": drafter_revision,
        "container_sha256": container_sha256,
        "training_horizon_steps": training_horizon_steps,
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValueError(
                f"gate manifest {name} must be {value!r}; got {manifest.get(name)!r}"
            )
    if (
        not isinstance(manifest.get("wandb_run_id"), str)
        or not manifest["wandb_run_id"]
    ):
        raise ValueError("gate manifest wandb_run_id must be a non-empty string")
    return manifest


def verify_nemo_resume_paths(step_dir: Path) -> None:
    """Use NeMo-RL's loader to prove optimizer state is resumable."""
    from nemo_rl.utils.checkpoint import CheckpointManager

    weights_path, optimizer_path = CheckpointManager.get_resume_paths(step_dir)
    if weights_path is None or optimizer_path is None:
        raise ValueError(f"NeMo-RL found no resumable optimizer state in {step_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--previous-step", type=int)
    parser.add_argument("--current-step", type=int)
    parser.add_argument("--training-horizon-steps", type=int, default=1000)
    parser.add_argument("--print-latest-step", action="store_true")
    parser.add_argument("--wandb-log", type=Path)
    parser.add_argument("--print-wandb-run-id", action="store_true")
    parser.add_argument("--gate-manifest", type=Path)
    parser.add_argument("--dflash-k", type=int)
    parser.add_argument("--git-sha")
    parser.add_argument("--target-revision")
    parser.add_argument("--drafter-revision")
    parser.add_argument("--container-sha256")
    parser.add_argument("--create-gate-manifest", action="store_true")
    parser.add_argument("--validate-gate-manifest", action="store_true")
    parser.add_argument("--print-manifest-wandb-run-id", action="store_true")
    parser.add_argument("--verify-nemo-resume-paths", action="store_true")
    args = parser.parse_args()

    if args.previous_step is not None or args.current_step is not None:
        if args.previous_step is None or args.current_step is None:
            raise ValueError(
                "--previous-step and --current-step must be provided together"
            )
        validate_progress(
            args.previous_step,
            args.current_step,
            horizon_steps=args.training_horizon_steps,
        )
    if args.print_latest_step:
        if args.checkpoint_dir is None:
            raise ValueError("--checkpoint-dir is required with --print-latest-step")
        steps = _checkpoint_steps(args.checkpoint_dir)
        if not steps:
            raise ValueError(f"no checkpoint exists under {args.checkpoint_dir}")
        print(steps[-1])
    if args.expected_step is not None:
        if args.checkpoint_dir is None:
            raise ValueError("--checkpoint-dir is required with --expected-step")
        step_dir = validate_checkpoint(
            args.checkpoint_dir,
            expected_step=args.expected_step,
            expected_horizon_steps=args.training_horizon_steps,
        )
        if args.verify_nemo_resume_paths:
            verify_nemo_resume_paths(step_dir)
    elif args.checkpoint_dir is not None and not args.print_latest_step:
        raise ValueError("--expected-step is required with --checkpoint-dir")
    if args.print_wandb_run_id:
        if args.wandb_log is None:
            raise ValueError("--wandb-log is required with --print-wandb-run-id")
        print(extract_wandb_run_id(args.wandb_log))
    manifest_args = (
        args.gate_manifest,
        args.dflash_k,
        args.git_sha,
        args.checkpoint_dir,
        args.target_revision,
        args.drafter_revision,
        args.container_sha256,
    )
    if (
        args.create_gate_manifest
        or args.validate_gate_manifest
        or args.print_manifest_wandb_run_id
    ):
        if any(value is None for value in manifest_args):
            raise ValueError("all gate manifest identity arguments are required")
        assert args.gate_manifest is not None
        assert args.dflash_k is not None
        assert args.git_sha is not None
        assert args.checkpoint_dir is not None
        assert args.target_revision is not None
        assert args.drafter_revision is not None
        assert args.container_sha256 is not None
        if args.create_gate_manifest:
            if args.wandb_log is None:
                raise ValueError("--wandb-log is required to create a gate manifest")
            write_gate_manifest(
                args.gate_manifest,
                dflash_k=args.dflash_k,
                git_sha=args.git_sha,
                checkpoint_root=args.checkpoint_dir,
                wandb_run_id=extract_wandb_run_id(args.wandb_log),
                target_revision=args.target_revision,
                drafter_revision=args.drafter_revision,
                container_sha256=args.container_sha256,
                training_horizon_steps=args.training_horizon_steps,
            )
        manifest = validate_gate_manifest(
            args.gate_manifest,
            dflash_k=args.dflash_k,
            git_sha=args.git_sha,
            checkpoint_root=args.checkpoint_dir,
            target_revision=args.target_revision,
            drafter_revision=args.drafter_revision,
            container_sha256=args.container_sha256,
            training_horizon_steps=args.training_horizon_steps,
        )
        if args.print_manifest_wandb_run_id:
            print(manifest["wandb_run_id"])


if __name__ == "__main__":
    main()
