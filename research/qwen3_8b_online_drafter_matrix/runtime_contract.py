#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any


HORIZON_STEPS = 1000


@dataclass(frozen=True)
class ArmContract:
    config_name: str
    draft_enabled: bool
    method: str | None
    k: int | None


_ARMS = {
    "baseline": ArmContract("baseline.yaml", False, None, None),
    "dflash-fixed-k5": ArmContract("dflash-fixed-k5.yaml", False, "dflash", 5),
    "dflash-fixed-k7": ArmContract("dflash-fixed-k7.yaml", False, "dflash", 7),
    "dflash-k5": ArmContract("dflash-k5.yaml", True, "dflash", 5),
    "dflash-k7": ArmContract("dflash-k7.yaml", True, "dflash", 7),
}


def arm_contract(arm: str) -> ArmContract:
    try:
        return _ARMS[arm]
    except KeyError as error:
        raise ValueError(f"unsupported matrix arm: {arm}") from error


def validate_arm_config(
    arm: str,
    config: dict[str, Any],
    *,
    expected_draft_enabled: bool,
    expected_method: str | None,
    expected_k: int | None,
) -> None:
    contract = arm_contract(arm)
    if (
        contract.draft_enabled != expected_draft_enabled
        or contract.method != expected_method
        or contract.k != expected_k
    ):
        raise ValueError("test expectation disagrees with arm contract")
    policy = config["policy"]
    grpo = config["grpo"]
    assert policy["sequence_packing"]["enabled"] is False
    assert policy["megatron_cfg"]["sequence_parallel"] is False
    assert policy["megatron_cfg"]["context_parallel_size"] == 1
    assert policy["megatron_cfg"]["tensor_model_parallel_size"] == 2
    assert policy["train_global_batch_size"] == 32
    assert grpo["num_prompts_per_step"] == 8
    assert grpo["num_generations_per_prompt"] == 4
    assert grpo["seed"] == 42
    assert policy["draft"]["enabled"] is expected_draft_enabled
    speculative = policy["generation"]["vllm_kwargs"]["speculative_config"]
    if expected_method is None:
        assert speculative is None
    else:
        assert speculative["method"] == expected_method
        assert speculative["num_speculative_tokens"] == expected_k
        assert policy["draft"]["gamma"] == expected_k


def checkpoint_steps(root: Path) -> list[int]:
    return sorted(
        int(match.group(1))
        for path in root.glob("step_*")
        if path.is_dir() and (match := re.fullmatch(r"step_(\d+)", path.name))
    )


def latest_step(root: Path) -> int:
    steps = checkpoint_steps(root)
    if not steps:
        raise ValueError(f"no checkpoint exists under {root}")
    return steps[-1]


def validate_progress(previous: int, current: int, minimum: int) -> None:
    if not previous < current <= HORIZON_STEPS or current < minimum:
        raise ValueError(
            f"invalid matrix progress {previous}->{current}, need {minimum}"
        )


def write_manifest(
    path: Path,
    *,
    arm: str,
    git_sha: str,
    checkpoint_root: Path,
    wandb_run_id: str,
    target_revision: str,
    drafter_revision: str,
    container_sha256: str,
) -> None:
    contract = arm_contract(arm)
    payload = {
        "schema_version": 1,
        "arm": arm,
        "git_sha": git_sha,
        "checkpoint_root": str(checkpoint_root.resolve()),
        "wandb_run_id": wandb_run_id,
        "target_revision": target_revision,
        "drafter_revision": drafter_revision,
        "container_sha256": container_sha256,
        "training_horizon_steps": HORIZON_STEPS,
        "draft_training_enabled": contract.draft_enabled,
        "speculator_type": contract.method,
        "k": contract.k,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_manifest(path: Path, **expected: object) -> dict[str, object]:
    payload = json.loads(path.read_text())
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"manifest {key} mismatch")
    if not payload.get("wandb_run_id"):
        raise ValueError("manifest requires W&B run ID")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(_ARMS))
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--print-latest-step", action="store_true")
    parser.add_argument("--previous-step", type=int)
    parser.add_argument("--current-step", type=int)
    parser.add_argument("--required-min-step", type=int)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--git-sha")
    parser.add_argument("--wandb-run-id")
    parser.add_argument("--target-revision")
    parser.add_argument("--drafter-revision")
    parser.add_argument("--container-sha256")
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--validate-manifest", action="store_true")
    parser.add_argument("--print-wandb-id", action="store_true")
    args = parser.parse_args()
    if args.arm is not None and args.print_config:
        print(arm_contract(args.arm).config_name)
        return
    if args.print_latest_step:
        if args.checkpoint_dir is None:
            parser.error("--checkpoint-dir is required")
        print(latest_step(args.checkpoint_dir))
        return
    if args.previous_step is not None:
        if args.current_step is None or args.required_min_step is None:
            parser.error("progress validation requires current and minimum")
        validate_progress(args.previous_step, args.current_step, args.required_min_step)
    if args.write_manifest:
        if None in (
            args.arm,
            args.manifest,
            args.checkpoint_dir,
            args.wandb_run_id,
            args.git_sha,
            args.target_revision,
            args.drafter_revision,
            args.container_sha256,
        ):
            parser.error("manifest identity is incomplete")
        write_manifest(
            args.manifest,
            arm=args.arm,
            git_sha=args.git_sha,
            checkpoint_root=args.checkpoint_dir,
            wandb_run_id=args.wandb_run_id,
            target_revision=args.target_revision,
            drafter_revision=args.drafter_revision,
            container_sha256=args.container_sha256,
        )
    if args.validate_manifest or args.print_wandb_id:
        if None in (
            args.arm,
            args.manifest,
            args.checkpoint_dir,
            args.git_sha,
            args.target_revision,
            args.drafter_revision,
            args.container_sha256,
        ):
            parser.error("manifest identity is incomplete")
        contract = arm_contract(args.arm)
        payload = validate_manifest(
            args.manifest,
            schema_version=1,
            arm=args.arm,
            git_sha=args.git_sha,
            checkpoint_root=str(args.checkpoint_dir.resolve()),
            target_revision=args.target_revision,
            drafter_revision=args.drafter_revision,
            container_sha256=args.container_sha256,
            training_horizon_steps=HORIZON_STEPS,
            draft_training_enabled=contract.draft_enabled,
            speculator_type=contract.method,
            k=contract.k,
        )
        if args.print_wandb_id:
            print(payload["wandb_run_id"])


if __name__ == "__main__":
    main()
