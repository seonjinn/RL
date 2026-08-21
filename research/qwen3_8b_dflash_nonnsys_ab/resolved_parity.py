#!/usr/bin/env python3

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, cast, Literal

Arm = Literal["fixed", "online"]

_ALLOWED_DIFFERENCE_ROOTS = frozenset(
    {
        "logger.wandb.config.ab_arm",
        "logger.wandb.config.draft_refit_enabled",
        "logger.wandb.config.draft_training_enabled",
        "logger.wandb.config.fixed_public_drafter",
        "policy.draft.enabled",
        "policy.draft.optimizer",
    }
)


@dataclass(frozen=True)
class RuntimeInputs:
    arm: Arm
    target_snapshot: str
    drafter_snapshot: str
    scratch_root: str
    wandb_run_id: str
    wandb_project: str
    expected_head: str


@dataclass(frozen=True)
class ParityReport:
    allowed_differences: tuple[str, ...]
    unexpected_differences: tuple[str, ...]
    fixed_update_probe_enabled: bool
    online_update_probe_enabled: bool
    common_fingerprint: str


def runtime_overrides(inputs: RuntimeInputs) -> tuple[str, ...]:
    draft_training_enabled = inputs.arm == "online"
    draft_training = str(draft_training_enabled).lower()
    return (
        "grpo.max_num_steps=50",
        "grpo.val_period=1000000",
        "grpo.val_at_start=false",
        "grpo.val_at_end=false",
        "checkpointing.enabled=false",
        f"checkpointing.checkpoint_dir={inputs.scratch_root}/checkpoints",
        f"policy.model_name={inputs.target_snapshot}",
        f"policy.tokenizer.name={inputs.target_snapshot}",
        f"policy.draft.model_name={inputs.drafter_snapshot}",
        "policy.draft.update_probe_enabled=false",
        (
            "policy.generation.vllm_kwargs.speculative_config.model="
            f"{inputs.drafter_snapshot}"
        ),
        f"logger.log_dir={inputs.scratch_root}/logs",
        "logger.wandb_enabled=true",
        "logger.tensorboard_enabled=false",
        "++logger.wandb.entity=nvidia",
        f"++logger.wandb.project={inputs.wandb_project}",
        "logger.wandb.group=qwen3-8b-dflash-k7-nonnsys-ab",
        "logger.wandb.name=qwen3-8b-dflash-k7-nonnsys-ab-50step",
        "logger.wandb.tags=[dflash,qwen3-8b,k7,cudagraph,nonnsys,ab]",
        f"++logger.wandb.id={inputs.wandb_run_id}",
        "++logger.wandb.resume=never",
        f"++logger.wandb.config.ab_arm={inputs.arm}",
        f"logger.wandb.config.draft_training_enabled={draft_training}",
        "++logger.wandb.config.optimized_source_sha="
        "79e80af96a13522e6049658663a8c40ab21e8314",
        f"++logger.wandb.config.harness_sha={inputs.expected_head}",
        "++logger.wandb.config.performance_window=steps_5_through_49",
    )


def _resolved_config(config_path: Path, inputs: RuntimeInputs) -> dict[str, Any]:
    from omegaconf import OmegaConf

    from nemo_rl.utils.config import (
        load_config,
        parse_hydra_overrides,
        register_omegaconf_resolvers,
    )

    register_omegaconf_resolvers()
    config = parse_hydra_overrides(
        load_config(config_path), list(runtime_overrides(inputs))
    )
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError(f"Resolved config must be a mapping: {config_path}")
    return cast(dict[str, Any], resolved)


def _difference_paths(
    online: Any,
    fixed: Any,
    path: str = "",
) -> set[str]:
    if path in _ALLOWED_DIFFERENCE_ROOTS:
        return {path} if online != fixed else set()
    if isinstance(online, dict) and isinstance(fixed, dict):
        differences: set[str] = set()
        for key in sorted(online.keys() | fixed.keys()):
            child_path = f"{path}.{key}" if path else str(key)
            if key not in online or key not in fixed:
                differences.add(child_path)
                continue
            differences.update(_difference_paths(online[key], fixed[key], child_path))
        return differences
    return {path} if online != fixed else set()


def _common_projection(value: dict[str, Any]) -> dict[str, Any]:
    projected = deepcopy(value)
    for path in _ALLOWED_DIFFERENCE_ROOTS:
        keys = path.split(".")
        parent: Any = projected
        for key in keys[:-1]:
            if not isinstance(parent, dict) or key not in parent:
                break
            parent = parent[key]
        else:
            if isinstance(parent, dict):
                parent.pop(keys[-1], None)
    return projected


def resolve_pair(
    *,
    online_config: Path,
    fixed_config: Path,
    online_inputs: RuntimeInputs,
    fixed_inputs: RuntimeInputs,
) -> ParityReport:
    if online_inputs.arm != "online" or fixed_inputs.arm != "fixed":
        raise ValueError("Parity inputs must be ordered online then fixed")
    online = _resolved_config(online_config, online_inputs)
    fixed = _resolved_config(fixed_config, fixed_inputs)
    differences = _difference_paths(online, fixed)
    allowed = tuple(sorted(differences & _ALLOWED_DIFFERENCE_ROOTS))
    unexpected = tuple(sorted(differences - _ALLOWED_DIFFERENCE_ROOTS))
    online_probe = online["policy"]["draft"]["update_probe_enabled"]
    fixed_probe = fixed["policy"]["draft"]["update_probe_enabled"]
    online_common = _common_projection(online)
    fixed_common = _common_projection(fixed)
    if online_common != fixed_common:
        unexpected = tuple(sorted(set(unexpected) | {"common_projection"}))
    common_payload = json.dumps(
        online_common,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    fingerprint = hashlib.sha256(common_payload).hexdigest()
    return ParityReport(
        allowed_differences=allowed,
        unexpected_differences=unexpected,
        fixed_update_probe_enabled=bool(fixed_probe),
        online_update_probe_enabled=bool(online_probe),
        common_fingerprint=fingerprint,
    )


def _inputs_from_args(args: argparse.Namespace, arm: Arm) -> RuntimeInputs:
    return RuntimeInputs(
        arm=arm,
        target_snapshot=args.target_snapshot,
        drafter_snapshot=args.drafter_snapshot,
        scratch_root=args.scratch_root,
        wandb_run_id=args.wandb_run_id,
        wandb_project=args.wandb_project,
        expected_head=args.expected_head,
    )


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target-snapshot", required=True)
    parser.add_argument("--drafter-snapshot", required=True)
    parser.add_argument("--scratch-root", required=True)
    parser.add_argument("--wandb-run-id", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--expected-head", required=True)


def validate_proof(
    *,
    proof: Path,
    expected_head: str,
    target_snapshot: str,
    drafter_snapshot: str,
    container_sha256: str,
    wandb_project: str,
    parity_job_id: str | None = None,
    online_config: Path | None = None,
    fixed_config: Path | None = None,
) -> dict[str, Any]:
    payload = json.loads(proof.read_text())
    required_values = {
        "status": "passed",
        "expected_head": expected_head,
        "target_snapshot": target_snapshot,
        "drafter_snapshot": drafter_snapshot,
        "container_sha256": container_sha256,
        "wandb_project": wandb_project,
        "allowed_differences": sorted(_ALLOWED_DIFFERENCE_ROOTS),
        "unexpected_differences": [],
        "fixed_update_probe_enabled": False,
        "online_update_probe_enabled": False,
    }
    mismatches: dict[str, dict[str, Any]] = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required_values.items()
        if payload.get(key) != expected
    }
    if parity_job_id is not None:
        required_artifacts = {
            "parity_job_id": parity_job_id,
            "online_config": str(online_config.resolve(strict=True))
            if online_config is not None
            else None,
            "fixed_config": str(fixed_config.resolve(strict=True))
            if fixed_config is not None
            else None,
            "online_config_sha256": hashlib.sha256(
                online_config.read_bytes()
            ).hexdigest()
            if online_config is not None
            else None,
            "fixed_config_sha256": hashlib.sha256(fixed_config.read_bytes()).hexdigest()
            if fixed_config is not None
            else None,
        }
        mismatches.update(
            {
                key: {"expected": expected, "actual": payload.get(key)}
                for key, expected in required_artifacts.items()
                if payload.get(key) != expected
            }
        )
    fingerprint = payload.get("common_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        mismatches["common_fingerprint"] = {
            "expected": "64-character SHA256",
            "actual": fingerprint,
        }
    if mismatches:
        raise ValueError(f"Invalid resolved parity proof: {mismatches}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    emit = subparsers.add_parser("emit-overrides")
    emit.add_argument("--arm", choices=("fixed", "online"), required=True)
    _add_runtime_arguments(emit)
    check = subparsers.add_parser("check")
    check.add_argument("--online-config", type=Path, required=True)
    check.add_argument("--fixed-config", type=Path, required=True)
    check.add_argument("--proof", type=Path, required=True)
    check.add_argument("--container-sha256", required=True)
    check.add_argument("--parity-job-id", required=True)
    _add_runtime_arguments(check)
    validate = subparsers.add_parser("validate-proof")
    validate.add_argument("--proof", type=Path, required=True)
    validate.add_argument("--expected-head", required=True)
    validate.add_argument("--target-snapshot", required=True)
    validate.add_argument("--drafter-snapshot", required=True)
    validate.add_argument("--container-sha256", required=True)
    validate.add_argument("--wandb-project", required=True)
    validate.add_argument("--parity-job-id", required=True)
    validate.add_argument("--online-config", type=Path, required=True)
    validate.add_argument("--fixed-config", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "emit-overrides":
        inputs = _inputs_from_args(args, args.arm)
        print(*runtime_overrides(inputs), sep="\n")
        return

    if args.command == "validate-proof":
        payload = validate_proof(
            proof=args.proof,
            expected_head=args.expected_head,
            target_snapshot=args.target_snapshot,
            drafter_snapshot=args.drafter_snapshot,
            container_sha256=args.container_sha256,
            wandb_project=args.wandb_project,
            parity_job_id=args.parity_job_id,
            online_config=args.online_config,
            fixed_config=args.fixed_config,
        )
        print(
            f"resolved_parity_proof=valid fingerprint={payload['common_fingerprint']}"
        )
        return

    online_inputs = _inputs_from_args(args, "online")
    fixed_inputs = _inputs_from_args(args, "fixed")
    report = resolve_pair(
        online_config=args.online_config,
        fixed_config=args.fixed_config,
        online_inputs=online_inputs,
        fixed_inputs=fixed_inputs,
    )
    payload = asdict(report)
    payload.update(
        {
            "expected_head": args.expected_head,
            "target_snapshot": args.target_snapshot,
            "drafter_snapshot": args.drafter_snapshot,
            "container_sha256": args.container_sha256,
            "wandb_project": args.wandb_project,
            "parity_job_id": args.parity_job_id,
            "online_config": str(args.online_config.resolve(strict=True)),
            "fixed_config": str(args.fixed_config.resolve(strict=True)),
            "online_config_sha256": hashlib.sha256(
                args.online_config.read_bytes()
            ).hexdigest(),
            "fixed_config_sha256": hashlib.sha256(
                args.fixed_config.read_bytes()
            ).hexdigest(),
        }
    )
    payload["status"] = (
        "passed"
        if not report.unexpected_differences
        and not report.fixed_update_probe_enabled
        and not report.online_update_probe_enabled
        else "failed"
    )
    args.proof.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if payload["status"] != "passed":
        raise RuntimeError(f"Resolved parity failed: {payload}")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
