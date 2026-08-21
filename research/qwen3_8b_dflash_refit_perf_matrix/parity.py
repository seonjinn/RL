#!/usr/bin/env python3

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, cast


_ALLOWED_DIFFERENCE_ROOTS = frozenset(
    {
        "logger.wandb.config.ab_arm",
        "logger.wandb.config.draft_refit_enabled",
        "logger.wandb.config.draft_training_enabled",
        "logger.wandb.config.fixed_public_drafter",
        "logger.wandb.config.matrix_cell",
        "logger.wandb.name",
        "logger.wandb.tags",
        "policy.draft.enabled",
        "policy.draft.optimizer",
    }
)


def _runtime_contract() -> ModuleType:
    path = Path(__file__).with_name("runtime_contract.py")
    spec = importlib.util.spec_from_file_location("matrix_runtime_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load runtime contract: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolved_config(
    config_path: Path,
    *,
    cell_name: str,
    target_snapshot: str,
    drafter_snapshot: str,
    expected_head: str,
) -> dict[str, Any]:
    from omegaconf import OmegaConf

    from nemo_rl.utils.config import (
        load_config,
        parse_hydra_overrides,
        register_omegaconf_resolvers,
    )

    contract = _runtime_contract()
    cell = contract.resolve_cell(cell_name)
    register_omegaconf_resolvers()
    config = parse_hydra_overrides(
        load_config(config_path),
        list(
            contract.runtime_overrides(
                cell,
                target_snapshot=target_snapshot,
                drafter_snapshot=drafter_snapshot,
                scratch_root="/raid/scratch/dflash-refit-matrix/PARITY",
                wandb_run_id="PARITY",
                expected_head=expected_head,
            )
        ),
    )
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError(f"Resolved config must be a mapping: {config_path}")
    return cast(dict[str, Any], resolved)


def _difference_paths(left: Any, right: Any, path: str = "") -> set[str]:
    if path in _ALLOWED_DIFFERENCE_ROOTS:
        return {path} if left != right else set()
    if isinstance(left, dict) and isinstance(right, dict):
        differences: set[str] = set()
        for key in sorted(left.keys() | right.keys()):
            child = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                differences.add(child)
            else:
                differences.update(_difference_paths(left[key], right[key], child))
        return differences
    return {path} if left != right else set()


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
    shape: str,
    online_config: Path,
    fixed_config: Path,
    target_snapshot: str,
    drafter_snapshot: str,
    expected_head: str,
) -> dict[str, Any]:
    online = _resolved_config(
        online_config,
        cell_name=f"{shape}_online",
        target_snapshot=target_snapshot,
        drafter_snapshot=drafter_snapshot,
        expected_head=expected_head,
    )
    fixed = _resolved_config(
        fixed_config,
        cell_name=f"{shape}_fixed",
        target_snapshot=target_snapshot,
        drafter_snapshot=drafter_snapshot,
        expected_head=expected_head,
    )
    differences = _difference_paths(online, fixed)
    unexpected = sorted(differences - _ALLOWED_DIFFERENCE_ROOTS)
    online_common = _common_projection(online)
    fixed_common = _common_projection(fixed)
    if online_common != fixed_common:
        unexpected = sorted(set(unexpected) | {"common_projection"})
    fingerprint = hashlib.sha256(
        json.dumps(online_common, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    allowed = sorted(differences & _ALLOWED_DIFFERENCE_ROOTS)
    return {
        "status": (
            "passed"
            if not unexpected and set(allowed) == _ALLOWED_DIFFERENCE_ROOTS
            else "failed"
        ),
        "shape": shape,
        "allowed_differences": allowed,
        "unexpected_differences": unexpected,
        "common_fingerprint": fingerprint,
    }


def validate_proof(
    proof: Path,
    *,
    shape: str,
    expected_head: str,
    container_sha256: str,
) -> dict[str, Any]:
    payload = json.loads(proof.read_text())
    required = {
        "status": "passed",
        "shape": shape,
        "expected_head": expected_head,
        "container_sha256": container_sha256,
        "allowed_differences": sorted(_ALLOWED_DIFFERENCE_ROOTS),
        "unexpected_differences": [],
    }
    mismatches = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    fingerprint = payload.get("common_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        mismatches["common_fingerprint"] = {
            "expected": "64-character SHA256",
            "actual": fingerprint,
        }
    if mismatches:
        raise ValueError(f"Invalid parity proof: {mismatches}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check")
    check.add_argument("--shape", required=True)
    check.add_argument("--online-config", type=Path, required=True)
    check.add_argument("--fixed-config", type=Path, required=True)
    check.add_argument("--target-snapshot", required=True)
    check.add_argument("--drafter-snapshot", required=True)
    check.add_argument("--expected-head", required=True)
    check.add_argument("--container-sha256", required=True)
    check.add_argument("--proof", type=Path, required=True)
    validate = commands.add_parser("validate-proof")
    validate.add_argument("--proof", type=Path, required=True)
    validate.add_argument("--shape", required=True)
    validate.add_argument("--expected-head", required=True)
    validate.add_argument("--container-sha256", required=True)
    args = parser.parse_args()
    if args.command == "validate-proof":
        payload = validate_proof(
            args.proof,
            shape=args.shape,
            expected_head=args.expected_head,
            container_sha256=args.container_sha256,
        )
        print(f"parity_proof=valid fingerprint={payload['common_fingerprint']}")
        return
    payload = resolve_pair(
        shape=args.shape,
        online_config=args.online_config,
        fixed_config=args.fixed_config,
        target_snapshot=args.target_snapshot,
        drafter_snapshot=args.drafter_snapshot,
        expected_head=args.expected_head,
    )
    payload.update(
        {
            "expected_head": args.expected_head,
            "container_sha256": args.container_sha256,
            "online_config_sha256": hashlib.sha256(
                args.online_config.read_bytes()
            ).hexdigest(),
            "fixed_config_sha256": hashlib.sha256(
                args.fixed_config.read_bytes()
            ).hexdigest(),
        }
    )
    args.proof.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if payload["status"] != "passed":
        raise RuntimeError(f"Resolved parity failed: {payload}")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
