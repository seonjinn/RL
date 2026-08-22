#!/usr/bin/env python3

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, cast


_ALLOWED = frozenset(
    {
        "logger.wandb.config.ab_arm",
        "logger.wandb.config.draft_refit_enabled",
        "logger.wandb.config.draft_training_enabled",
        "logger.wandb.name",
        "logger.wandb.tags",
        "policy.draft.enabled",
    }
)


def allowed_difference_paths() -> set[str]:
    return set(_ALLOWED)


def _contract() -> ModuleType:
    path = Path(__file__).with_name("runtime_contract.py")
    spec = importlib.util.spec_from_file_location("pack_cp2_runtime_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load runtime contract: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _difference_paths(left: Any, right: Any, path: str = "") -> set[str]:
    if path in _ALLOWED:
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
    for path in _ALLOWED:
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


def _resolved(
    config: Path,
    *,
    arm: str,
    target_snapshot: str,
    drafter_snapshot: str,
    expected_head: str,
) -> dict[str, Any]:
    from omegaconf import OmegaConf  # pyrefly: ignore [missing-import]

    from nemo_rl.utils.config import (
        load_config,
        parse_hydra_overrides,
        register_omegaconf_resolvers,
    )

    register_omegaconf_resolvers()
    contract = _contract()
    overrides = contract.runtime_overrides(
        contract.resolve_arm(arm),
        target_snapshot=target_snapshot,
        drafter_snapshot=drafter_snapshot,
        scratch_root="/raid/scratch/parity",
        wandb_run_id="parity",
        expected_head=expected_head,
    )
    resolved = OmegaConf.to_container(
        parse_hydra_overrides(load_config(config), list(overrides)), resolve=True
    )
    if not isinstance(resolved, dict):
        raise TypeError("resolved config must be a mapping")
    result = cast(dict[str, Any], resolved)
    policy = result["policy"]
    grpo = result["grpo"]
    data = result["data"]
    assert grpo["seed"] == 42
    assert grpo["max_num_steps"] == 30
    assert data["shuffle"] is True
    assert data["train"]["dataset_name"] == "DAPOMath17K"
    assert policy["sequence_packing"]["enabled"] is True
    assert policy["megatron_cfg"]["tensor_model_parallel_size"] == 2
    assert policy["megatron_cfg"]["context_parallel_size"] == 2
    assert policy["megatron_cfg"]["sequence_parallel"] is True
    assert policy["make_sequence_length_divisible_by"] == 16
    assert policy["draft"]["gamma"] == 5
    assert (
        policy["generation"]["vllm_kwargs"]["speculative_config"]
        ["num_speculative_tokens"]
        == 5
    )
    return result


def resolve_pair(
    config: Path,
    *,
    target_snapshot: str,
    drafter_snapshot: str,
    expected_head: str,
) -> dict[str, Any]:
    fixed = _resolved(
        config,
        arm="fixed",
        target_snapshot=target_snapshot,
        drafter_snapshot=drafter_snapshot,
        expected_head=expected_head,
    )
    online = _resolved(
        config,
        arm="online",
        target_snapshot=target_snapshot,
        drafter_snapshot=drafter_snapshot,
        expected_head=expected_head,
    )
    differences = _difference_paths(fixed, online)
    unexpected = sorted(differences - _ALLOWED)
    allowed = sorted(differences & _ALLOWED)
    common = _common_projection(fixed)
    status = "passed" if not unexpected and set(allowed) == _ALLOWED else "failed"
    return {
        "status": status,
        "allowed_differences": allowed,
        "unexpected_differences": unexpected,
        "common_fingerprint": hashlib.sha256(
            json.dumps(common, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target-snapshot", required=True)
    parser.add_argument("--drafter-snapshot", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--proof", type=Path, required=True)
    args = parser.parse_args()
    payload = resolve_pair(
        args.config,
        target_snapshot=args.target_snapshot,
        drafter_snapshot=args.drafter_snapshot,
        expected_head=args.expected_head,
    )
    payload["expected_head"] = args.expected_head
    args.proof.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if payload["status"] != "passed":
        raise RuntimeError(f"resolved config parity failed: {payload}")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
