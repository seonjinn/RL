#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


BASE_PRODUCT_SHA = "79e80af96a13522e6049658663a8c40ab21e8314"
OPTIMIZED_PRODUCT_SHA = "f909e3d124bb663db4099e88f6846e55b0500912"
EXPERIMENT_PATH = "research/qwen3_8b_dflash_nonnsys_ab"
ONLINE_CONFIG_PATH = "research/qwen3_8b_dflash_online_cp1/config.yaml"
EXPECTED_PRODUCT_DELTA = (
    "nemo_rl/algorithms/loss/wrapper.py",
    "nemo_rl/models/megatron/draft/step_state.py",
    "tests/unit/algorithms/test_draft_loss_wrapper.py",
    "tests/unit/models/megatron/test_draft_step_state.py",
)
SUBMODULE_CLEAN_COMMAND = (
    "git diff-index --quiet --ignore-submodules=all HEAD -- && "
    'test -z "$(git ls-files --others --exclude-standard)"'
)


def _git(checkout: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", checkout, *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_checkout(
    checkout: Path,
    *,
    product_head: str,
    harness_head: str,
) -> dict[str, Any]:
    checkout = checkout.resolve(strict=True)
    if not str(checkout).startswith("/home/"):
        raise ValueError(f"source checkout must be under /home: {checkout}")
    actual_head = _git(checkout, "rev-parse", "HEAD")
    if actual_head != harness_head:
        raise ValueError(f"harness head mismatch: {actual_head} != {harness_head}")
    merge_base = _git(checkout, "merge-base", product_head, harness_head)
    if merge_base != product_head:
        raise ValueError(f"product is not harness ancestor: {merge_base}")
    harness_delta = tuple(
        line
        for line in _git(
            checkout, "diff", "--name-only", product_head, harness_head, "--"
        ).splitlines()
        if line
    )
    if not harness_delta or any(
        not path.startswith(f"{EXPERIMENT_PATH}/") for path in harness_delta
    ):
        raise ValueError(f"non-experiment harness delta: {harness_delta}")
    status = _git(checkout, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise ValueError(f"dirty source checkout: {status}")
    _git(
        checkout,
        "submodule",
        "foreach",
        "--quiet",
        "--recursive",
        SUBMODULE_CLEAN_COMMAND,
    )
    return {
        "checkout": str(checkout),
        "product_head": product_head,
        "harness_head": harness_head,
        "harness_tree": _git(checkout, "rev-parse", f"HEAD:{EXPERIMENT_PATH}"),
        "online_config_sha256": _sha256(checkout / ONLINE_CONFIG_PATH),
        "harness_delta": list(harness_delta),
    }


def check_pair(
    *,
    base_checkout: Path,
    optimized_checkout: Path,
    base_harness_head: str,
    optimized_harness_head: str,
) -> dict[str, Any]:
    if base_checkout.resolve(strict=True) == optimized_checkout.resolve(strict=True):
        raise ValueError("base and optimized checkouts must be distinct")
    base = _validate_checkout(
        base_checkout,
        product_head=BASE_PRODUCT_SHA,
        harness_head=base_harness_head,
    )
    optimized = _validate_checkout(
        optimized_checkout,
        product_head=OPTIMIZED_PRODUCT_SHA,
        harness_head=optimized_harness_head,
    )
    product_delta = tuple(
        line
        for line in _git(
            base_checkout,
            "diff",
            "--name-only",
            BASE_PRODUCT_SHA,
            OPTIMIZED_PRODUCT_SHA,
            "--",
        ).splitlines()
        if line
    )
    if product_delta != EXPECTED_PRODUCT_DELTA:
        raise ValueError(f"unexpected product delta: {product_delta}")
    if base["harness_tree"] != optimized["harness_tree"]:
        raise ValueError("base and optimized harness trees differ")
    if base["online_config_sha256"] != optimized["online_config_sha256"]:
        raise ValueError("base and optimized online configs differ")
    return {
        "status": "passed",
        "base": base,
        "optimized": optimized,
        "product_delta": list(product_delta),
        "performance_window": "steps_5_through_49",
        "profiled": False,
    }


def validate_proof(
    proof: Path,
    *,
    source_arm: str,
    product_head: str,
    harness_head: str,
) -> None:
    payload = json.loads(proof.read_text())
    if payload.get("status") != "passed":
        raise ValueError("source parity proof did not pass")
    arm = payload.get(source_arm)
    if not isinstance(arm, dict):
        raise ValueError(f"source parity proof has no {source_arm} arm")
    expected = {"product_head": product_head, "harness_head": harness_head}
    actual = {key: arm.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"source parity identity mismatch: {actual} != {expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check")
    check.add_argument("--base-checkout", type=Path, required=True)
    check.add_argument("--optimized-checkout", type=Path, required=True)
    check.add_argument("--base-harness-head", required=True)
    check.add_argument("--optimized-harness-head", required=True)
    check.add_argument("--proof", type=Path, required=True)
    validate = commands.add_parser("validate-proof")
    validate.add_argument("--proof", type=Path, required=True)
    validate.add_argument("--source-arm", choices=("base", "optimized"), required=True)
    validate.add_argument("--product-head", required=True)
    validate.add_argument("--harness-head", required=True)
    args = parser.parse_args()
    if args.command == "validate-proof":
        validate_proof(
            args.proof,
            source_arm=args.source_arm,
            product_head=args.product_head,
            harness_head=args.harness_head,
        )
        print(f"source_parity_proof=valid arm={args.source_arm}")
        return
    payload = check_pair(
        base_checkout=args.base_checkout,
        optimized_checkout=args.optimized_checkout,
        base_harness_head=args.base_harness_head,
        optimized_harness_head=args.optimized_harness_head,
    )
    args.proof.parent.mkdir(parents=True, exist_ok=False)
    args.proof.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
