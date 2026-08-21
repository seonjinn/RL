#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import re
import subprocess
from typing import Any


EXPERIMENT_PATH = "research/qwen3_8b_dflash_refit_perf_matrix"
_DCO = re.compile(r"(?m)^Signed-off-by: .+ <[^>]+>$")


def _git(checkout: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", checkout, *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_checkout(
    checkout: Path,
    *,
    product_head: str,
    harness_head: str,
    require_signed_dco: bool = True,
    require_home: bool = False,
) -> dict[str, Any]:
    checkout = checkout.resolve(strict=True)
    if require_home and not str(checkout).startswith("/home/"):
        raise ValueError(f"source checkout must be under /home: {checkout}")
    actual_head = _git(checkout, "rev-parse", "HEAD")
    if actual_head != harness_head:
        raise ValueError(f"harness head mismatch: {actual_head} != {harness_head}")
    if _git(checkout, "merge-base", product_head, harness_head) != product_head:
        raise ValueError("product head is not the harness ancestor")
    delta = [
        path
        for path in _git(
            checkout, "diff", "--name-only", product_head, harness_head, "--"
        ).splitlines()
        if path
    ]
    if not delta or any(not path.startswith(f"{EXPERIMENT_PATH}/") for path in delta):
        raise ValueError(f"non-experiment harness delta: {delta}")
    status = _git(checkout, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise ValueError(f"dirty source checkout: {status}")
    _git(
        checkout,
        "submodule",
        "foreach",
        "--quiet",
        "--recursive",
        'git diff-index --quiet --ignore-submodules=all HEAD -- && test -z "$(git ls-files --others --exclude-standard)"',
    )
    commits = [
        commit
        for commit in _git(
            checkout, "rev-list", "--reverse", f"{product_head}..{harness_head}"
        ).splitlines()
        if commit
    ]
    if require_signed_dco:
        for commit in commits:
            _git(checkout, "verify-commit", commit)
            body = _git(checkout, "show", "-s", "--format=%B", commit)
            if _DCO.search(body) is None:
                raise ValueError(f"commit lacks DCO sign-off: {commit}")
    return {
        "status": "passed",
        "checkout": str(checkout),
        "product_head": product_head,
        "harness_head": harness_head,
        "harness_tree": _git(checkout, "rev-parse", f"HEAD:{EXPERIMENT_PATH}"),
        "harness_delta": delta,
        "harness_commits": commits,
        "signed_dco_required": require_signed_dco,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--product-head", required=True)
    parser.add_argument("--harness-head", required=True)
    parser.add_argument("--proof", type=Path, required=True)
    args = parser.parse_args()
    proof = validate_checkout(
        args.checkout,
        product_head=args.product_head,
        harness_head=args.harness_head,
        require_signed_dco=True,
        require_home=True,
    )
    args.proof.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    print(json.dumps(proof, sort_keys=True))


if __name__ == "__main__":
    main()
