#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import re
import subprocess
from typing import Any


PRODUCT_HEAD = "443e7243ae2a235b6dcd8f4918fea86e693630a9"
EXPERIMENT_PATH = "research/qwen3_8b_dflash_pack_cp2_baseline"
_DCO = re.compile(r"(?m)^Signed-off-by: .+ <[^>]+>$")


def _git(checkout: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", checkout, *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_raw(checkout: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", checkout, *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.rstrip("\n")


def validate_submodule_status(status: str) -> list[str]:
    states = {
        "-": "uninitialized",
        "+": "unexpected commit",
        "U": "merge conflict",
    }
    exact: list[str] = []
    for line in status.splitlines():
        state = line[:1]
        if state in states:
            path = line[42:].split(" ", 1)[0]
            raise ValueError(f"{states[state]} recursive submodule: {path}")
        if state != " ":
            raise ValueError(f"invalid recursive submodule status: {line}")
        exact.append(line[1:])
    if not exact:
        raise ValueError("recursive submodule status is empty")
    return exact


def validate_checkout(checkout: Path, harness_head: str) -> dict[str, Any]:
    root = checkout.resolve(strict=True)
    if not str(root).startswith("/home/"):
        raise ValueError(f"source checkout must be under /home: {root}")
    actual = _git(root, "rev-parse", "HEAD")
    if actual != harness_head:
        raise ValueError(f"harness head mismatch: {actual} != {harness_head}")
    if _git(root, "merge-base", PRODUCT_HEAD, harness_head) != PRODUCT_HEAD:
        raise ValueError("validated product head is not the harness ancestor")
    delta = [
        path
        for path in _git(root, "diff", "--name-only", PRODUCT_HEAD, harness_head).splitlines()
        if path
    ]
    if not delta or any(not path.startswith(f"{EXPERIMENT_PATH}/") for path in delta):
        raise ValueError(f"non-experiment harness delta: {delta}")
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("dirty source checkout")
    submodules = validate_submodule_status(
        _git_raw(root, "submodule", "status", "--recursive")
    )
    _git(
        root,
        "submodule",
        "foreach",
        "--quiet",
        "--recursive",
        'git diff-index --quiet --ignore-submodules=all HEAD -- && test -z "$(git ls-files --others --exclude-standard)"',
    )
    signers = root / EXPERIMENT_PATH / "allowed_signers"
    commits = _git(root, "rev-list", "--reverse", f"{PRODUCT_HEAD}..{harness_head}").splitlines()
    for commit in commits:
        _git(root, "-c", f"gpg.ssh.allowedSignersFile={signers}", "verify-commit", commit)
        if _DCO.search(_git(root, "show", "-s", "--format=%B", commit)) is None:
            raise ValueError(f"commit lacks DCO sign-off: {commit}")
    return {
        "status": "passed",
        "product_head": PRODUCT_HEAD,
        "harness_head": harness_head,
        "harness_delta": delta,
        "harness_commits": commits,
        "submodules": submodules,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--harness-head", required=True)
    parser.add_argument("--proof", type=Path, required=True)
    args = parser.parse_args()
    proof = validate_checkout(args.checkout, args.harness_head)
    args.proof.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    print(json.dumps(proof, sort_keys=True))


if __name__ == "__main__":
    main()
