#!/usr/bin/env python3
"""Validate and apply the pinned vLLM 0.28 MRV2 DynamicMTP patch stack."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class PatchValidationError(RuntimeError):
    """Raised before an unverified patch stack can mutate a source tree."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PatchValidationError(f"cannot load patch manifest: {path}") from error
    _validate_manifest(manifest)
    return manifest


def _validate_manifest(manifest: object) -> None:
    if not isinstance(manifest, dict):
        raise PatchValidationError("patch manifest must be an object")
    if manifest.get("schema_version") != 1:
        raise PatchValidationError("unsupported patch manifest schema")
    base_commit = manifest.get("base_commit")
    if not isinstance(base_commit, str) or not _HEX40.fullmatch(base_commit):
        raise PatchValidationError("base_commit must be a full git SHA")
    patches = manifest.get("patches")
    if not isinstance(patches, list) or not patches:
        raise PatchValidationError("patch manifest must contain a nonempty series")
    for row in patches:
        if not isinstance(row, dict):
            raise PatchValidationError("patch rows must be objects")
        if not isinstance(row.get("file"), str) or Path(row["file"]).is_absolute():
            raise PatchValidationError("patch file must be relative")
        if ".." in Path(row["file"]).parts:
            raise PatchValidationError("patch file cannot escape the patch directory")
        if not isinstance(row.get("commit"), str) or not _HEX40.fullmatch(
            row["commit"]
        ):
            raise PatchValidationError("patch commit must be a full git SHA")
        if not isinstance(row.get("sha256"), str) or not _HEX64.fullmatch(
            row["sha256"]
        ):
            raise PatchValidationError("patch sha256 must be a lowercase digest")


def _git(source_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise PatchValidationError(f"git command failed: {' '.join(args)}") from error
    return completed.stdout.strip()


def _stable_patch_id(diff: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "patch-id", "--stable"],
            input=diff,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise PatchValidationError("unable to calculate stable patch ID") from error
    fields = completed.stdout.split()
    if not fields or not _HEX40.fullmatch(fields[0]):
        raise PatchValidationError("patch has no stable patch ID")
    return fields[0]


def _validate_patch_files(
    patch_dir: Path, manifest: dict[str, Any]
) -> tuple[list[Path], list[str]]:
    try:
        patch_root = patch_dir.resolve(strict=True)
    except OSError as error:
        raise PatchValidationError(f"patch directory does not exist: {patch_dir}") from error
    paths: list[Path] = []
    patch_ids: list[str] = []
    for row in manifest["patches"]:
        unresolved_path = patch_root / row["file"]
        try:
            path = unresolved_path.resolve(strict=True)
        except OSError as error:
            raise PatchValidationError(
                f"missing patch file: {unresolved_path}"
            ) from error
        if not path.is_relative_to(patch_root) or not path.is_file():
            raise PatchValidationError(f"missing patch file: {path}")
        if sha256_file(path) != row["sha256"]:
            raise PatchValidationError(f"patch digest mismatch: {path}")
        paths.append(path)
        patch_ids.append(_stable_patch_id(path.read_text(encoding="utf-8")))
    return paths, patch_ids


def apply_patch_stack(
    *, source_root: Path, patch_dir: Path, manifest: dict[str, Any]
) -> list[str]:
    """Apply a fully verified mail patch series or recognize it as already applied."""
    _validate_manifest(manifest)
    paths, expected_patch_ids = _validate_patch_files(patch_dir, manifest)
    if not source_root.is_dir():
        raise PatchValidationError(f"source root does not exist: {source_root}")
    if _git(source_root, "status", "--porcelain", "--untracked-files=all"):
        raise PatchValidationError("source tree must be clean before patching")

    current = _git(source_root, "rev-parse", "HEAD")
    base_commit = str(manifest["base_commit"])
    if current != base_commit:
        merge_base = _git(source_root, "merge-base", base_commit, current)
        applied_commits = _git(
            source_root, "rev-list", "--reverse", f"{base_commit}..{current}"
        ).splitlines()
        applied_patch_ids = [
            _stable_patch_id(
                _git(source_root, "show", "--pretty=format:", "--binary", commit)
            )
            for commit in applied_commits
        ]
        if merge_base == base_commit and applied_patch_ids == expected_patch_ids:
            return []
        raise PatchValidationError(
            f"source HEAD {current} is neither base {base_commit} nor this patch stack"
        )

    try:
        _git(source_root, "am", "--3way", *(str(path) for path in paths))
    except PatchValidationError:
        subprocess.run(
            ["git", "am", "--abort"],
            cwd=source_root,
            check=False,
            capture_output=True,
            text=True,
        )
        raise
    if _git(source_root, "status", "--porcelain", "--untracked-files=all"):
        raise PatchValidationError("patch application left a dirty source tree")
    return [str(row["commit"]) for row in manifest["patches"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--patch-dir", type=Path, required=True)
    parsed = parser.parse_args()
    manifest = load_manifest(parsed.manifest)
    applied = apply_patch_stack(
        source_root=parsed.source_root,
        patch_dir=parsed.patch_dir,
        manifest=manifest,
    )
    print(json.dumps({"applied_commits": applied}, sort_keys=True))


if __name__ == "__main__":
    main()
