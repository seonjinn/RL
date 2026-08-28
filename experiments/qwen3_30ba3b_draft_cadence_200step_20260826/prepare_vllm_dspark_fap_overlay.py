#!/usr/bin/env python3
"""Create a source-verified vLLM overlay for DSpark on Blackwell FAP."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path


UPSTREAM_PR = "https://github.com/vllm-project/vllm/pull/48167"
RUNTIME_PATCH_NAME = "vllm-0.25.1-pr48167-runtime.patch"
EXPECTED_RUNTIME_PATCH_SHA256 = (
    "504730a52614fddeb8ea899ec37a0aa820dcbc3a57c704fc13f5834fcc07b317"
)
RECEIPT_NAME = "dspark-fap-vllm-48167-runtime.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def installed_vllm_package() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise RuntimeError("installed vLLM package is unavailable")
    package = Path(spec.origin).resolve().parent
    if package.name != "vllm":
        raise RuntimeError(f"unexpected vLLM package path: {package}")
    return package


def runtime_patch_files(patch_path: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    for line in patch_path.read_text().splitlines():
        if not line.startswith("+++ b/"):
            continue
        relative = Path(line.removeprefix("+++ b/"))
        if relative.is_absolute() or not relative.parts or relative.parts[0] != "vllm":
            raise ValueError(f"runtime patch escapes vLLM package: {relative}")
        files.append(relative)
    if not files or len(files) != len(set(files)):
        raise ValueError("runtime patch has no files or duplicate file entries")
    return tuple(files)


def git_apply_check(root: Path, patch_path: Path, *, reverse: bool) -> bool:
    command = ["git", "apply", "--check", "--whitespace=nowarn"]
    if reverse:
        command.append("--reverse")
    command.append(str(patch_path))
    return subprocess.run(
        command,
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    ).returncode == 0


def apply_patch(root: Path, patch_path: Path) -> None:
    subprocess.run(
        ["git", "apply", "--whitespace=nowarn", str(patch_path)],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )


def prepare_overlay(
    source_package: Path,
    overlay_root: Path,
    patch_path: Path,
    *,
    expected_patch_sha256: str,
) -> Path:
    source_package = source_package.resolve()
    overlay_root = overlay_root.resolve()
    patch_path = patch_path.resolve()
    if not patch_path.is_file():
        raise FileNotFoundError(f"missing vLLM runtime patch: {patch_path}")
    actual_patch_sha256 = sha256(patch_path)
    if actual_patch_sha256 != expected_patch_sha256:
        raise ValueError(
            "runtime patch digest mismatch: "
            f"expected {expected_patch_sha256}, got {actual_patch_sha256}"
        )
    patched_paths = runtime_patch_files(patch_path)
    for relative in patched_paths:
        source_file = source_package.parent / relative
        if not source_file.is_file():
            raise FileNotFoundError(f"installed vLLM runtime file is absent: {relative}")

    overlay_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = overlay_root.with_name(
        f".{overlay_root.name}.{uuid.uuid4().hex}.tmp"
    )
    if overlay_root.exists():
        raise FileExistsError(f"overlay root already exists: {overlay_root}")
    try:
        overlay_package = temporary_root / "vllm"
        shutil.copytree(source_package, overlay_package, symlinks=True)
        if git_apply_check(temporary_root, patch_path, reverse=False):
            apply_patch(temporary_root, patch_path)
            status = "applied"
        elif git_apply_check(temporary_root, patch_path, reverse=True):
            status = "already-patched"
        else:
            raise ValueError(
                "vLLM #48167 runtime patch does not apply cleanly in either direction"
            )
        patched_files = {
            str(relative): sha256(temporary_root / relative)
            for relative in patched_paths
        }
        receipt = {
            "overlay_package": str(overlay_root / "vllm"),
            "patch_sha256": actual_patch_sha256,
            "patched_files": patched_files,
            "schema_version": 2,
            "source_package": str(source_package),
            "status": status,
            "upstream_pr": UPSTREAM_PR,
        }
        receipt_path = temporary_root / RECEIPT_NAME
        with receipt_path.open("x") as stream:
            stream.write(json.dumps(receipt, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_root, overlay_root)
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
    return overlay_root / "vllm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-package", type=Path)
    parser.add_argument("--overlay-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_package = args.source_package or installed_vllm_package()
    patch_path = Path(__file__).resolve().parent / "patches" / RUNTIME_PATCH_NAME
    overlay_package = prepare_overlay(
        source_package,
        args.overlay_root,
        patch_path,
        expected_patch_sha256=EXPECTED_RUNTIME_PATCH_SHA256,
    )
    print((overlay_package.parent / RECEIPT_NAME).read_text(), end="")


if __name__ == "__main__":
    main()
