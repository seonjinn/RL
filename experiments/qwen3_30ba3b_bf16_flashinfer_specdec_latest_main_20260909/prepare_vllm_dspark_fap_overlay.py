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
from collections.abc import Mapping, Sequence
from pathlib import Path


UPSTREAM_PR = "https://github.com/vllm-project/vllm/pull/48167"
RUNTIME_PATCH_NAME = "vllm-0.25.1-pr48167-runtime.patch"
FOLLOWUP_COMMIT = "bf372f9bb5b8d0aed609332cd640c99afd440a15"
FOLLOWUP_PATCH_NAME = "vllm-0.25.1-pr48167-group-causality-followup.patch"
EXPECTED_RUNTIME_PATCH_SHA256 = (
    "504730a52614fddeb8ea899ec37a0aa820dcbc3a57c704fc13f5834fcc07b317"
)
EXPECTED_FOLLOWUP_PATCH_SHA256 = (
    "8e5ff0e385ee44cf71e1e07031e5cd19658b29eb7b90bc172a4754c599d1dd90"
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
    return (
        subprocess.run(
            command,
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


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
    followup_patch_path: Path | None = None,
    expected_followup_patch_sha256: str | None = None,
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
    if (followup_patch_path is None) != (expected_followup_patch_sha256 is None):
        raise ValueError("follow-up patch path and digest must be provided together")
    followup_paths: tuple[Path, ...] = ()
    actual_followup_patch_sha256: str | None = None
    if followup_patch_path is not None:
        followup_patch_path = followup_patch_path.resolve()
        if not followup_patch_path.is_file():
            raise FileNotFoundError(
                f"missing vLLM follow-up patch: {followup_patch_path}"
            )
        actual_followup_patch_sha256 = sha256(followup_patch_path)
        if actual_followup_patch_sha256 != expected_followup_patch_sha256:
            raise ValueError(
                "follow-up patch digest mismatch: "
                f"expected {expected_followup_patch_sha256}, "
                f"got {actual_followup_patch_sha256}"
            )
        followup_paths = runtime_patch_files(followup_patch_path)
    for relative in patched_paths:
        source_file = source_package.parent / relative
        if not source_file.is_file():
            raise FileNotFoundError(
                f"installed vLLM runtime file is absent: {relative}"
            )

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
        followup_status: str | None = None
        if followup_patch_path is not None:
            if git_apply_check(temporary_root, followup_patch_path, reverse=False):
                apply_patch(temporary_root, followup_patch_path)
                followup_status = "applied"
            elif git_apply_check(temporary_root, followup_patch_path, reverse=True):
                followup_status = "already-patched"
            else:
                raise ValueError(
                    "vLLM group-causality follow-up patch does not apply cleanly "
                    "in either direction"
                )
        patched_files = {
            str(relative): sha256(temporary_root / relative)
            for relative in patched_paths
        }
        receipt = {
            "overlay_package": str(overlay_root / "vllm"),
            "patch_sha256": actual_patch_sha256,
            "patched_files": patched_files,
            "schema_version": 3,
            "source_package": str(source_package),
            "status": status,
            "upstream_pr": UPSTREAM_PR,
        }
        if followup_patch_path is not None:
            receipt.update(
                {
                    "followup_commit": FOLLOWUP_COMMIT,
                    "followup_patch_sha256": actual_followup_patch_sha256,
                    "followup_patched_files": {
                        str(relative): sha256(temporary_root / relative)
                        for relative in followup_paths
                    },
                    "followup_status": followup_status,
                }
            )
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


def parse_args(
    argv: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-package", type=Path)
    parser.add_argument("--overlay-root", type=Path)
    args = parser.parse_args(argv)
    environment = os.environ if environ is None else environ
    if args.overlay_root is None:
        overlay_root = environment.get("Q30_VLLM_OVERLAY")
        if not overlay_root:
            parser.error("--overlay-root or Q30_VLLM_OVERLAY is required")
        args.overlay_root = Path(overlay_root)
    return args


def main() -> None:
    args = parse_args()
    source_package = args.source_package or installed_vllm_package()
    patch_path = Path(__file__).resolve().parent / "patches" / RUNTIME_PATCH_NAME
    followup_patch_path = (
        Path(__file__).resolve().parent / "patches" / FOLLOWUP_PATCH_NAME
    )
    overlay_package = prepare_overlay(
        source_package,
        args.overlay_root,
        patch_path,
        expected_patch_sha256=EXPECTED_RUNTIME_PATCH_SHA256,
        followup_patch_path=followup_patch_path,
        expected_followup_patch_sha256=EXPECTED_FOLLOWUP_PATCH_SHA256,
    )
    print((overlay_package.parent / RECEIPT_NAME).read_text(), end="")


if __name__ == "__main__":
    main()
