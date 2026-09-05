#!/usr/bin/env python3
"""Build a source-verified vLLM overlay with refit-aware sleep semantics."""

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


RECEIPT_NAME = "frozen-drafter-sleep-overlay.json"
RUNTIME_PATCH_NAME = "vllm-0.25.1-refit-aware-frozen-drafter-sleep.patch"
RUNTIME_PATCH_SHA256 = (
    "b61df83aa855edae9e36aef560b03dbd148aa703b326fe42a90c1fdd451564ef"
)
POLICY_MODULE_NAME = "frozen_drafter_sleep_policy.py"
POLICY_MODULE_SHA256 = (
    "4cdfb9adbb9dd2ec346460c437fce1a108c20ca8dfdcfa5dec391de136448e59"
)
DSPARK_PREREQUISITES = (
    (
        "vllm-0.25.1-pr48167-runtime.patch",
        "504730a52614fddeb8ea899ec37a0aa820dcbc3a57c704fc13f5834fcc07b317",
    ),
    (
        "vllm-0.25.1-pr48167-group-causality-followup.patch",
        "8e5ff0e385ee44cf71e1e07031e5cd19658b29eb7b90bc172a4754c599d1dd90",
    ),
)


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
    for line in patch_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("+++ b/"):
            continue
        relative = Path(line.removeprefix("+++ b/"))
        if relative.is_absolute() or not relative.parts or relative.parts[0] != "vllm":
            raise ValueError(f"runtime patch escapes vLLM package: {relative}")
        files.append(relative)
    if not files or len(files) != len(set(files)):
        raise ValueError("runtime patch has no files or duplicate file entries")
    return tuple(files)


def apply_verified_patch(
    root: Path,
    patch_path: Path,
    expected_sha256: str,
) -> tuple[str, tuple[Path, ...]]:
    actual_sha256 = sha256(patch_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"runtime patch digest mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    patched_paths = runtime_patch_files(patch_path)
    command = ["git", "apply", "--check", "--whitespace=nowarn", str(patch_path)]
    if subprocess.run(command, cwd=root, check=False).returncode == 0:
        subprocess.run(
            ["git", "apply", "--whitespace=nowarn", str(patch_path)],
            cwd=root,
            check=True,
        )
        return "applied", patched_paths
    reverse = [
        "git",
        "apply",
        "--check",
        "--reverse",
        "--whitespace=nowarn",
        str(patch_path),
    ]
    if subprocess.run(reverse, cwd=root, check=False).returncode == 0:
        return "already-patched", patched_paths
    raise ValueError(f"runtime patch does not apply cleanly: {patch_path}")


def prepare_overlay(
    *,
    source_package: Path,
    overlay_root: Path,
    runtime_patch_path: Path,
    expected_runtime_patch_sha256: str,
    policy_module_path: Path,
    expected_policy_module_sha256: str,
    prerequisite_patches: Sequence[tuple[Path, str]] = (),
) -> Path:
    source_package = source_package.resolve()
    overlay_root = overlay_root.resolve()
    runtime_patch_path = runtime_patch_path.resolve()
    policy_module_path = policy_module_path.resolve()
    if source_package.name != "vllm" or not source_package.is_dir():
        raise ValueError(f"invalid vLLM source package: {source_package}")
    if not policy_module_path.is_file():
        raise FileNotFoundError(f"missing sleep policy module: {policy_module_path}")
    actual_policy_sha256 = sha256(policy_module_path)
    if actual_policy_sha256 != expected_policy_module_sha256:
        raise ValueError(
            "policy module digest mismatch: "
            f"expected {expected_policy_module_sha256}, got {actual_policy_sha256}"
        )
    if overlay_root.exists():
        raise FileExistsError(f"overlay root already exists: {overlay_root}")

    overlay_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = overlay_root.with_name(
        f".{overlay_root.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        overlay_package = temporary_root / "vllm"
        shutil.copytree(source_package, overlay_package, symlinks=True)
        installed_policy = (
            overlay_package / "device_allocator" / "nemo_rl_frozen_drafter_sleep.py"
        )
        installed_policy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(policy_module_path, installed_policy)

        prerequisite_receipts: list[dict[str, object]] = []
        for prerequisite_path, prerequisite_sha256 in prerequisite_patches:
            resolved = prerequisite_path.resolve()
            status, patched_paths = apply_verified_patch(
                temporary_root,
                resolved,
                prerequisite_sha256,
            )
            prerequisite_receipts.append(
                {
                    "path": str(resolved),
                    "sha256": prerequisite_sha256,
                    "status": status,
                    "patched_files": [str(path) for path in patched_paths],
                }
            )

        status, patched_paths = apply_verified_patch(
            temporary_root,
            runtime_patch_path,
            expected_runtime_patch_sha256,
        )
        receipt = {
            "overlay_package": str(overlay_root / "vllm"),
            "patched_files": [str(path) for path in patched_paths],
            "policy_module_sha256": actual_policy_sha256,
            "prerequisite_patches": prerequisite_receipts,
            "runtime_patch_sha256": expected_runtime_patch_sha256,
            "schema_version": 1,
            "source_package": str(source_package),
            "status": status,
        }
        receipt_path = temporary_root / RECEIPT_NAME
        with receipt_path.open("x", encoding="utf-8") as stream:
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
        overlay_root = environment.get("Q235_VLLM_OVERLAY")
        if not overlay_root:
            parser.error("--overlay-root or Q235_VLLM_OVERLAY is required")
        args.overlay_root = Path(overlay_root)
    return args


def prerequisite_patches(
    script_dir: Path,
    *,
    dspark_enabled: bool,
) -> tuple[tuple[Path, str], ...]:
    if not dspark_enabled:
        return ()
    return tuple(
        (script_dir / "patches" / name, digest) for name, digest in DSPARK_PREREQUISITES
    )


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    source_package = args.source_package or installed_vllm_package()
    overlay_package = prepare_overlay(
        source_package=source_package,
        overlay_root=args.overlay_root,
        runtime_patch_path=script_dir / "patches" / RUNTIME_PATCH_NAME,
        expected_runtime_patch_sha256=RUNTIME_PATCH_SHA256,
        policy_module_path=script_dir / POLICY_MODULE_NAME,
        expected_policy_module_sha256=POLICY_MODULE_SHA256,
        prerequisite_patches=prerequisite_patches(
            script_dir,
            dspark_enabled=os.environ.get("Q235_DSPARK_FAP_OVERLAY") == "1",
        ),
    )
    print((overlay_package.parent / RECEIPT_NAME).read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
