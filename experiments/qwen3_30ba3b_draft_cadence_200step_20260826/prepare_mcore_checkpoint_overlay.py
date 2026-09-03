#!/usr/bin/env python3
"""Prepare a pinned node-local MCore overlay for checkpoint serialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any


PATCHED_FILE = Path("megatron/core/optimizer/distrib_optimizer.py")
RECEIPT_NAME = "mcore-precision-aware-lazy-state-checkpoint.json"
EXPECTED_SOURCE_SHA256 = "40cae307ccb2a3f484ebd5d28ee8549eb572d1b0ba6e91dc5661c137bf560907"
EXPECTED_PATCH_SHA256 = "912a66662e235b6b01fac5df4b4881efa1784f7a06cc5057c1f41f215376abe2"
EXPECTED_PATCHED_SHA256 = "b35932e4f025a83bedb0e1a6f6cb21076bbb156e54241ff6aef84a38348e3541"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")


def _validate_patch_targets(patch_path: Path) -> None:
    paths: set[str] = set()
    with patch_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("--- a/") or line.startswith("+++ b/"):
                paths.add(line[6:].strip())
    expected = {PATCHED_FILE.as_posix()}
    if paths != expected:
        raise RuntimeError(f"patch targets must be exactly {sorted(expected)}, got {sorted(paths)}")


def _expected_receipt(
    *,
    source_root: Path,
    overlay_root: Path,
    patch_path: Path,
    expected_source_sha256: str,
    expected_patch_sha256: str,
    expected_patched_sha256: str,
) -> dict[str, Any]:
    return {
        "overlay_root": str(overlay_root.resolve()),
        "patch_path": str(patch_path.resolve()),
        "patch_sha256": expected_patch_sha256,
        "patched_file": PATCHED_FILE.as_posix(),
        "patched_sha256": expected_patched_sha256,
        "schema_version": 1,
        "source_root": str(source_root.resolve()),
        "source_sha256": expected_source_sha256,
        "status": "applied",
    }


def _validate_existing_overlay(overlay_root: Path, expected_receipt: dict[str, Any]) -> Path:
    if overlay_root.is_symlink() or not overlay_root.is_dir():
        raise RuntimeError(f"existing overlay is not a real directory: {overlay_root}")
    receipt_path = overlay_root / RECEIPT_NAME
    patched_file = overlay_root / PATCHED_FILE
    try:
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise RuntimeError(f"existing overlay has no valid receipt: {overlay_root}")
        if receipt_path.stat().st_size > 64 * 1024:
            raise RuntimeError(f"overlay receipt is unexpectedly large: {receipt_path}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"existing overlay has no valid receipt: {overlay_root}") from error
    if receipt != expected_receipt:
        raise RuntimeError(f"existing overlay receipt does not match requested inputs: {overlay_root}")
    if patched_file.is_symlink() or not patched_file.is_file():
        raise RuntimeError(f"existing patched MCore file is not a regular file: {patched_file}")
    _require_sha256(
        patched_file,
        str(expected_receipt["patched_sha256"]),
        "existing patched MCore file",
    )
    return receipt_path


def prepare_overlay(
    *,
    source_root: Path,
    overlay_root: Path,
    patch_path: Path,
    expected_source_sha256: str = EXPECTED_SOURCE_SHA256,
    expected_patch_sha256: str = EXPECTED_PATCH_SHA256,
    expected_patched_sha256: str = EXPECTED_PATCHED_SHA256,
) -> Path:
    source_root = source_root.resolve(strict=True)
    patch_path = patch_path.resolve(strict=True)
    overlay_root = overlay_root.absolute()
    source_package = source_root / "megatron"
    source_file = source_root / PATCHED_FILE
    if not source_package.is_dir() or source_package.is_symlink():
        raise RuntimeError(f"MCore source package is not a real directory: {source_package}")
    if not source_file.is_file() or source_file.is_symlink():
        raise RuntimeError(f"MCore source file is not a regular file: {source_file}")

    _require_sha256(source_file, expected_source_sha256, "source MCore file")
    _require_sha256(patch_path, expected_patch_sha256, "patch")
    _validate_patch_targets(patch_path)
    receipt = _expected_receipt(
        source_root=source_root,
        overlay_root=overlay_root,
        patch_path=patch_path,
        expected_source_sha256=expected_source_sha256,
        expected_patch_sha256=expected_patch_sha256,
        expected_patched_sha256=expected_patched_sha256,
    )

    if overlay_root.exists() or overlay_root.is_symlink():
        return _validate_existing_overlay(overlay_root, receipt)

    overlay_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{overlay_root.name}.", dir=overlay_root.parent)
    )
    try:
        shutil.copytree(source_package, temporary_root / "megatron", symlinks=True)
        temporary_target = temporary_root / PATCHED_FILE
        temporary_target.chmod(temporary_target.stat().st_mode | stat.S_IWUSR)
        subprocess.run(
            ["git", "-C", str(temporary_root), "apply", "--check", str(patch_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(temporary_root), "apply", str(patch_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        _require_sha256(temporary_target, expected_patched_sha256, "patched MCore file")
        receipt_path = temporary_root / RECEIPT_NAME
        with receipt_path.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            temporary_root.rename(overlay_root)
        except OSError:
            if overlay_root.exists():
                return _validate_existing_overlay(overlay_root, receipt)
            raise
        return overlay_root / RECEIPT_NAME
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--overlay-root", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt_path = prepare_overlay(
        source_root=args.source_root,
        overlay_root=args.overlay_root,
        patch_path=args.patch,
    )
    print(receipt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
