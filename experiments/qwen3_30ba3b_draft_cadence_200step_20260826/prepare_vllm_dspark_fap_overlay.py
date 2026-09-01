#!/usr/bin/env python3
"""Create a source-verified vLLM overlay for the DSpark Blackwell FAP guard."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import uuid
from pathlib import Path


UPSTREAM_PR = "https://github.com/vllm-project/vllm/pull/48167"
RELATIVE_BACKEND = Path("v1/attention/backends/flashinfer.py")
RECEIPT_NAME = "dspark-fap-vllm-48167-attention-guard.json"
STOCK_GUARD = """        if has_trtllm_support:
            return AttentionCGSupport.UNIFORM_BATCH
"""
PATCHED_GUARD = """        # trtllm-gen only supports causal attention.
        if has_trtllm_support and not vllm_config.attention_config.use_non_causal:
            return AttentionCGSupport.UNIFORM_BATCH
"""


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


def prepare_overlay(source_package: Path, overlay_root: Path) -> Path:
    source_package = source_package.resolve()
    overlay_root = overlay_root.resolve()
    source_backend = source_package / RELATIVE_BACKEND
    if not source_backend.is_file():
        raise FileNotFoundError(f"missing vLLM backend: {source_backend}")
    source_text = source_backend.read_text()
    stock_count = source_text.count(STOCK_GUARD)
    patched_count = source_text.count(PATCHED_GUARD)
    if (stock_count, patched_count) == (1, 0):
        patched_text = source_text.replace(STOCK_GUARD, PATCHED_GUARD, 1)
        status = "applied"
    elif (stock_count, patched_count) == (0, 1):
        patched_text = source_text
        status = "already-patched"
    else:
        raise ValueError(
            "expected vLLM 0.25.1 attention guard exactly once; "
            f"found stock={stock_count}, patched={patched_count}"
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
        patched_backend = overlay_package / RELATIVE_BACKEND
        patched_backend.write_text(patched_text)
        receipt = {
            "overlay_package": str(overlay_root / "vllm"),
            "patched_file": str(RELATIVE_BACKEND),
            "patched_sha256": sha256(patched_backend),
            "schema_version": 1,
            "source_package": str(source_package),
            "source_sha256": sha256(source_backend),
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
    overlay_root = os.environ.get("Q30_VLLM_OVERLAY")
    parser.add_argument(
        "--overlay-root",
        type=Path,
        default=Path(overlay_root) if overlay_root else None,
    )
    args = parser.parse_args()
    if args.overlay_root is None:
        parser.error(
            "--overlay-root is required when Q30_VLLM_OVERLAY is not set"
        )
    return args


def main() -> None:
    args = parse_args()
    source_package = args.source_package or installed_vllm_package()
    overlay_package = prepare_overlay(source_package, args.overlay_root)
    print((overlay_package.parent / RECEIPT_NAME).read_text(), end="")


if __name__ == "__main__":
    main()
