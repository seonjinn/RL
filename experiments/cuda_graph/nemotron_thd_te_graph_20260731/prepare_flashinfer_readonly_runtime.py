#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


_PATCH_MARKER = "NeMo-RL: immutable runtimes reuse an existing cubin symlink"
_LOCKED_SYMLINK_PREFIX = """    link.parent.mkdir(parents=True, exist_ok=True)
    lock_path = str(link) + ".lock"
    lock = filelock.FileLock(lock_path, timeout=60)
"""
_READONLY_SYMLINK_PREFIX = f"""    # {_PATCH_MARKER}.
    if link.is_symlink() and link.resolve() == target.resolve():
        return

    link.parent.mkdir(parents=True, exist_ok=True)
    lock_path = str(link) + ".lock"
    lock = filelock.FileLock(lock_path, timeout=60)
"""


def _flashinfer_cubin_loader_path() -> Path:
    package = importlib.util.find_spec("flashinfer")
    if package is None or not package.submodule_search_locations:
        raise RuntimeError("FlashInfer is missing from the staged vLLM runtime")
    loader = (
        Path(next(iter(package.submodule_search_locations))) / "jit/cubin_loader.py"
    )
    if loader.is_symlink() or not loader.is_file():
        raise RuntimeError(f"Unsafe FlashInfer cubin loader path: {loader}")
    return loader


def patch_flashinfer_cubin_loader(loader: Path) -> None:
    source = loader.read_text()
    if _PATCH_MARKER in source:
        return
    if source.count(_LOCKED_SYMLINK_PREFIX) != 1:
        raise RuntimeError(
            "FlashInfer cubin symlink anchor changed; refusing an unverified patch"
        )
    loader.write_text(source.replace(_LOCKED_SYMLINK_PREFIX, _READONLY_SYMLINK_PREFIX))


def _prepare_flashinfer_export_symlink(
    cubin_root: Path, target: Path, alias: str, export_name: str
) -> Path:
    resolved_root = cubin_root.resolve(strict=True)
    resolved_target = target.resolve(strict=True)
    if not resolved_target.is_dir() or not resolved_target.is_relative_to(
        resolved_root
    ):
        raise RuntimeError(
            f"FlashInfer {export_name} export target escaped the cubin package: "
            f"{target}"
        )

    link = resolved_root / alias
    if link.is_symlink():
        if link.resolve() != resolved_target:
            raise RuntimeError(
                f"FlashInfer {export_name} export symlink has a stale target: {link}"
            )
        return link
    if link.exists():
        raise RuntimeError(
            f"FlashInfer {export_name} export alias is not a symlink: {link}"
        )

    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(resolved_target)
    return link


def prepare_flashinfer_bmm_symlink(cubin_root: Path, target: Path) -> Path:
    return _prepare_flashinfer_export_symlink(
        cubin_root,
        target,
        "flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export",
        "BMM",
    )


def prepare_flashinfer_gemm_symlink(cubin_root: Path, target: Path) -> Path:
    return _prepare_flashinfer_export_symlink(
        cubin_root,
        target,
        "flashinfer/trtllm/gemm/trtllmGen_gemm_export",
        "GEMM",
    )


def _installed_export_paths() -> tuple[Path, Path, Path]:
    import flashinfer_cubin
    from flashinfer.artifacts import ArtifactPath

    cubin_root = Path(flashinfer_cubin.get_cubin_dir())
    bmm_target = (
        cubin_root / ArtifactPath.TRTLLM_GEN_BMM / "include/trtllmGen_bmm_export"
    )
    gemm_target = (
        cubin_root / ArtifactPath.TRTLLM_GEN_GEMM / "include/trtllmGen_gemm_export"
    )
    return cubin_root, bmm_target, gemm_target


def _verify_installed_modules(expected_links: tuple[Path, Path]) -> None:
    from flashinfer.jit.gemm.core import gen_trtllm_gen_gemm_module
    from flashinfer.jit.fused_moe import gen_trtllm_gen_fused_moe_sm100_module

    gen_trtllm_gen_fused_moe_sm100_module()
    gen_trtllm_gen_gemm_module()
    for expected_link in expected_links:
        if not expected_link.is_symlink():
            raise RuntimeError(
                f"FlashInfer export symlink disappeared: {expected_link}"
            )
        lock = Path(f"{expected_link}.lock")
        if lock.exists() or lock.is_symlink():
            raise RuntimeError(f"FlashInfer attempted a runtime lock mutation: {lock}")


def _prepare_installed_aliases() -> tuple[Path, Path]:
    cubin_root, bmm_target, gemm_target = _installed_export_paths()
    return (
        prepare_flashinfer_bmm_symlink(cubin_root, bmm_target),
        prepare_flashinfer_gemm_symlink(cubin_root, gemm_target),
    )


def prepare() -> None:
    loader = _flashinfer_cubin_loader_path()
    patch_flashinfer_cubin_loader(loader)
    _verify_installed_modules(_prepare_installed_aliases())


def verify() -> None:
    loader = _flashinfer_cubin_loader_path()
    if _PATCH_MARKER not in loader.read_text():
        raise RuntimeError("FlashInfer read-only cubin patch is missing")
    _verify_installed_modules(_prepare_installed_aliases())


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.prepare:
        prepare()
    else:
        verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
