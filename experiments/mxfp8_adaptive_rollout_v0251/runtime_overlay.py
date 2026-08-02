from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import importlib.metadata
import json
import os
import shutil
import tempfile
from pathlib import Path

_MANIFEST_NAME = "runtime-overlay.json"
_SCHEMA_VERSION = 1


def _stable_extensions(package: Path) -> list[Path]:
    return sorted(package.glob("_C_stable_libtorch*.so"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _installed_vllm_package() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise ValueError("the locked environment does not contain vLLM")
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def prepare_runtime_overlay(
    *,
    installed_package: Path,
    source_package: Path,
    destination_base: Path,
    source_revision: str,
    installed_version: str = "unknown",
) -> Path:
    installed_package = installed_package.resolve()
    source_package = source_package.resolve()
    destination_base = destination_base.resolve()
    if not installed_package.is_dir():
        raise ValueError(f"installed vLLM package is missing: {installed_package}")
    if not source_package.is_dir():
        raise ValueError(f"custom vLLM package is missing: {source_package}")

    extensions = _stable_extensions(installed_package)
    if not extensions:
        raise ValueError(
            "installed vLLM package does not contain _C_stable_libtorch"
        )

    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "source_revision": source_revision,
        "installed_version": installed_version,
        "installed_package": str(installed_package),
        "stable_extensions": [
            {
                "name": path.name,
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in extensions
        ],
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
    overlay_id = hashlib.sha256(manifest_bytes).hexdigest()[:24]
    destination_root = destination_base / overlay_id
    manifest_path = destination_root / _MANIFEST_NAME
    destination_base.mkdir(parents=True, exist_ok=True)
    lock_path = destination_base / f".{overlay_id}.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            runtime_package = destination_root / "vllm"
            if existing == manifest and _stable_extensions(runtime_package):
                return destination_root
        if destination_root.exists():
            raise ValueError(
                f"immutable runtime overlay is incomplete: {destination_root}"
            )

        temporary_root = Path(
            tempfile.mkdtemp(prefix=f".{overlay_id}.", dir=destination_base)
        )
        try:
            runtime_package = temporary_root / "vllm"
            shutil.copytree(installed_package, runtime_package, symlinks=True)
            shutil.copytree(
                source_package,
                runtime_package,
                dirs_exist_ok=True,
                symlinks=True,
            )
            for cache_dir in runtime_package.rglob("__pycache__"):
                shutil.rmtree(cache_dir)
            if not _stable_extensions(runtime_package):
                raise ValueError("runtime overlay lost _C_stable_libtorch")
            (temporary_root / _MANIFEST_NAME).write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_root, destination_root)
        finally:
            if temporary_root.exists():
                shutil.rmtree(temporary_root)
    return destination_root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-base", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    runtime_root = prepare_runtime_overlay(
        installed_package=_installed_vllm_package(),
        source_package=args.source_root / "vllm",
        destination_base=args.destination_base,
        source_revision=args.source_revision,
        installed_version=importlib.metadata.version("vllm"),
    )
    print(runtime_root)


if __name__ == "__main__":
    main()
