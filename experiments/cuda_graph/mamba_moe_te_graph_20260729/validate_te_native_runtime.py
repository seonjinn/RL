#!/usr/bin/env python3

import argparse
import importlib
import json
from pathlib import Path


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--site-packages", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-wheel-sha256", required=True)
    parser.add_argument("--expected-image", required=True)
    parser.add_argument("--expected-image-sha256", required=True)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--validate-imports", action="store_true")
    args = parser.parse_args()

    provenance = json.loads(args.provenance.read_text())
    expected = {
        "commit": args.expected_commit,
        "wheel_sha256": args.expected_wheel_sha256,
        "image": args.expected_image,
        "image_sha256": args.expected_image_sha256,
        "install_prefix": str(args.site_packages.parent),
    }
    mismatches = {
        key: {"expected": value, "actual": provenance.get(key)}
        for key, value in expected.items()
        if provenance.get(key) != value
    }
    if mismatches:
        raise SystemExit(
            "Transformer Engine native runtime provenance mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )

    required = (
        args.site_packages / "transformer_engine" / "__init__.py",
        args.site_packages / "transformer_engine" / "libtransformer_engine.so",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Transformer Engine native runtime is incomplete: {missing}")

    if not args.validate_imports:
        return

    transformer_engine = importlib.import_module("transformer_engine")
    transformer_engine_pytorch = importlib.import_module("transformer_engine.pytorch")
    transformer_engine_torch = importlib.import_module("transformer_engine_torch")
    if transformer_engine.__version__ != args.expected_version:
        raise SystemExit(
            "Transformer Engine version mismatch: "
            f"expected {args.expected_version}, got {transformer_engine.__version__}"
        )

    from transformer_engine.common import _get_shared_object_file

    resolved = {
        "transformer_engine": Path(transformer_engine.__file__),
        "transformer_engine.pytorch": Path(transformer_engine_pytorch.__file__),
        "transformer_engine_torch": Path(transformer_engine_torch.__file__),
        "libtransformer_engine": Path(_get_shared_object_file("core")),
    }
    outside = {
        name: str(path.resolve())
        for name, path in resolved.items()
        if not _within(path, args.site_packages)
    }
    if outside:
        raise SystemExit(
            "Transformer Engine resolved outside the native runtime: "
            + json.dumps(outside, sort_keys=True)
        )
    for name, path in resolved.items():
        print(f"{name}={path.resolve()}")


if __name__ == "__main__":
    main()
