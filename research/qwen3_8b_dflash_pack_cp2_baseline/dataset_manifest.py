#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DATASET_REVISION = "65877096c24ffa7abc4e4fa5edb95cf3413a5674"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    cache_root: Path,
    *,
    source_parquet: Path | None = None,
    expected_source_parquet_sha256: str | None = None,
) -> dict[str, Any]:
    root = cache_root.resolve(strict=True)
    files = [path for path in sorted(root.rglob("*")) if path.is_file()]
    if not files:
        raise ValueError(f"dataset cache is empty: {root}")
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    ]
    if expected_source_parquet_sha256 is not None:
        candidate = source_parquet
        if candidate is None:
            parquet_files = [path for path in files if path.suffix == ".parquet"]
            if len(parquet_files) != 1:
                raise ValueError("source parquet SHA256 cannot be verified uniquely")
            candidate = parquet_files[0]
        actual = _sha256(candidate.resolve(strict=True))
        if actual != expected_source_parquet_sha256:
            raise ValueError(
                f"source parquet SHA256 mismatch: {actual} != "
                f"{expected_source_parquet_sha256}"
            )
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": 1,
        "dataset_revision": DATASET_REVISION,
        "cache_root": str(root),
        "files": entries,
        "tree_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "source_parquet_sha256": expected_source_parquet_sha256,
    }


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def verify_pair(left: Path, right: Path) -> None:
    first = json.loads(left.read_text())
    second = json.loads(right.read_text())
    keys = ("dataset_revision", "files", "tree_sha256", "source_parquet_sha256")
    if any(first.get(key) != second.get(key) for key in keys):
        raise ValueError("cache manifest mismatch between paired arms")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--cache-root", type=Path, required=True)
    build.add_argument("--source-parquet", type=Path, required=True)
    build.add_argument("--source-parquet-sha256", required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify-pair")
    verify.add_argument("--reference", type=Path, required=True)
    verify.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify-pair":
        verify_pair(args.reference, args.candidate)
        print("dataset_cache_pair=matched")
        return
    payload = build_manifest(
        args.cache_root,
        source_parquet=args.source_parquet,
        expected_source_parquet_sha256=args.source_parquet_sha256,
    )
    write_manifest(args.output, payload)
    print(f"dataset_tree_sha256={payload['tree_sha256']}")


if __name__ == "__main__":
    main()
