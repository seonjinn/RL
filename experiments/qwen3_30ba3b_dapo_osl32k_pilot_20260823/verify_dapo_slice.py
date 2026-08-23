"""Materialize and verify an immutable first-64-row DAPOMath17K slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_rows(path: Path, source_format: str, count: int) -> Iterator[dict[str, Any]]:
    if source_format == "jsonl":
        with path.open() as stream:
            for index, line in enumerate(stream):
                if index == count:
                    return
                yield json.loads(line)
        return
    if source_format != "parquet":
        raise SystemExit(f"unsupported source format: {source_format}")
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise SystemExit("pyarrow is required for the pinned parquet source") from error
    batch = next(pq.ParquetFile(path).iter_batches(batch_size=count))
    yield from batch.to_pylist()


def canonical_row(row: dict[str, Any]) -> bytes:
    try:
        prompt = row["prompt"]
        if not isinstance(prompt, list) or not prompt:
            raise TypeError("prompt is not a non-empty list")
        output = {
            "input": prompt[0]["content"],
            "output": row["reward_model"]["ground_truth"],
        }
    except (KeyError, TypeError) as error:
        raise SystemExit(f"source row schema mismatch: {error}") from error
    if not all(isinstance(value, str) for value in output.values()):
        raise SystemExit("DAPO input and ground truth must both be strings")
    return (
        json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def verify_file(path: Path, expected: dict[str, Any], label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"missing {label}: {path}")
    actual_size = path.stat().st_size
    if actual_size != expected["size"]:
        raise SystemExit(f"{label} size mismatch: {actual_size} != {expected['size']}")
    actual_sha = sha256(path)
    if actual_sha != expected["sha256"]:
        raise SystemExit(f"{label} sha256 mismatch: {actual_sha} != {expected['sha256']}")


parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--identity-file", type=Path, required=True)
parser.add_argument("--verify-only", action="store_true")
args = parser.parse_args()

identity = json.loads(args.identity_file.read_text())
source_identity = identity["source"]
slice_identity = identity["slice"]
verify_file(args.source, source_identity, "DAPOMath17K source")
if not args.verify_only:
    count = slice_identity["rows"]
    if slice_identity["source_order"] != list(range(count)):
        raise SystemExit("slice source_order is not the immutable leading range")
    rows = list(source_rows(args.source, source_identity["format"], count))
    if len(rows) != count:
        raise SystemExit(f"source yielded {len(rows)} rows, expected {count}")
    contents = b"".join(canonical_row(row) for row in rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, args.output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
verify_file(args.output, slice_identity, "DAPOMath17K slice")
with args.output.open() as stream:
    rows = [json.loads(line) for line in stream]
if len(rows) != slice_identity["rows"]:
    raise SystemExit(f"slice row count mismatch: {len(rows)}")
if any(set(row) != {"input", "output"} for row in rows):
    raise SystemExit("slice row schema mismatch")
print(
    f"DATA_IDENTITY_GATE_PASS source_sha={source_identity['sha256']} "
    f"slice_sha={slice_identity['sha256']} rows={len(rows)} seed={slice_identity['seed']}"
)
