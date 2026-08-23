"""Fail closed unless all pinned model files match their byte identities."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--artifact", choices=("target", "dflash", "dspark"), required=True)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--identity-file", type=Path, required=True)
parser.add_argument("--verify-content-sha", action="store_true")
args = parser.parse_args()

expected = json.loads(args.identity_file.read_text())[args.artifact]
for filename, metadata in expected.items():
    path = args.root / filename
    if not path.is_file():
        raise SystemExit(f"missing {args.artifact} file: {path}")
    if path.stat().st_size != metadata["size"]:
        raise SystemExit(f"{args.artifact} size mismatch: {path}")
    if args.verify_content_sha and sha256(path) != metadata["sha256"]:
        raise SystemExit(f"{args.artifact} sha256 mismatch: {path}")

print(
    f"MODEL_IDENTITY_GATE_PASS artifact={args.artifact} "
    f"files={len(expected)} content_sha={args.verify_content_sha}"
)
