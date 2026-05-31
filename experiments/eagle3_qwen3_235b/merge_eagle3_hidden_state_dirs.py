#!/usr/bin/env python3
"""Merge Eagle3 hidden-state directories with symlinks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    linked = 0
    skipped_existing = 0
    collisions: list[str] = []
    inputs: list[dict[str, object]] = []

    for input_dir in args.input_dir:
        if not input_dir.is_dir():
            raise FileNotFoundError(f"input dir does not exist: {input_dir}")
        files = sorted(input_dir.glob("*.pt"))
        inputs.append({"path": str(input_dir), "pt_files": len(files)})
        for source in files:
            dest = args.output_dir / source.name
            if dest.exists() or dest.is_symlink():
                if args.replace:
                    dest.unlink()
                else:
                    if dest.resolve() == source.resolve():
                        skipped_existing += 1
                        continue
                    collisions.append(source.name)
                    continue
            os.symlink(source, dest)
            linked += 1

    report = {
        "output_dir": str(args.output_dir),
        "inputs": inputs,
        "linked": linked,
        "skipped_existing": skipped_existing,
        "collision_count": len(collisions),
        "collisions": collisions[:200],
        "total_pt_files": len(list(args.output_dir.glob("*.pt"))),
        "status": "pass" if not collisions else "fail",
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if collisions:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
