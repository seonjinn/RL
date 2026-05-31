#!/usr/bin/env python3
"""Merge generated training-conversation JSONL chunks.

The direct vLLM target-generation jobs are intentionally chunked so they can
fit within the cluster's four-hour batch wall time. This script joins those
chunk files into the single JSONL consumed by the Speculators pipeline while
checking for missing, duplicate, or malformed conversation ids.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("inputs", type=Path, nargs="+")
    return parser.parse_args()


def record_id(record: dict[str, Any], fallback: str) -> str:
    value = record.get("conversation_id") or record.get("id")
    if not isinstance(value, str) or not value:
        return fallback
    return value


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    missing_inputs: list[str] = []
    duplicate_ids: list[str] = []
    input_counts: dict[str, int] = {}
    written = 0

    with args.output.open("w", encoding="utf-8") as out:
        for input_path in args.inputs:
            if not input_path.exists() or input_path.stat().st_size == 0:
                missing_inputs.append(str(input_path))
                continue
            count = 0
            with input_path.open(encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    rid = record_id(record, f"{input_path.name}:{line_no}")
                    if rid in seen:
                        duplicate_ids.append(rid)
                        continue
                    seen.add(rid)
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
                    written += 1
            input_counts[str(input_path)] = count

    status = "pass"
    if duplicate_ids:
        status = "fail"
    if missing_inputs and not args.allow_missing:
        status = "fail"
    if args.expected_count is not None and written != args.expected_count:
        status = "fail"

    summary = {
        "status": status,
        "output": str(args.output),
        "records_written": written,
        "expected_count": args.expected_count,
        "input_counts": input_counts,
        "missing_inputs": missing_inputs,
        "duplicate_ids": duplicate_ids[:100],
        "duplicate_count": len(duplicate_ids),
    }
    if args.summary_json:
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
