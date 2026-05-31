#!/usr/bin/env python3
"""Build an exact unique training-conversation JSONL.

Primary target-generation chunks may contain a small number of repeated math
prompts.  This utility writes unique primary rows first, skips denylisted eval
prompts, then appends replacement rows until the requested count is reached.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--expected-count", type=int, default=500_000)
    parser.add_argument("--denylist-prompts-from", type=Path, action="append", default=[])
    parser.add_argument("--primary", type=Path, nargs="+", required=True)
    parser.add_argument("--replacement", type=Path, nargs="*", default=[])
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def normalize_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt.strip()).lower()


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(normalize_prompt(prompt).encode("utf-8")).hexdigest()


def prompt_from_record(record: dict[str, Any]) -> str:
    for key in ("prompt", "messages", "conversations"):
        value = record.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            for message in value:
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role", message.get("from", ""))).lower()
                if role in {"user", "human"}:
                    content = message.get("content", message.get("value"))
                    if isinstance(content, str):
                        return content
    return ""


def record_id(record: dict[str, Any], fallback: str) -> str:
    value = record.get("id") or record.get("conversation_id")
    return str(value) if value not in (None, "") else fallback


def load_denylist(paths: list[Path]) -> set[str]:
    denylist: set[str] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"denylist file does not exist: {path}")
        for _, record in iter_jsonl(path):
            prompt = prompt_from_record(record)
            if prompt:
                denylist.add(prompt_hash(prompt))
    return denylist


def main() -> int:
    args = parse_args()
    missing = [path for path in [*args.primary, *args.replacement] if not path.exists()]
    if missing:
        raise FileNotFoundError("missing input files:\n" + "\n".join(str(path) for path in missing))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    denylist = load_denylist(args.denylist_prompts_from)
    seen = set(denylist)
    primary_rows = 0
    replacement_rows = 0
    skipped_duplicate = 0
    skipped_denylist = 0
    skipped_empty_prompt = 0
    input_counts: dict[str, int] = {}
    written_ids: list[str] = []

    def maybe_write(out_f, record: dict[str, Any], input_path: Path, line_no: int, is_replacement: bool) -> bool:
        nonlocal replacement_rows, skipped_duplicate, skipped_denylist, skipped_empty_prompt
        prompt = prompt_from_record(record)
        rid = record_id(record, f"{input_path.name}:{line_no}")
        if not prompt:
            skipped_empty_prompt += 1
            return False
        key = prompt_hash(prompt)
        if key in denylist:
            skipped_denylist += 1
            return False
        if key in seen:
            skipped_duplicate += 1
            return False
        seen.add(key)
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        written_ids.append(rid)
        if is_replacement:
            replacement_rows += 1
        return True

    with args.output.open("w", encoding="utf-8") as out_f:
        for path in args.primary:
            count = 0
            for line_no, record in iter_jsonl(path):
                count += 1
                if len(written_ids) < args.expected_count:
                    maybe_write(out_f, record, path, line_no, False)
            input_counts[str(path)] = count

        primary_rows = len(written_ids)
        for path in args.replacement:
            count = 0
            for line_no, record in iter_jsonl(path):
                count += 1
                if len(written_ids) >= args.expected_count:
                    break
                maybe_write(out_f, record, path, line_no, True)
            input_counts[str(path)] = count
            if len(written_ids) >= args.expected_count:
                break

    status = "pass" if len(written_ids) == args.expected_count else "fail"
    summary = {
        "status": status,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "output": str(args.output),
        "expected_count": args.expected_count,
        "records_written": len(written_ids),
        "primary_unique_written_before_replacements": primary_rows,
        "replacement_rows_written": replacement_rows,
        "denylist_prompt_count": len(denylist),
        "skipped_duplicate_or_seen": skipped_duplicate,
        "skipped_denylist": skipped_denylist,
        "skipped_empty_prompt": skipped_empty_prompt,
        "input_counts": input_counts,
        "first_written_ids": written_ids[:5],
        "last_written_ids": written_ids[-5:],
    }
    if args.summary_json:
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
