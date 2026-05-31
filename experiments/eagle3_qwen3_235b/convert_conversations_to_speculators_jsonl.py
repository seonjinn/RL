#!/usr/bin/env python3
"""Convert training conversations to vLLM Speculators local JSONL.

Speculators' ``prepare_data.py`` can load local JSON/JSONL files and expects a
``conversations`` column. Its current preprocessing normalizes turns from either
``role/content`` or ShareGPT-style ``from/value`` into system/user/assistant
messages before applying the target model chat template and assistant loss mask.

This converter accepts the two schemas used in this workstream:

    {"conversation_id": "...", "messages": [...]}      # ModelOpt
    {"id": "...", "conversations": [...]}              # SpecForge/Speculators

and writes:

    {"id": "...", "conversations": [{"role": "...", "content": "..."}]}
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
    "model": "assistant",
    "system": "system",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="ModelOpt/SpecForge conversation JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Speculators local JSONL output")
    parser.add_argument("--model", default="Qwen/Qwen3-235B-A22B-Thinking-2507")
    parser.add_argument("--seq-length", type=int, default=16384)
    parser.add_argument("--prepared-output-dir", default="./speculators_training_data")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--sample-offset",
        type=int,
        default=0,
        help="Skip this many valid source rows before writing output. Useful for chunked training.",
    )
    parser.add_argument("--minimum-valid-tokens", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-assistant-chars", type=int, default=1)
    parser.add_argument("--include-metadata", action="store_true")
    parser.add_argument(
        "--drop-unsupported-roles",
        action="store_true",
        help="Drop unsupported roles instead of failing the row.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line_num, line in enumerate(fh, 1):
            text = line.strip()
            if not text:
                continue
            try:
                yield line_num, json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_num}: {exc}") from exc


def stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def record_id(record: dict[str, Any], line_num: int) -> str:
    for key in ("id", "conversation_id", "source_id", "instance_id"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    return f"row_{line_num}"


def record_messages(record: dict[str, Any]) -> tuple[Any, str | None]:
    for key in ("conversations", "messages"):
        if key in record:
            return record[key], key
    return None, None


def normalize_messages(
    raw_messages: Any,
    drop_unsupported_roles: bool,
) -> tuple[list[dict[str, str]], list[str]]:
    warnings: list[str] = []
    if not isinstance(raw_messages, list):
        return [], ["messages/conversations is not a list"]

    messages: list[dict[str, str]] = []
    for idx, item in enumerate(raw_messages):
        if not isinstance(item, dict):
            warnings.append(f"turn {idx} is not an object")
            continue
        raw_role = item.get("role", item.get("from"))
        raw_content = item.get("content", item.get("value"))
        if raw_role in (None, "") or raw_content in (None, ""):
            warnings.append(f"turn {idx} missing role/content")
            continue
        role = ROLE_MAP.get(str(raw_role).lower())
        if role is None:
            warning = f"turn {idx} unsupported role {raw_role!r}"
            if drop_unsupported_roles:
                warnings.append(warning)
                continue
            return [], [warning]
        content = stringify(raw_content).strip()
        if not content:
            warnings.append(f"turn {idx} blank content")
            continue
        messages.append({"role": role, "content": content})
    return messages, warnings


def assistant_chars(messages: list[dict[str, str]]) -> int:
    return sum(len(msg["content"].strip()) for msg in messages if msg["role"] == "assistant")


def prepare_data_command(args: argparse.Namespace) -> str:
    parts = [
        "python scripts/prepare_data.py",
        f"--model {args.model}",
        f"--data {args.output}",
        f"--output {args.prepared_output_dir}",
        f"--seq-length {args.seq_length}",
        f"--minimum-valid-tokens {args.minimum_valid_tokens}",
    ]
    if args.max_samples is not None:
        parts.append(f"--max-samples {args.max_samples}")
    return " \\\n  ".join(parts)


def write_reports(report: dict[str, Any], args: argparse.Namespace) -> None:
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# vLLM Speculators Data Conversion",
            "",
            f"Overall: **{report['overall'].upper()}**",
            f"Input: `{report['input']}`",
            f"Output: `{report['output']}`",
            f"Sample offset: **{report['sample_offset']}**",
            f"Max samples: **{report['max_samples']}**",
            f"Rows seen: **{report['rows_seen']}**",
            f"Rows skipped: **{report['rows_skipped']}**",
            f"Rows read: **{report['rows_read']}**",
            f"Rows written: **{report['rows_written']}**",
            f"Rows failed: **{report['rows_failed']}**",
            "",
            "## Source Schemas",
            "",
            "| schema | rows |",
            "| --- | ---: |",
        ]
        for key, count in sorted(report["source_schemas"].items()):
            lines.append(f"| `{key}` | {count} |")
        lines.extend(
            [
                "",
                "## prepare_data.py Command",
                "",
                "```bash",
                report["prepare_data_command"],
                "```",
            ]
        )
        if report["warnings"]:
            lines.extend(["", "## Warnings", "", "| warning | count |", "| --- | ---: |"])
            for key, count in sorted(report["warnings"].items()):
                lines.append(f"| `{key}` | {count} |")
        args.markdown_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.max_samples is not None and args.max_samples < 0:
        raise ValueError("--max-samples must be non-negative")
    if args.sample_offset < 0:
        raise ValueError("--sample-offset must be non-negative")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows_seen = 0
    rows_skipped = 0
    rows_read = 0
    rows_written = 0
    rows_failed = 0
    warnings: Counter[str] = Counter()
    source_schemas: Counter[str] = Counter()

    with args.output.open("w", encoding="utf-8") as out_fh:
        for line_num, record in iter_jsonl(args.input):
            rows_seen += 1
            if rows_skipped < args.sample_offset:
                rows_skipped += 1
                continue
            if args.limit is not None and rows_read >= args.limit:
                break
            if args.max_samples is not None and rows_written >= args.max_samples:
                break
            rows_read += 1
            if not isinstance(record, dict):
                rows_failed += 1
                warnings["row is not an object"] += 1
                continue

            raw_messages, schema = record_messages(record)
            source_schemas[schema or "missing"] += 1
            if raw_messages is None:
                rows_failed += 1
                warnings["missing messages/conversations"] += 1
                continue

            messages, row_warnings = normalize_messages(raw_messages, args.drop_unsupported_roles)
            for warning in row_warnings:
                warnings[warning] += 1
            if not messages:
                rows_failed += 1
                continue
            if messages[-1]["role"] != "assistant":
                rows_failed += 1
                warnings["last message is not assistant"] += 1
                continue
            if assistant_chars(messages) < args.min_assistant_chars:
                rows_failed += 1
                warnings["assistant content below minimum"] += 1
                continue

            output: dict[str, Any] = {
                "id": record_id(record, line_num),
                "conversations": messages,
            }
            if args.include_metadata:
                metadata = {
                    key: value
                    for key, value in record.items()
                    if key not in {"id", "conversation_id", "messages", "conversations"}
                }
                if metadata:
                    output["metadata"] = metadata
            out_fh.write(json.dumps(output, ensure_ascii=False) + "\n")
            rows_written += 1

    overall = "pass" if rows_written > 0 and rows_failed == 0 else "fail"
    if args.max_samples is not None and rows_written < args.max_samples:
        overall = "fail"
        warnings["fewer rows written than max-samples"] += 1
    report = {
        "overall": overall,
        "input": str(args.input),
        "output": str(args.output),
        "sample_offset": args.sample_offset,
        "max_samples": args.max_samples,
        "rows_seen": rows_seen,
        "rows_skipped": rows_skipped,
        "rows_read": rows_read,
        "rows_written": rows_written,
        "rows_failed": rows_failed,
        "source_schemas": dict(source_schemas),
        "warnings": dict(warnings),
        "prepare_data_command": prepare_data_command(args),
    }
    write_reports(report, args)

    print(json.dumps(report, indent=2, sort_keys=True))
    if overall != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
