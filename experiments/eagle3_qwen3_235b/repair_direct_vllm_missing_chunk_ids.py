#!/usr/bin/env python3
"""Repair direct-vLLM target chunks by generating exact missing ids.

Count-based resume only works while a chunk file is a contiguous leading prefix
of the prompt slice. Older concurrent writers flushed completions out of order,
so some short chunks need an id-based repair instead: compare the expected ids
from the source prompt slice against ids already present in each chunk, generate
only those missing prompts, then append the generated records to their chunk.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def parse_chunks(value: str) -> list[int]:
    chunks: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            chunks.extend(range(int(start), int(end) + 1))
        else:
            chunks.append(int(item))
    return sorted(dict.fromkeys(chunks))


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_num, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_num}: {exc}") from exc


def record_id(record: Any) -> str:
    if not isinstance(record, dict):
        return ""
    for key in ("conversation_id", "id"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def source_id(record: dict[str, Any], line_num: int, id_key: str | None) -> str:
    keys = [id_key] if id_key else []
    keys.extend(["conversation_id", "uuid", "id", "task_id", "instance_id"])
    for key in keys:
        if key and key in record and record[key] not in (None, ""):
            return str(record[key])
    return f"row-{line_num:08d}"


def load_existing_ids(path: Path) -> tuple[set[str], int]:
    if not path.exists():
        return set(), 0
    seen: set[str] = set()
    rows = 0
    for _, record in iter_jsonl(path):
        rows += 1
        cid = record_id(record)
        if not cid:
            raise ValueError(f"{path} contains a record without conversation_id/id")
        if cid in seen:
            raise ValueError(f"{path} contains duplicate id {cid!r}")
        seen.add(cid)
    return seen, rows


def chunk_path(chunk_dir: Path, model_label: str, chunk: int) -> Path:
    return chunk_dir / f"{model_label}_{chunk:03d}.jsonl"


def prepare(args: argparse.Namespace) -> int:
    chunks = parse_chunks(args.chunks)
    chunk_set = set(chunks)
    existing: dict[int, set[str]] = {}
    before_rows: dict[str, int] = {}
    missing_counts: dict[str, int] = {}
    missing_map: dict[str, int] = {}

    for chunk in chunks:
        ids, rows = load_existing_ids(chunk_path(args.chunk_dir, args.model_label, chunk))
        existing[chunk] = ids
        before_rows[f"{chunk:03d}"] = rows

    args.missing_prompts.parent.mkdir(parents=True, exist_ok=True)
    missing_rows = 0
    with args.missing_prompts.open("w", encoding="utf-8") as out:
        for line_num, record in iter_jsonl(args.prompt_data):
            chunk = (line_num - 1) // args.chunk_size
            if chunk not in chunk_set:
                continue
            in_chunk_index = (line_num - 1) % args.chunk_size
            if in_chunk_index >= args.chunk_size:
                continue
            sid = source_id(record, line_num, args.id_key)
            cid = f"{sid}-r00"
            if cid in existing[chunk]:
                continue
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            missing_map[cid] = chunk
            missing_rows += 1

    for chunk in chunks:
        missing_counts[f"{chunk:03d}"] = sum(1 for value in missing_map.values() if value == chunk)

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "mode": "prepare_missing_id_repair",
        "prompt_data": str(args.prompt_data),
        "chunk_dir": str(args.chunk_dir),
        "model_label": args.model_label,
        "chunk_size": args.chunk_size,
        "chunks": [f"{chunk:03d}" for chunk in chunks],
        "rows_before": before_rows,
        "missing_counts": missing_counts,
        "missing_total": missing_rows,
        "missing_prompts": str(args.missing_prompts),
        "missing_map": {cid: f"{chunk:03d}" for cid, chunk in sorted(missing_map.items())},
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def apply(args: argparse.Namespace) -> int:
    payload = json.loads(args.prepare_json.read_text(encoding="utf-8"))
    missing_map = {cid: int(chunk) for cid, chunk in payload.get("missing_map", {}).items()}
    chunks = parse_chunks(args.chunks)
    existing: dict[int, set[str]] = {}
    rows_before: dict[str, int] = {}
    for chunk in chunks:
        ids, rows = load_existing_ids(chunk_path(args.chunk_dir, args.model_label, chunk))
        existing[chunk] = ids
        rows_before[f"{chunk:03d}"] = rows

    append_records: dict[int, list[dict[str, Any]]] = {chunk: [] for chunk in chunks}
    generated_ids: set[str] = set()
    unexpected_ids: list[str] = []
    duplicate_generated_ids: list[str] = []
    already_present_ids: list[str] = []
    for _, record in iter_jsonl(args.generated_output):
        cid = record_id(record)
        if not cid:
            raise ValueError(f"{args.generated_output} contains a record without conversation_id/id")
        if cid in generated_ids:
            duplicate_generated_ids.append(cid)
            continue
        generated_ids.add(cid)
        chunk = missing_map.get(cid)
        if chunk is None:
            unexpected_ids.append(cid)
            continue
        if cid in existing[chunk]:
            already_present_ids.append(cid)
            continue
        append_records[chunk].append(record)
        existing[chunk].add(cid)

    for chunk, records in append_records.items():
        if not records:
            continue
        path = chunk_path(args.chunk_dir, args.model_label, chunk)
        with path.open("a", encoding="utf-8") as out:
            for record in records:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

    rows_after: dict[str, int] = {}
    still_missing: dict[str, list[str]] = {}
    for chunk in chunks:
        ids, rows = load_existing_ids(chunk_path(args.chunk_dir, args.model_label, chunk))
        rows_after[f"{chunk:03d}"] = rows
        chunk_missing = [
            cid for cid, mapped_chunk in missing_map.items() if mapped_chunk == chunk and cid not in ids
        ]
        if chunk_missing:
            still_missing[f"{chunk:03d}"] = chunk_missing

    appended_counts = {f"{chunk:03d}": len(records) for chunk, records in append_records.items()}
    status = "pass"
    reasons: list[str] = []
    if still_missing:
        status = "incomplete"
        reasons.append("generated output did not cover all missing ids")
    short_chunks = {
        f"{chunk:03d}": rows
        for chunk, rows in ((chunk, rows_after[f"{chunk:03d}"]) for chunk in chunks)
        if rows < args.chunk_size
    }
    overfull_chunks = {
        f"{chunk:03d}": rows
        for chunk, rows in ((chunk, rows_after[f"{chunk:03d}"]) for chunk in chunks)
        if rows > args.chunk_size
    }
    if short_chunks:
        status = "incomplete"
        reasons.append("one or more chunks remain short")
    if overfull_chunks:
        status = "fail"
        reasons.append("one or more chunks are overfull")
    if unexpected_ids or duplicate_generated_ids:
        status = "fail"
        reasons.append("generated output contained unexpected or duplicate ids")

    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "mode": "apply_missing_id_repair",
        "status": status,
        "reasons": reasons,
        "chunk_dir": str(args.chunk_dir),
        "model_label": args.model_label,
        "chunk_size": args.chunk_size,
        "chunks": [f"{chunk:03d}" for chunk in chunks],
        "rows_before": rows_before,
        "rows_after": rows_after,
        "appended_counts": appended_counts,
        "missing_requested": len(missing_map),
        "generated_ids": len(generated_ids),
        "already_present_ids": already_present_ids,
        "unexpected_ids": unexpected_ids,
        "duplicate_generated_ids": duplicate_generated_ids,
        "still_missing": still_missing,
        "short_chunks": short_chunks,
        "overfull_chunks": overfull_chunks,
        "generated_output": str(args.generated_output),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if status == "pass" else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--chunk-dir", type=Path, required=True)
        p.add_argument("--model-label", required=True)
        p.add_argument("--chunk-size", type=int, default=5000)
        p.add_argument("--chunks", required=True, help="Comma-separated chunks or ranges, e.g. 5,6,7,8 or 5-8")
        p.add_argument("--id-key", default=None)
        p.add_argument("--json-out", type=Path, required=True)

    prep = sub.add_parser("prepare")
    add_common(prep)
    prep.add_argument("--prompt-data", type=Path, required=True)
    prep.add_argument("--missing-prompts", type=Path, required=True)

    apply_p = sub.add_parser("apply")
    add_common(apply_p)
    apply_p.add_argument("--prepare-json", type=Path, required=True)
    apply_p.add_argument("--generated-output", type=Path, required=True)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        return prepare(args)
    if args.command == "apply":
        return apply(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
