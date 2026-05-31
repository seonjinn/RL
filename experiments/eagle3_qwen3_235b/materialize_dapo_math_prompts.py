#!/usr/bin/env python3
"""Materialize unique DAPO-Math prompts for verifier generation.

DAPO-Math-17k is published as 1,791,700 parquet rows, but the first
17,917-row block repeats 100 times. This script deduplicates by normalized
prompt text and writes prompt rows that ``generate_training_conversations``
can consume directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET = "BytedTsinghua-SIA/DAPO-Math-17k"
DEFAULT_SPLIT = "train"
DEFAULT_UNIQUE_PERIOD = 17917
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
PARQUET_ENDPOINT = "https://datasets-server.huggingface.co/parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--limit", type=int, default=DEFAULT_UNIQUE_PERIOD)
    parser.add_argument("--offset", type=int, default=0, help="Unique-prompt offset after dedup")
    parser.add_argument("--max-scan", type=int, default=DEFAULT_UNIQUE_PERIOD)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--page-sleep-sec", type=float, default=0.5)
    parser.add_argument("--source", choices=("auto", "parquet", "api"), default="auto")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for downloaded parquet files. Defaults to OUTPUT.parent/.dapo_cache.",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-235B-A22B")
    parser.add_argument(
        "--exclude-prompts-from",
        type=Path,
        action="append",
        default=[],
        help="Prompt/conversation JSONL files whose user prompts must be excluded.",
    )
    parser.add_argument("--inspect-only", action="store_true")
    return parser.parse_args()


def normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def prompt_hash(text: str) -> str:
    return hashlib.sha256(normalize_prompt(text).encode("utf-8")).hexdigest()


def content_to_text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def prompt_from_messages(messages: Any) -> str | None:
    if not isinstance(messages, list):
        return None
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", message.get("from", ""))).lower()
        role = {"human": "user"}.get(role, role)
        if role != "user":
            continue
        content = content_to_text(message.get("content", message.get("value")))
        if content:
            return content
    return None


def prompt_from_record(record: dict[str, Any]) -> str | None:
    messages = record.get("messages", record.get("conversations", record.get("prompt")))
    prompt = prompt_from_messages(messages)
    if prompt:
        return prompt
    for key in ("problem", "question", "instruction", "input", "query"):
        value = content_to_text(record.get(key))
        if value:
            return value
    return None


def load_excluded_prompt_hashes(paths: Iterable[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"--exclude-prompts-from does not exist: {path}")
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line_num, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_num}: invalid JSON: {exc}") from exc
                if not isinstance(record, dict):
                    continue
                prompt = prompt_from_record(record)
                if prompt:
                    excluded.add(prompt_hash(prompt))
    return excluded


def fetch_rows(
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
    retries: int = 12,
) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    url = f"{ROWS_ENDPOINT}?{params}"
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return [item["row"] for item in payload.get("rows", [])]
        except urllib.error.HTTPError as exc:  # pragma: no cover - network dependent
            last_error = exc
            if attempt < retries:
                retry_after = exc.headers.get("Retry-After")
                try:
                    sleep_sec = float(retry_after) if retry_after else 0.0
                except ValueError:
                    sleep_sec = 0.0
                if exc.code == 429:
                    sleep_sec = max(sleep_sec, min(120.0, 10.0 * attempt))
                else:
                    sleep_sec = max(sleep_sec, 2.0 * attempt)
                time.sleep(sleep_sec)
                continue
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"failed to fetch DAPO rows from {url}") from last_error


def fetch_parquet_urls(dataset: str) -> list[str]:
    params = urllib.parse.urlencode({"dataset": dataset})
    url = f"{PARQUET_ENDPOINT}?{params}"
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [item["url"] for item in payload.get("parquet_files", [])]


def download_file(url: str, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0:
        return output
    tmp = output.with_suffix(output.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=600) as response, tmp.open("wb") as fh:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    tmp.replace(output)
    return output


def iter_parquet_rows(args: argparse.Namespace):
    import pyarrow.parquet as pq  # type: ignore

    cache_dir = args.cache_dir or (args.output.parent / ".dapo_cache")
    yielded = 0
    for file_index, url in enumerate(fetch_parquet_urls(args.dataset)):
        filename = Path(urllib.parse.urlparse(url).path).name or f"{file_index:04d}.parquet"
        local_path = download_file(url, cache_dir / filename)
        parquet = pq.ParquetFile(local_path)
        for batch in parquet.iter_batches(batch_size=args.page_size):
            for row in batch.to_pylist():
                yield yielded, row
                yielded += 1
                if yielded >= args.max_scan:
                    return


def iter_api_rows(args: argparse.Namespace):
    scanned = 0
    source_offset = 0
    while scanned < args.max_scan:
        page_len = min(args.page_size, args.max_scan - scanned)
        page = fetch_rows(args.dataset, args.config, args.split, source_offset, page_len)
        if not page:
            break
        for page_idx, row in enumerate(page):
            source_index = source_offset + page_idx
            scanned += 1
            yield source_index, row
            if scanned >= args.max_scan:
                return
        source_offset += len(page)
        if args.page_sleep_sec > 0:
            time.sleep(args.page_sleep_sec)


def iter_source_rows(args: argparse.Namespace):
    if args.source in {"auto", "parquet"}:
        try:
            yield from iter_parquet_rows(args)
            return
        except Exception:
            if args.source == "parquet":
                raise
    yield from iter_api_rows(args)


def row_id(row: dict[str, Any], source_index: int) -> str:
    extra = row.get("extra_info")
    if isinstance(extra, dict) and extra.get("index") not in (None, ""):
        return f"dapo-{extra['index']}"
    value = row.get("id")
    if value not in (None, ""):
        return f"dapo-{value}"
    return f"dapo-{source_index:08d}"


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.offset < 0:
        raise ValueError("--offset must be >= 0")
    if args.max_scan < 1:
        raise ValueError("--max-scan must be >= 1")
    if not 1 <= args.page_size <= 100:
        raise ValueError("--page-size must be in [1, 100] for the HF rows API")

    excluded = load_excluded_prompt_hashes(args.exclude_prompts_from)
    seen_hashes: set[str] = set()
    rows: list[dict[str, Any]] = []
    unique_seen = 0
    scanned = 0
    skipped_duplicate = 0
    skipped_excluded = 0
    skipped_missing_prompt = 0

    for source_index, row in iter_source_rows(args):
        scanned += 1
        prompt = prompt_from_record(row)
        if not prompt:
            skipped_missing_prompt += 1
            continue
        key = prompt_hash(prompt)
        if key in seen_hashes:
            skipped_duplicate += 1
            continue
        seen_hashes.add(key)
        if key in excluded:
            skipped_excluded += 1
            continue
        if unique_seen < args.offset:
            unique_seen += 1
            continue
        unique_seen += 1

        reward_model = row.get("reward_model")
        if not isinstance(reward_model, dict):
            reward_model = {}
        rows.append(
            {
                "id": row_id(row, source_index),
                "prompt": [{"role": "user", "content": prompt}],
                "source_dataset": args.dataset,
                "source_config": args.config,
                "source_split": args.split,
                "source_index": source_index,
                "data_source": row.get("data_source"),
                "ability": row.get("ability"),
                "ground_truth": reward_model.get("ground_truth"),
                "reward_style": reward_model.get("style"),
                "model": args.model,
            }
        )
        if len(rows) >= args.limit:
            break

    summary = {
        "rows_written": len(rows),
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "limit": args.limit,
        "offset": args.offset,
        "max_scan": args.max_scan,
        "source": args.source,
        "scanned": scanned,
        "unique_seen": unique_seen,
        "excluded_prompt_count": len(excluded),
        "skipped_duplicate_prompt": skipped_duplicate,
        "skipped_excluded_prompt": skipped_excluded,
        "skipped_missing_prompt": skipped_missing_prompt,
        "output": str(args.output),
        "sample": rows[:2],
    }
    if args.inspect_only:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return
    if not rows:
        print(json.dumps(summary, indent=2, ensure_ascii=False), file=sys.stderr)
        raise RuntimeError("No DAPO prompts materialized")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "sample"}, indent=2))


if __name__ == "__main__":
    main()
