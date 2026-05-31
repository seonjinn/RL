#!/usr/bin/env python3
"""Materialize a non-OpenMath mixed math prompt pool for Eagle3 training.

The output is prompt-only JSONL intended for target-model synthesis with
``generate_training_conversations_openai.py`` before Speculators training.
Rows have the schema:

    {
      "id": "...",
      "prompt": [{"role": "user", "content": "..."}],
      "source": "...",
      "source_dataset": "...",
      "source_index": 123
    }

The default mix avoids OpenMath so NeMo-RL/OpenMath evaluation is not also the
training source. It uses public math/reasoning prompt pools commonly seen in
reasoning and speculator workflows, then de-duplicates normalized prompts and
optionally excludes prompts from existing JSONL denylist files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PARQUET_ENDPOINT = "https://datasets-server.huggingface.co/parquet"


@dataclass(frozen=True)
class SourceSpec:
    name: str
    dataset: str
    config: str
    split: str
    quota: int
    extractor: str


DEFAULT_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        name="openthoughts2_1m",
        dataset="open-thoughts/OpenThoughts2-1M",
        config="default",
        split="train",
        quota=160_000,
        extractor="openthoughts2",
    ),
    SourceSpec(
        name="numinamath_1p5",
        dataset="AI-MO/NuminaMath-1.5",
        config="default",
        split="train",
        quota=160_000,
        extractor="problem_solution",
    ),
    SourceSpec(
        name="numinamath_cot",
        dataset="AI-MO/NuminaMath-CoT",
        config="default",
        split="train",
        quota=90_000,
        extractor="numinamath_cot",
    ),
    SourceSpec(
        name="openthoughts_114k_math",
        dataset="open-thoughts/OpenThoughts-114k",
        config="metadata",
        split="train",
        quota=60_000,
        extractor="openthoughts114_metadata",
    ),
    SourceSpec(
        name="math_deepscaler",
        dataset="PALM-Lab/math-deepscaler",
        config="default",
        split="train",
        quota=20_000,
        extractor="math_deepscaler",
    ),
    SourceSpec(
        name="dapo_math_17k",
        dataset="BytedTsinghua-SIA/DAPO-Math-17k",
        config="default",
        split="train",
        quota=10_000,
        extractor="dapo_math",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--max-total", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--fetch-mode",
        choices=("api", "parquet"),
        default="api",
        help=(
            "api uses HuggingFace datasets-server rows and avoids downloading "
            "full parquet files; parquet downloads converted parquet files."
        ),
    )
    parser.add_argument(
        "--api-page-size",
        type=int,
        default=100,
        help="HuggingFace rows endpoint page size. The service currently caps this at 100.",
    )
    parser.add_argument("--api-page-sleep-sec", type=float, default=0.0)
    parser.add_argument(
        "--source-quota",
        action="append",
        default=[],
        metavar="NAME=N",
        help="Override a default source quota. Use NAME=0 to disable a source.",
    )
    parser.add_argument(
        "--only-source",
        action="append",
        default=[],
        help="Restrict materialization to one or more default source names.",
    )
    parser.add_argument(
        "--denylist-prompts-from",
        type=Path,
        action="append",
        default=[],
        help="JSONL prompts/conversations whose normalized user prompt must be excluded.",
    )
    parser.add_argument("--min-prompt-chars", type=int, default=8)
    parser.add_argument("--max-prompt-chars", type=int, default=32_000)
    parser.add_argument("--download-timeout", type=float, default=1200.0)
    parser.add_argument("--inspect-only", action="store_true")
    return parser.parse_args()


def normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def prompt_hash(text: str) -> str:
    return hashlib.sha256(normalize_prompt(text).encode("utf-8")).hexdigest()


def text_or_none(value: Any) -> str | None:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return None


def normalize_role(role: Any) -> str:
    role = str(role or "").lower()
    return {"human": "user", "gpt": "assistant", "bot": "assistant"}.get(role, role)


def first_user_from_messages(messages: Any) -> str | None:
    if isinstance(messages, str):
        return text_or_none(messages)
    if not isinstance(messages, list):
        return None
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = normalize_role(message.get("role", message.get("from")))
        content = message.get("content", message.get("value"))
        if role == "user":
            return text_or_none(content)
    return None


def prompt_from_record(record: dict[str, Any], extractor: str) -> tuple[str | None, dict[str, Any]]:
    metadata: dict[str, Any] = {}

    if extractor == "openthoughts2":
        prompt = text_or_none(record.get("question")) or first_user_from_messages(record.get("conversations"))
        metadata["original_source"] = record.get("source")
        return prompt, metadata

    if extractor == "problem_solution":
        if str(record.get("problem_is_valid", "Yes")).lower() not in {"yes", "true", "1", ""}:
            return None, {"filtered": "problem_is_valid"}
        prompt = text_or_none(record.get("problem"))
        metadata["problem_type"] = record.get("problem_type")
        metadata["question_type"] = record.get("question_type")
        metadata["original_source"] = record.get("source")
        metadata["synthetic"] = record.get("synthetic")
        return prompt, metadata

    if extractor == "numinamath_cot":
        prompt = text_or_none(record.get("problem")) or first_user_from_messages(record.get("messages"))
        metadata["original_source"] = record.get("source")
        return prompt, metadata

    if extractor == "openthoughts114_metadata":
        if str(record.get("domain", "")).lower() != "math":
            return None, {"filtered": "domain_not_math"}
        prompt = text_or_none(record.get("problem"))
        metadata["domain"] = record.get("domain")
        metadata["original_source"] = record.get("source")
        return prompt, metadata

    if extractor == "math_deepscaler":
        prompt = text_or_none(record.get("question")) or first_user_from_messages(record.get("prompt"))
        metadata["data_source"] = record.get("data_source")
        return prompt, metadata

    if extractor == "dapo_math":
        prompt = first_user_from_messages(record.get("prompt"))
        metadata["data_source"] = record.get("data_source")
        metadata["ability"] = record.get("ability")
        return prompt, metadata

    for key in ("prompt", "question", "problem", "instruction", "input", "query"):
        value = record.get(key)
        prompt = first_user_from_messages(value) if isinstance(value, list) else text_or_none(value)
        if prompt:
            return prompt, metadata
    return None, metadata


def record_prompt_for_denylist(record: dict[str, Any]) -> str | None:
    for key in ("prompt", "messages", "conversations"):
        if key in record:
            prompt = first_user_from_messages(record[key])
            if prompt:
                return prompt
    for key in ("question", "problem", "instruction", "input", "query"):
        prompt = text_or_none(record.get(key))
        if prompt:
            return prompt
    return None


def load_denylist(paths: Iterable[Path]) -> set[str]:
    denylist: set[str] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"denylist file does not exist: {path}")
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line_num, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_num}: invalid JSON: {exc}") from exc
                if isinstance(record, dict):
                    prompt = record_prompt_for_denylist(record)
                    if prompt:
                        denylist.add(prompt_hash(prompt))
    return denylist


def apply_quota_overrides(sources: tuple[SourceSpec, ...], overrides: list[str]) -> list[SourceSpec]:
    quotas = {source.name: source.quota for source in sources}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--source-quota must be NAME=N, got: {item}")
        name, raw_value = item.split("=", 1)
        if name not in quotas:
            raise ValueError(f"unknown source quota name {name!r}; choices: {sorted(quotas)}")
        quotas[name] = int(raw_value)
    return [
        SourceSpec(
            name=source.name,
            dataset=source.dataset,
            config=source.config,
            split=source.split,
            quota=quotas[source.name],
            extractor=source.extractor,
        )
        for source in sources
        if quotas[source.name] > 0
    ]


def parquet_url_payload(dataset: str) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode({"dataset": dataset})
    url = f"{PARQUET_ENDPOINT}?{params}"
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    files = payload.get("parquet_files", [])
    if not isinstance(files, list):
        raise RuntimeError(f"unexpected parquet endpoint response for {dataset}: {payload}")
    return [item for item in files if isinstance(item, dict)]


def source_parquet_files(source: SourceSpec) -> list[dict[str, Any]]:
    files = []
    for item in parquet_url_payload(source.dataset):
        if item.get("config") == source.config and item.get("split") == source.split:
            files.append(item)
    if not files:
        raise RuntimeError(
            f"no parquet files for {source.dataset} config={source.config} split={source.split}"
        )
    return files


def fetch_api_rows(
    source: SourceSpec,
    offset: int,
    length: int,
    retries: int = 8,
) -> tuple[list[dict[str, Any]], int | None]:
    params = urllib.parse.urlencode(
        {
            "dataset": source.dataset,
            "config": source.config,
            "split": source.split,
            "offset": offset,
            "length": length,
        }
    )
    url = f"https://datasets-server.huggingface.co/rows?{params}"
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                payload = json.loads(response.read().decode("utf-8"))
            rows = [
                item["row"]
                for item in payload.get("rows", [])
                if isinstance(item, dict) and isinstance(item.get("row"), dict)
            ]
            total = payload.get("num_rows_total")
            return rows, int(total) if isinstance(total, int) else None
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            if attempt < retries:
                time.sleep(min(60.0, 2.0 * attempt))
    raise RuntimeError(f"failed to fetch rows for {source.name} offset={offset}") from last_error


def iter_api_rows(source: SourceSpec, page_size: int, page_sleep_sec: float):
    if page_size <= 0 or page_size > 100:
        raise ValueError("--api-page-size must be between 1 and 100")
    offset = 0
    total: int | None = None
    while total is None or offset < total:
        rows, total = fetch_api_rows(source, offset, page_size)
        if not rows:
            break
        for page_index, row in enumerate(rows):
            yield offset + page_index, row
        offset += len(rows)
        if page_sleep_sec > 0:
            time.sleep(page_sleep_sec)


def download_file(url: str, output: Path, timeout: float) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0:
        return output
    tmp = output.with_suffix(output.suffix + ".tmp")
    curl = shutil.which("curl")
    if curl:
        cmd = [
            curl,
            "-L",
            "--fail",
            "--show-error",
            "--connect-timeout",
            "60",
            "--max-time",
            str(int(timeout)),
            "--retry",
            "5",
            "--retry-delay",
            "5",
            "--output",
            str(tmp),
            url,
        ]
        subprocess.run(cmd, check=True)
        tmp.replace(output)
        return output
    with urllib.request.urlopen(url, timeout=timeout) as response, tmp.open("wb") as fh:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    tmp.replace(output)
    return output


def iter_parquet_rows(path: Path, batch_size: int):
    import pyarrow.parquet as pq  # type: ignore

    parquet = pq.ParquetFile(path)
    source_index = 0
    for batch in parquet.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            yield source_index, row
            source_index += 1


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_total <= 0:
        raise ValueError("--max-total must be positive")

    sources = apply_quota_overrides(DEFAULT_SOURCES, args.source_quota)
    if args.only_source:
        wanted = set(args.only_source)
        known = {source.name for source in sources}
        missing = wanted - known
        if missing:
            raise ValueError(f"unknown --only-source entries: {sorted(missing)}")
        sources = [source for source in sources if source.name in wanted]

    output = args.output
    manifest = args.manifest or output.with_suffix(output.suffix + ".manifest.json")
    cache_dir = args.cache_dir or (output.parent / ".mixed_math_parquet_cache")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    denylist = load_denylist(args.denylist_prompts_from)
    seen_hashes = set(denylist)
    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "output": str(output),
        "manifest": str(manifest),
        "cache_dir": str(cache_dir),
        "max_total": args.max_total,
        "fetch_mode": args.fetch_mode,
        "denylist_prompts": len(denylist),
        "sources": {},
        "rows_written": 0,
        "unique_prompt_hashes": 0,
    }

    out_fh = None if args.inspect_only else output.open("w", encoding="utf-8")
    try:
        for source in sources:
            stats = {
                "dataset": source.dataset,
                "config": source.config,
                "split": source.split,
                "quota": source.quota,
                "extractor": source.extractor,
                "fetch_mode": args.fetch_mode,
                "parquet_files_seen": 0,
                "api_rows_endpoint_pages": 0,
                "rows_scanned": 0,
                "rows_written": 0,
                "rows_no_prompt": 0,
                "rows_filtered": 0,
                "rows_too_short": 0,
                "rows_too_long": 0,
                "rows_duplicate_or_denied": 0,
                "downloaded_bytes": 0,
            }
            report["sources"][source.name] = stats
            if args.fetch_mode == "api":
                row_iter = iter_api_rows(source, args.api_page_size, args.api_page_sleep_sec)
                for row_index, row in row_iter:
                    if stats["rows_scanned"] % args.api_page_size == 0:
                        stats["api_rows_endpoint_pages"] += 1
                    if stats["rows_written"] >= source.quota:
                        break
                    if report["rows_written"] >= args.max_total:
                        break
                    stats["rows_scanned"] += 1
                    if not isinstance(row, dict):
                        stats["rows_no_prompt"] += 1
                        continue
                    prompt, metadata = prompt_from_record(row, source.extractor)
                    if metadata.get("filtered"):
                        stats["rows_filtered"] += 1
                        continue
                    if not prompt:
                        stats["rows_no_prompt"] += 1
                        continue
                    normalized = normalize_prompt(prompt)
                    if len(normalized) < args.min_prompt_chars:
                        stats["rows_too_short"] += 1
                        continue
                    if len(normalized) > args.max_prompt_chars:
                        stats["rows_too_long"] += 1
                        continue
                    key = prompt_hash(normalized)
                    if key in seen_hashes:
                        stats["rows_duplicate_or_denied"] += 1
                        continue
                    seen_hashes.add(key)
                    row_id = f"{source.name}-{stats['rows_written']:06d}"
                    output_row = {
                        "id": row_id,
                        "prompt": [{"role": "user", "content": normalized}],
                        "source": source.name,
                        "source_dataset": source.dataset,
                        "source_config": source.config,
                        "source_split": source.split,
                        "source_index": row_index,
                        "prompt_sha256": key,
                    }
                    for meta_key, meta_value in metadata.items():
                        if meta_value not in (None, ""):
                            output_row[meta_key] = meta_value
                    if out_fh is not None:
                        out_fh.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                    stats["rows_written"] += 1
                    report["rows_written"] += 1
                if report["rows_written"] >= args.max_total:
                    break
                continue

            for file_info in source_parquet_files(source):
                if stats["rows_written"] >= source.quota:
                    break
                if report["rows_written"] >= args.max_total:
                    break
                stats["parquet_files_seen"] += 1
                url = str(file_info["url"])
                filename = file_info.get("filename") or Path(urllib.parse.urlparse(url).path).name
                local_path = cache_dir / source.name / str(filename)
                local_path = download_file(url, local_path, args.download_timeout)
                stats["downloaded_bytes"] += local_path.stat().st_size

                for row_index, row in iter_parquet_rows(local_path, args.batch_size):
                    if stats["rows_written"] >= source.quota:
                        break
                    if report["rows_written"] >= args.max_total:
                        break
                    stats["rows_scanned"] += 1
                    if not isinstance(row, dict):
                        stats["rows_no_prompt"] += 1
                        continue
                    prompt, metadata = prompt_from_record(row, source.extractor)
                    if metadata.get("filtered"):
                        stats["rows_filtered"] += 1
                        continue
                    if not prompt:
                        stats["rows_no_prompt"] += 1
                        continue
                    normalized = normalize_prompt(prompt)
                    if len(normalized) < args.min_prompt_chars:
                        stats["rows_too_short"] += 1
                        continue
                    if len(normalized) > args.max_prompt_chars:
                        stats["rows_too_long"] += 1
                        continue
                    key = prompt_hash(normalized)
                    if key in seen_hashes:
                        stats["rows_duplicate_or_denied"] += 1
                        continue
                    seen_hashes.add(key)
                    row_id = f"{source.name}-{stats['rows_written']:06d}"
                    output_row = {
                        "id": row_id,
                        "prompt": [{"role": "user", "content": normalized}],
                        "source": source.name,
                        "source_dataset": source.dataset,
                        "source_config": source.config,
                        "source_split": source.split,
                        "source_index": row_index,
                        "prompt_sha256": key,
                    }
                    for meta_key, meta_value in metadata.items():
                        if meta_value not in (None, ""):
                            output_row[meta_key] = meta_value
                    if out_fh is not None:
                        out_fh.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                    stats["rows_written"] += 1
                    report["rows_written"] += 1
            if report["rows_written"] >= args.max_total:
                break
    finally:
        if out_fh is not None:
            out_fh.close()

    report["unique_prompt_hashes"] = len(seen_hashes) - len(denylist)
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    try:
        report = materialize(parse_args())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
