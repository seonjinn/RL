#!/usr/bin/env python3
"""Materialize SWE-Gym rows into SWE-agent NemoGym JSONL.

The Qwen3 SWE rollout path consumes rows shaped like:

    {"responses_create_params": {"input": [], "metadata": {...}}}

The visible tk-nemo-gym example has most SWE metadata but lacks
`metadata.instance_dict`. The OpenHands runner later calls
`json.loads(data_point["instance_dict"])`, so this materializer writes
`instance_dict` as a JSON string and validates the first-order contract before
the expensive Qwen3 rollout capture job is submitted.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator


DEFAULT_DATASET_NAME = "SWE-Gym/SWE-Gym"
DEFAULT_SPLIT = "train"
DEFAULT_MODEL = os.environ.get("MODEL_PATH", "Qwen/Qwen3-235B-A22B-Thinking-2507")
REQUIRED_METADATA_KEYS = (
    "problem_statement",
    "instance_id",
    "base_commit",
    "dataset_name",
    "split",
    "instance_dict",
)
PASSTHROUGH_METADATA_KEYS = (
    "golden_patch",
    "patch",
    "hints_text",
    "test_patch",
    "repo",
    "repo_name",
    "version",
    "created_at",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "fail_to_pass",
    "pass_to_pass",
    "fail_to_pass_select",
    "pass_to_pass_select",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--source-jsonl",
        type=Path,
        help="Existing JSONL to repair/normalize. Supports already wrapped NemoGym rows.",
    )
    source.add_argument(
        "--dataset-name",
        default=None,
        help="Hugging Face dataset id to load with datasets.load_dataset, for example SWE-Gym/SWE-Gym.",
    )
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--streaming", action="store_true", help="Use HF streaming mode for large datasets.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--preserve-source-generation-params",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --source-jsonl is already wrapped, keep source model/temperature/top_p unless overridden.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "as_py"):
        return json_safe(value.as_py())
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    return str(value)


def nonempty(value: Any) -> bool:
    return value is not None and value != ""


def coalesce(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if nonempty(value):
            return value
    return None


def iter_source_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line_num, line in enumerate(fh, 1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_num}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_num}")
            yield value


def iter_hf_dataset(dataset_name: str, split: str, streaming: bool) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError(
            "datasets is required for --dataset-name. On the cluster use: "
            "uv run --with datasets --with pyarrow --with huggingface_hub python3 ..."
        ) from exc

    dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    for row in dataset:
        if not isinstance(row, dict):
            raise ValueError(f"Expected dict row from {dataset_name}/{split}, got {type(row).__name__}")
        yield row


def unwrap_record(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return source instance, existing metadata, and existing generation params."""
    params = record.get("responses_create_params")
    if isinstance(params, dict):
        metadata = params.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        instance_raw = metadata.get("instance_dict")
        instance: dict[str, Any] | None = None
        if isinstance(instance_raw, str) and instance_raw.strip():
            try:
                parsed = json.loads(instance_raw)
                if isinstance(parsed, dict):
                    instance = parsed
            except json.JSONDecodeError:
                instance = None
        if instance is None:
            instance = {key: value for key, value in metadata.items() if key != "instance_dict"}
        return json_safe(instance), json_safe(metadata), params
    return json_safe(record), {}, {}


def build_instance_dict(instance: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    result = dict(instance)
    for key in ("instance_id", "repo", "base_commit", "problem_statement"):
        value = coalesce(result, key) or metadata.get(key)
        if nonempty(value):
            result[key] = value
    repo = coalesce(result, "repo", "repo_name") or metadata.get("repo")
    if nonempty(repo):
        result["repo"] = repo
        result.setdefault("repo_name", repo)
    patch = coalesce(result, "patch", "golden_patch") or coalesce(metadata, "patch", "golden_patch")
    if nonempty(patch):
        result["patch"] = patch
        result.setdefault("golden_patch", patch)
    return json_safe(result)


def build_metadata(
    instance: dict[str, Any],
    existing_metadata: dict[str, Any],
    dataset_name: str,
    split: str,
) -> dict[str, Any]:
    metadata = dict(existing_metadata)
    metadata["instance_id"] = coalesce(metadata, "instance_id") or coalesce(instance, "instance_id")
    metadata["base_commit"] = coalesce(metadata, "base_commit") or coalesce(instance, "base_commit")
    metadata["dataset_name"] = coalesce(metadata, "dataset_name") or dataset_name
    metadata["split"] = coalesce(metadata, "split") or split
    metadata["problem_statement"] = coalesce(metadata, "problem_statement") or coalesce(instance, "problem_statement")

    repo = coalesce(metadata, "repo", "repo_name") or coalesce(instance, "repo", "repo_name")
    if nonempty(repo):
        metadata["repo"] = repo

    patch = coalesce(metadata, "patch", "golden_patch") or coalesce(instance, "patch", "golden_patch")
    if nonempty(patch):
        metadata["patch"] = patch
        metadata["golden_patch"] = patch

    for key in PASSTHROUGH_METADATA_KEYS:
        if key in metadata:
            continue
        value = instance.get(key)
        if nonempty(value):
            metadata[key] = value

    instance_dict = build_instance_dict(instance, metadata)
    metadata["instance_dict"] = json.dumps(instance_dict, ensure_ascii=False, sort_keys=True)
    return json_safe(metadata)


def build_output_record(
    record: dict[str, Any],
    dataset_name: str,
    split: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    instance, existing_metadata, existing_params = unwrap_record(record)
    metadata = build_metadata(instance, existing_metadata, dataset_name, split)
    params: dict[str, Any] = {
        "input": [],
        "metadata": metadata,
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if args.preserve_source_generation_params and existing_params:
        for key in ("model", "temperature", "top_p"):
            value = existing_params.get(key)
            if nonempty(value):
                params[key] = value
    return {"responses_create_params": params}


def validate_output_record(record: dict[str, Any]) -> tuple[list[str], list[str], dict[str, Any]]:
    failures: list[str] = []
    warnings: list[str] = []
    params = record.get("responses_create_params")
    if not isinstance(params, dict):
        return ["missing responses_create_params"], warnings, {}
    metadata = params.get("metadata")
    if not isinstance(metadata, dict):
        return ["missing responses_create_params.metadata"], warnings, {}
    for key in REQUIRED_METADATA_KEYS:
        if not nonempty(metadata.get(key)):
            failures.append(f"missing metadata.{key}")
    instance_keys: list[str] = []
    instance_raw = metadata.get("instance_dict")
    if isinstance(instance_raw, str) and instance_raw.strip():
        try:
            instance = json.loads(instance_raw)
            if not isinstance(instance, dict):
                failures.append("metadata.instance_dict is not a JSON object string")
            else:
                instance_keys = sorted(instance.keys())
                for key in ("instance_id", "repo", "base_commit"):
                    if not nonempty(instance.get(key)):
                        warnings.append(f"instance_dict missing {key}")
                dataset_name = str(metadata.get("dataset_name") or "")
                if dataset_name == "nv-internal-1":
                    for key in ("run_script.sh", "parsing_script.py"):
                        if key not in instance:
                            warnings.append(f"instance_dict missing {key}; nv-internal eval requires it")
                if dataset_name and "R2E-Gym" not in dataset_name and dataset_name != "nv-internal-1":
                    if not nonempty(instance.get("patch")):
                        warnings.append("instance_dict missing patch alias for SWE-bench-style eval")
        except json.JSONDecodeError as exc:
            failures.append(f"metadata.instance_dict is not valid JSON: {exc}")
    elif "instance_dict" in metadata:
        failures.append("metadata.instance_dict must be a nonempty JSON string")
    details = {
        "instance_id": metadata.get("instance_id"),
        "dataset_name": metadata.get("dataset_name"),
        "split": metadata.get("split"),
        "model": params.get("model"),
        "instance_dict_keys": instance_keys,
    }
    return failures, warnings, details


def limited(iterable: Iterable[dict[str, Any]], limit: int | None) -> Iterator[dict[str, Any]]:
    for idx, item in enumerate(iterable):
        if limit is not None and idx >= limit:
            return
        yield item


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# SWE-Gym NemoGym Materialization",
        "",
        f"Generated: `{payload['generated_at']}`",
        f"Overall: **{payload['overall_status'].upper()}**",
        f"Source: `{payload['source']}`",
        f"Output: `{payload['output_jsonl']}`",
        f"Rows written: **{payload['rows_written']}**",
        f"Rows failed validation: **{payload['rows_failed']}**",
        "",
        "## Sample Rows",
        "",
        "| row | instance | dataset | split | model | warnings |",
        "| ---: | --- | --- | --- | --- | ---: |",
    ]
    for idx, sample in enumerate(payload["samples"], 1):
        lines.append(
            f"| {idx} | `{sample.get('instance_id')}` | `{sample.get('dataset_name')}` | "
            f"`{sample.get('split')}` | `{sample.get('model')}` | {sample.get('warning_count', 0)} |"
        )
    if not payload["samples"]:
        lines.append("| - | - | - | - | - | - |")
    lines.extend(
        [
            "",
            "## Warnings",
            "",
            "| warning | count |",
            "| --- | ---: |",
        ]
    )
    for warning, count in payload["warning_counts"].items():
        lines.append(f"| {warning} | {count} |")
    if not payload["warning_counts"]:
        lines.append("| none | 0 |")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if args.source_jsonl:
        source_label = str(args.source_jsonl)
        rows = iter_source_jsonl(args.source_jsonl)
        dataset_name = DEFAULT_DATASET_NAME
    else:
        dataset_name = args.dataset_name or DEFAULT_DATASET_NAME
        source_label = f"{dataset_name}:{args.split}"
        rows = iter_hf_dataset(dataset_name, args.split, args.streaming)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows_seen = 0
    rows_written = 0
    rows_failed = 0
    failure_examples: list[dict[str, Any]] = []
    warning_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []

    with args.output_jsonl.open("w", encoding="utf-8") as out:
        for source_record in limited(rows, args.limit):
            rows_seen += 1
            output_record = build_output_record(source_record, dataset_name, args.split, args)
            failures, warnings, details = validate_output_record(output_record)
            if failures:
                rows_failed += 1
                if len(failure_examples) < 10:
                    failure_examples.append({"row": rows_seen, "failures": failures, "details": details})
                continue
            warning_counts.update(warnings)
            if len(samples) < 8:
                sample = dict(details)
                sample["warning_count"] = len(warnings)
                samples.append(sample)
            out.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            rows_written += 1

    overall_status = "pass" if rows_written and not rows_failed else ("warn" if rows_written else "fail")
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": overall_status,
        "source": source_label,
        "output_jsonl": str(args.output_jsonl),
        "rows_seen": rows_seen,
        "rows_written": rows_written,
        "rows_failed": rows_failed,
        "limit": args.limit,
        "required_metadata_keys": list(REQUIRED_METADATA_KEYS),
        "warning_counts": dict(warning_counts),
        "failure_examples": failure_examples,
        "samples": samples,
    }
    markdown = render_markdown(payload)
    print(markdown, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")
    return 0 if overall_status in {"pass", "warn"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
