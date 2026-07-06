#!/usr/bin/env python3
"""Pinned SPEED-Bench dataset helpers for staging and overlay selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DATASET_ID = "nvidia/SPEED-Bench"
DATASET_REVISION = "487aa718444e816458d1a0a52bfce7a454285cf4"
DATASET_LICENSE_NAME = "nvidia-evaluation-dataset-license"
DATASET_LICENSE_FILES = ("License.pdf", "README.md")

MODELOPT_REPO = "NVIDIA/Model-Optimizer"
MODELOPT_REPO_URL = "https://github.com/NVIDIA/Model-Optimizer.git"
MODELOPT_REVISION = "43fee0cd70fa9e5f85782d52a4bd8ad9c8b88446"
MODELOPT_PREPARE_DATA_SCRIPT = "examples/specdec_bench/prepare_data.py"
MODELOPT_LICENSE_FILE = "LICENSE"

UNRESOLVED_TURNS_MARKER = (
    "FULL BENCHMARK DATA SHOULD BE FETCHED FROM THE SOURCE USING SPECDEC_BENCH"
)
EXPECTED_CONFIGS = (
    "qualitative",
    "throughput_1k",
    "throughput_2k",
    "throughput_8k",
    "throughput_16k",
    "throughput_32k",
)
THROUGHPUT_CATEGORY_ORDER = ("low_entropy", "mixed", "high_entropy")
SYNC_OVERLAY_LAYOUTS = (
    {"low_entropy": 6, "mixed": 5, "high_entropy": 5},
    {"low_entropy": 5, "mixed": 6, "high_entropy": 5},
    {"low_entropy": 5, "mixed": 5, "high_entropy": 6},
)
CONFIG_TO_NOMINAL_ISL = {
    "qualitative": None,
    "throughput_1k": 1024,
    "throughput_2k": 2048,
    "throughput_8k": 8192,
    "throughput_16k": 16384,
    "throughput_32k": 32768,
}
MASK_FIELD_NAMES = ("masked", "is_masked", "mask_reason", "mask_status")
MASK_TOKENS = ("<mask>", "[mask]")


@dataclass(frozen=True, slots=True)
class SpeedBenchRecord:
    question_id: str
    category: str
    sub_category: str
    turns: tuple[str, ...]
    source: str
    src_id: str
    difficulty: str | None
    multiturn: bool
    dataset_config: str
    nominal_isl: int | None
    actual_tokenizer_isl: int | None
    canonical_hash: str


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def nominal_isl_for_config(dataset_config: str) -> int | None:
    try:
        return CONFIG_TO_NOMINAL_ISL[dataset_config]
    except KeyError as exc:
        supported = ", ".join(EXPECTED_CONFIGS)
        raise ValueError(
            f"unsupported SPEED-Bench config {dataset_config!r}; expected one of {supported}"
        ) from exc


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _coerce_row(row: SpeedBenchRecord | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(row, SpeedBenchRecord):
        return asdict(row)
    return row


def _string_field(row: Mapping[str, Any], field_name: str) -> str:
    value = row.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"expected non-empty string field {field_name!r}")
    return value


def _optional_string_field(row: Mapping[str, Any], field_name: str) -> str | None:
    value = row.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"expected optional string field {field_name!r}")
    return value


def _extract_turns(row: Mapping[str, Any]) -> tuple[str, ...]:
    turns = row.get("turns")
    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)):
        raise ValueError("expected turns to be a sequence of strings")
    result: list[str] = []
    for index, turn in enumerate(turns):
        if not isinstance(turn, str) or not turn:
            raise ValueError(f"expected turns[{index}] to be a non-empty string")
        lowered = turn.lower()
        if turn.startswith(UNRESOLVED_TURNS_MARKER) or any(
            token in lowered for token in MASK_TOKENS
        ):
            raise ValueError("masked row contains unresolved turns")
        result.append(turn)
    if not result:
        raise ValueError("expected at least one turn")
    return tuple(result)


def _reject_masked_row(row: Mapping[str, Any]) -> None:
    for field_name in MASK_FIELD_NAMES:
        value = row.get(field_name)
        if value not in (None, False, "", "resolved"):
            raise ValueError(f"masked row must be resolved before use: {field_name}")


def _coerce_actual_isl(
    row: Mapping[str, Any],
    actual_tokenizer_isl: int | None,
) -> int | None:
    if actual_tokenizer_isl is not None:
        return actual_tokenizer_isl
    value = row.get("actual_tokenizer_isl")
    if value is None:
        return None
    if not isinstance(value, int) or value <= 0:
        raise ValueError("actual_tokenizer_isl must be a positive integer")
    return value


def build_record(
    row: SpeedBenchRecord | Mapping[str, Any],
    *,
    dataset_config: str,
    actual_tokenizer_isl: int | None = None,
) -> SpeedBenchRecord:
    raw_row = _coerce_row(row)
    nominal_isl = nominal_isl_for_config(dataset_config)
    _reject_masked_row(raw_row)
    turns = _extract_turns(raw_row)
    multiturn_value = raw_row.get("multiturn")
    multiturn = len(turns) > 1 if multiturn_value is None else bool(multiturn_value)
    payload = {
        "question_id": _string_field(raw_row, "question_id"),
        "category": _string_field(raw_row, "category"),
        "sub_category": _string_field(raw_row, "sub_category"),
        "turns": list(turns),
        "source": _string_field(raw_row, "source"),
        "src_id": _string_field(raw_row, "src_id"),
        "difficulty": _optional_string_field(raw_row, "difficulty"),
        "multiturn": multiturn,
        "dataset_config": dataset_config,
        "nominal_isl": nominal_isl,
        "actual_tokenizer_isl": _coerce_actual_isl(raw_row, actual_tokenizer_isl),
    }
    payload_hash = sha256_bytes(_canonical_json(payload).encode("utf-8"))
    return SpeedBenchRecord(
        question_id=payload["question_id"],
        category=payload["category"],
        sub_category=payload["sub_category"],
        turns=turns,
        source=payload["source"],
        src_id=payload["src_id"],
        difficulty=payload["difficulty"],
        multiturn=payload["multiturn"],
        dataset_config=payload["dataset_config"],
        nominal_isl=payload["nominal_isl"],
        actual_tokenizer_isl=payload["actual_tokenizer_isl"],
        canonical_hash=payload_hash,
    )


def build_records(
    rows: Iterable[SpeedBenchRecord | Mapping[str, Any]],
    *,
    dataset_config: str,
    actual_tokenizer_isl: int | None = None,
) -> tuple[SpeedBenchRecord, ...]:
    return tuple(
        build_record(
            row,
            dataset_config=dataset_config,
            actual_tokenizer_isl=actual_tokenizer_isl,
        )
        for row in rows
    )


def count_categories(records: Sequence[SpeedBenchRecord]) -> dict[str, int]:
    counts = {category: 0 for category in THROUGHPUT_CATEGORY_ORDER}
    for record in records:
        if record.category in counts:
            counts[record.category] += 1
    return counts


def _pick_category_records(
    records: Sequence[SpeedBenchRecord],
    *,
    category: str,
    seed: int,
) -> list[SpeedBenchRecord]:
    candidates = [
        record
        for record in records
        if record.category == category and record.nominal_isl is not None
    ]
    if len(candidates) < 16:
        raise ValueError(f"need at least 16 throughput rows for category {category!r}")
    chosen = sorted(candidates, key=lambda record: record.canonical_hash)
    random.Random(f"{seed}:{category}").shuffle(chosen)
    return chosen[:16]


def select_sync_overlay_rows(
    records: Sequence[SpeedBenchRecord],
    *,
    seed: int,
) -> tuple[tuple[SpeedBenchRecord, ...], ...]:
    selected = {
        category: _pick_category_records(records, category=category, seed=seed)
        for category in THROUGHPUT_CATEGORY_ORDER
    }
    offsets = {category: 0 for category in THROUGHPUT_CATEGORY_ORDER}
    batches: list[tuple[SpeedBenchRecord, ...]] = []
    for layout in SYNC_OVERLAY_LAYOUTS:
        batch: list[SpeedBenchRecord] = []
        for category in THROUGHPUT_CATEGORY_ORDER:
            start = offsets[category]
            stop = start + layout[category]
            batch.extend(selected[category][start:stop])
            offsets[category] = stop
        batches.append(tuple(batch))
    return tuple(batches)


def build_prepared_manifest(prepared_root: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for config_name in EXPECTED_CONFIGS:
        parquet_path = prepared_root / config_name / "test.parquet"
        if not parquet_path.is_file():
            raise FileNotFoundError(f"missing prepared parquet: {parquet_path}")
        entries.append(
            {
                "config_name": config_name,
                "relative_path": str(parquet_path.relative_to(prepared_root)),
                "sha256": sha256_file(parquet_path),
                "nominal_isl": nominal_isl_for_config(config_name),
                "actual_tokenizer_isl": None,
                "overlay_eligible": config_name.startswith("throughput_"),
            }
        )
    return {
        "schema_version": 1,
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "license_name": DATASET_LICENSE_NAME,
            "license_files": list(DATASET_LICENSE_FILES),
        },
        "model_optimizer": {
            "repo": MODELOPT_REPO,
            "repo_url": MODELOPT_REPO_URL,
            "revision": MODELOPT_REVISION,
            "prepare_data_script": MODELOPT_PREPARE_DATA_SCRIPT,
            "license_file": MODELOPT_LICENSE_FILE,
        },
        "prepared_configs": entries,
    }


def write_prepared_manifest(prepared_root: Path, output_path: Path) -> dict[str, Any]:
    manifest = build_prepared_manifest(prepared_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    write_manifest = subparsers.add_parser(
        "write-manifest",
        help="Write a deterministic prepared-data manifest for pinned SPEED-Bench outputs.",
    )
    write_manifest.add_argument("--prepared-root", type=Path, required=True)
    write_manifest.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "write-manifest":
        manifest = write_prepared_manifest(args.prepared_root, args.output)
        print(
            json.dumps(
                {
                    "prepared_configs": len(manifest["prepared_configs"]),
                    "output": str(args.output),
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
