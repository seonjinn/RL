# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


DATASET_ID = "openai/gsm8k"
DATASET_CONFIG = "main"
DATASET_SPLIT = "test"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
EXPECTED_ROWS = 1319


def _load_source_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid source JSON on line {line_number}: {error}"
            ) from error
        if not isinstance(row, dict):
            raise ValueError(f"source row {line_number} is not an object")
        rows.append(row)
    return rows


def _load_huggingface_rows() -> tuple[list[dict[str, Any]], str | None]:
    # datasets is intentionally loaded only for the production materialization path.
    from datasets import load_dataset

    dataset = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split=DATASET_SPLIT,
        revision=DATASET_REVISION,
    )
    fingerprint = getattr(dataset, "_fingerprint", None)
    return [dict(row) for row in dataset], fingerprint


def _normalized_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    questions: set[str] = set()
    for index, row in enumerate(rows):
        question = row.get("question")
        answer = row.get("answer")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"row {index} has no non-empty question")
        if not isinstance(answer, str) or "####" not in answer:
            raise ValueError(f"row {index} has no GSM8K '####' answer delimiter")
        question = question.strip()
        if question in questions:
            raise ValueError(f"row {index} duplicates a previous question")
        questions.add(question)
        extracted_answer = answer.rsplit("####", 1)[1].strip()
        if not extracted_answer:
            raise ValueError(f"row {index} has an empty extracted answer")
        normalized.append(
            {
                "input": question,
                "output": extracted_answer,
                "sample_id": f"gsm8k-test-{index:04d}",
            }
        )
    return normalized


def _encode_jsonl(rows: Iterable[dict[str, str]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
        for row in rows
    ).encode()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_bytes(payload)
    temporary_path.replace(path)


def materialize(
    *,
    output_path: Path,
    manifest_path: Path,
    expected_rows: int,
    expected_sha256: str | None,
    source_jsonl: Path | None,
) -> dict[str, object]:
    if source_jsonl is None:
        source_rows, source_fingerprint = _load_huggingface_rows()
        source = DATASET_ID
    else:
        source_rows = _load_source_jsonl(source_jsonl)
        source_fingerprint = hashlib.sha256(source_jsonl.read_bytes()).hexdigest()
        source = str(source_jsonl.resolve())

    rows = _normalized_rows(source_rows)
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} GSM8K rows, found {len(rows)}")

    encoded = _encode_jsonl(rows)
    jsonl_sha256 = hashlib.sha256(encoded).hexdigest()
    if expected_sha256 is not None and jsonl_sha256 != expected_sha256:
        raise ValueError(
            "materialized GSM8K SHA256 mismatch: "
            f"expected {expected_sha256}, found {jsonl_sha256}"
        )

    manifest: dict[str, object] = {
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "split": DATASET_SPLIT,
        "revision": DATASET_REVISION,
        "source": source,
        "source_fingerprint": source_fingerprint,
        "row_count": len(rows),
        "jsonl_sha256": jsonl_sha256,
        "schema": ["input", "output", "sample_id"],
    }
    _atomic_write(output_path, encoded)
    _atomic_write(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize a pinned, deterministic GSM8K test JSONL"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--source-jsonl", type=Path)
    args = parser.parse_args()

    manifest = materialize(
        output_path=args.output,
        manifest_path=args.manifest,
        expected_rows=args.expected_rows,
        expected_sha256=args.expected_sha256,
        source_jsonl=args.source_jsonl,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
