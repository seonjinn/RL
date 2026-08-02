import json
from pathlib import Path
from typing import Any

import pytest

from experiments.mxfp8_adaptive_rollout_v0251.build_shape_manifest import (
    ShapeTraceError,
    build_shape_manifest,
    main,
    write_shape_outputs,
)


def _record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "event": "mxfp8_dense_shape",
        "family": "FC1",
        "hostname": "node-a",
        "k": 8192,
        "layout": "128x4",
        "m": 1000,
        "n_logical": 8768,
        "n_physical": 8832,
        "pid": 101,
        "prefix": "model.layers.0.mlp.fc1",
    }
    record.update(overrides)
    return record


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_manifest_deduplicates_ranks_and_preserves_provenance(tmp_path: Path) -> None:
    rank0 = tmp_path / "rank0.jsonl"
    rank1 = tmp_path / "rank1.jsonl"
    repeated = _record()
    _write_jsonl(rank0, [repeated, repeated, _record(m=8, layout="8x4")])
    _write_jsonl(
        rank1,
        [
            _record(
                hostname="node-b",
                pid=202,
                prefix="model.layers.47.mlp.fc1",
            )
        ],
    )

    manifest = build_shape_manifest([rank1, rank0])

    assert manifest == {
        "schema_version": 1,
        "input_file_count": 2,
        "record_count": 4,
        "unique_signature_count": 2,
        "duplicate_record_count": 2,
        "layout_counts": {
            "8x4": {"record_count": 1, "unique_signature_count": 1},
            "128x4": {"record_count": 3, "unique_signature_count": 1},
        },
        "signatures": [
            {
                "m": 8,
                "n_logical": 8768,
                "n_physical": 8832,
                "k": 8192,
                "layout": "8x4",
                "record_count": 1,
                "prefixes": ["model.layers.0.mlp.fc1"],
                "provenance": [
                    {
                        "source_file": "rank0.jsonl",
                        "hostname": "node-a",
                        "pid": 101,
                        "prefix": "model.layers.0.mlp.fc1",
                        "family": "FC1",
                        "record_count": 1,
                    }
                ],
            },
            {
                "m": 1000,
                "n_logical": 8768,
                "n_physical": 8832,
                "k": 8192,
                "layout": "128x4",
                "record_count": 3,
                "prefixes": [
                    "model.layers.0.mlp.fc1",
                    "model.layers.47.mlp.fc1",
                ],
                "provenance": [
                    {
                        "source_file": "rank0.jsonl",
                        "hostname": "node-a",
                        "pid": 101,
                        "prefix": "model.layers.0.mlp.fc1",
                        "family": "FC1",
                        "record_count": 2,
                    },
                    {
                        "source_file": "rank1.jsonl",
                        "hostname": "node-b",
                        "pid": 202,
                        "prefix": "model.layers.47.mlp.fc1",
                        "family": "FC1",
                        "record_count": 1,
                    },
                ],
            },
        ],
    }


def test_outputs_are_deterministic_and_use_physical_shmoo_n(tmp_path: Path) -> None:
    rank0 = tmp_path / "rank0.jsonl"
    rank1 = tmp_path / "rank1.jsonl"
    _write_jsonl(rank0, [_record(m=1000), _record(m=8, layout="8x4")])
    _write_jsonl(rank1, [_record(m=256), _record(m=8, layout="8x4")])

    first = build_shape_manifest([rank1, rank0])
    second = build_shape_manifest([rank0, rank1])
    first_output = tmp_path / "first" / "manifest.json"
    second_output = tmp_path / "second" / "manifest.json"
    write_shape_outputs(first, first_output, first_output.parent / "shmoo")
    write_shape_outputs(second, second_output, second_output.parent / "shmoo")

    assert first_output.read_bytes() == second_output.read_bytes()
    assert (first_output.parent / "shmoo/shapes_8x4.txt").read_text() == (
        "8,8832,8192\n"
    )
    assert (first_output.parent / "shmoo/shapes_128x4.txt").read_text() == (
        "256,8832,8192;1000,8832,8192\n"
    )


@pytest.mark.parametrize(
    ("record", "match"),
    [
        ({"event": "other"}, "missing required fields"),
        (_record(event="other"), "unexpected event"),
        (_record(m=True), "m must be a positive integer"),
        (_record(k="8192"), "k must be a positive integer"),
        (_record(prefix=""), "prefix must be a non-empty string"),
        (_record(layout="32x4"), "unsupported layout"),
        (_record(n_logical=9000), "n_physical must be greater than or equal"),
        (_record(k=8193), "k must be divisible by 32"),
    ],
)
def test_manifest_rejects_malformed_records(
    tmp_path: Path, record: dict[str, Any], match: str
) -> None:
    trace = tmp_path / "rank0.jsonl"
    _write_jsonl(trace, [record])

    with pytest.raises(ShapeTraceError, match=match):
        build_shape_manifest([trace])


def test_manifest_reports_invalid_json_location(tmp_path: Path) -> None:
    trace = tmp_path / "rank0.jsonl"
    trace.write_text("\n{not-json}\n", encoding="utf-8")

    with pytest.raises(ShapeTraceError, match=r"rank0\.jsonl:2: invalid JSON"):
        build_shape_manifest([trace])


def test_manifest_rejects_inconsistent_prefix_metadata(tmp_path: Path) -> None:
    trace = tmp_path / "rank0.jsonl"
    _write_jsonl(trace, [_record(), _record(n_physical=8960)])

    with pytest.raises(ShapeTraceError, match="inconsistent dimensions for prefix"):
        build_shape_manifest([trace])


def test_cli_writes_manifest_and_layout_files(tmp_path: Path) -> None:
    trace = tmp_path / "rank0.jsonl"
    output = tmp_path / "artifacts" / "manifest.json"
    shmoo_dir = tmp_path / "artifacts" / "shmoo"
    _write_jsonl(trace, [_record(m=8, layout="8x4")])

    assert (
        main(
            [
                str(trace),
                "--output",
                str(output),
                "--shmoo-dir",
                str(shmoo_dir),
            ]
        )
        == 0
    )

    assert json.loads(output.read_text())["unique_signature_count"] == 1
    assert (shmoo_dir / "shapes_8x4.txt").read_text() == "8,8832,8192\n"
