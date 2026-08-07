#!/usr/bin/env python3
"""Convert actual NSys NVTX range summaries into audit component CSV evidence."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import csv
from pathlib import Path


PREFIX = "MXFP8_MOE_AUDIT|"
FIELDS = (
    "signature_key",
    "cache_key",
    "arm",
    "component",
    "tactic",
    "cache_event",
    "call_weight",
)
OUTPUT_FIELDS = (*FIELDS, "call_count", "median_us")


def _range_fields(value: str) -> dict[str, str] | None:
    if not value.startswith(PREFIX):
        return None
    result: dict[str, str] = {}
    for item in value.removeprefix(PREFIX).split("|"):
        key, separator, field_value = item.partition("=")
        if not separator or not key or not field_value or key in result:
            raise ValueError("malformed MXFP8 MoE audit NVTX range")
        result[key] = field_value
    if tuple(result) != FIELDS:
        raise ValueError("MXFP8 MoE audit NVTX range fields are incomplete")
    if result["arm"] not in {"stock", "candidate"} or result["component"] not in {
        "FC1/GEMM1",
        "FC2/GEMM2",
    }:
        raise ValueError("MXFP8 MoE audit NVTX range has invalid arm or component")
    if result["cache_event"] not in {"cache hit", "fallback"}:
        raise ValueError("MXFP8 MoE audit NVTX range has invalid cache event")
    if len(result["tactic"].split(",")) != 2 or not all(
        item.isdecimal() for item in result["tactic"].split(",")
    ):
        raise ValueError("MXFP8 MoE audit NVTX range has invalid tactic")
    if not result["call_weight"].isdecimal() or int(result["call_weight"]) <= 0:
        raise ValueError("MXFP8 MoE audit NVTX range has invalid call weight")
    return result


def convert(nvtx_csv: Path, output: Path) -> None:
    """Write component rows from actual NSys range count and total duration."""
    with nvtx_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("NSys NVTX CSV has no header")
        range_field = next(
            (
                name
                for name in reader.fieldnames
                if name.lower() in {"range", "name", "nvtx range"}
            ),
            None,
        )
        count_field = next(
            (
                name
                for name in reader.fieldnames
                if name.lower() in {"instances", "calls", "count"}
            ),
            None,
        )
        total_field = next(
            (
                name
                for name in reader.fieldnames
                if "total" in name.lower() and "time" in name.lower()
            ),
            None,
        )
        if range_field is None or count_field is None or total_field is None:
            raise ValueError("NSys NVTX CSV lacks range/count/total-time columns")
        rows: list[dict[str, str]] = []
        for raw in reader:
            tagged = _range_fields(raw.get(range_field, ""))
            if tagged is None:
                continue
            try:
                instances = int(raw[count_field].replace(",", ""))
                total_ns = float(raw[total_field].replace(",", ""))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("NSys NVTX CSV has malformed timing fields") from error
            if instances <= 0 or total_ns <= 0:
                raise ValueError("NSys NVTX CSV has nonpositive timing evidence")
            rows.append(
                {
                    **tagged,
                    "call_count": str(instances),
                    "median_us": f"{total_ns / instances / 1000.0:.9g}",
                }
            )
    if not rows:
        raise ValueError("NSys NVTX CSV has no MXFP8 MoE audit component ranges")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)  # pyright: ignore[reportArgumentType]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvtx-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    convert(args.nvtx_csv, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
