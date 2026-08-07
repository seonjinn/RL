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
    "comparison_tactic",
    "cache_event",
    "call_weight",
)
OUTPUT_FIELDS = (*FIELDS, "call_count", "mean_us")
CUMULATIVE_COMPONENTS = (
    "FC1/GEMM1 cumulative",
    "FC1+FC2/GEMM1+GEMM2 cumulative",
)


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
        *CUMULATIVE_COMPONENTS,
    }:
        raise ValueError("MXFP8 MoE audit NVTX range has invalid arm or component")
    if result["cache_event"] not in {"cache hit", "fallback"}:
        raise ValueError("MXFP8 MoE audit NVTX range has invalid cache event")
    for field_name in ("tactic", "comparison_tactic"):
        if len(result[field_name].split(",")) != 2 or not all(
            item.isdecimal() for item in result[field_name].split(",")
        ):
            raise ValueError(f"MXFP8 MoE audit NVTX range has invalid {field_name}")
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
        cumulative: dict[
            tuple[str, ...], dict[str, tuple[dict[str, str], int, float]]
        ] = {}
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
            component = tagged["component"]
            key = tuple(
                tagged[field]
                for field in FIELDS
                if field not in {"cache_event", "cache_key", "component"}
            )
            stages = cumulative.setdefault(key, {})
            if component in stages:
                raise ValueError("duplicate MXFP8 MoE cumulative NVTX range")
            stages[component] = (tagged, instances, total_ns)
        for stages in cumulative.values():
            if set(stages) != set(CUMULATIVE_COMPONENTS):
                raise ValueError(
                    "NSys cumulative timing requires both FC1 and paired ranges"
                )
            fc1_tagged, fc1_instances, fc1_total_ns = stages["FC1/GEMM1 cumulative"]
            pair_tagged, pair_instances, pair_total_ns = stages[
                "FC1+FC2/GEMM1+GEMM2 cumulative"
            ]
            if fc1_instances != pair_instances:
                raise ValueError("NSys cumulative stage counts differ")
            fc2_total_ns = pair_total_ns - fc1_total_ns
            if fc2_total_ns <= 0:
                raise ValueError("NSys cumulative pair time does not exceed FC1 time")
            for tagged, component, total_ns in (
                (fc1_tagged, "FC1/GEMM1", fc1_total_ns),
                (pair_tagged, "FC2/GEMM2", fc2_total_ns),
            ):
                rows.append(
                    {
                        **tagged,
                        "component": component,
                        "call_count": str(fc1_instances),
                        "mean_us": f"{total_ns / fc1_instances / 1000.0:.9g}",
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
