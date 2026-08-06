#!/usr/bin/env python3
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

"""Summarize Nsight Systems CUDA Graph launch evidence."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import statistics
import subprocess
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


GRAPH_LAUNCH_NAME = re.compile(r"^(?:cuda|cu)GraphLaunch(?:_ptsz)?$")


def parse_cuda_api_summary(report: str) -> dict[str, int]:
    """Parse one `nsys stats cuda_api_sum` CSV report."""
    lines = report.splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if "Name" in line
            and ("Num Calls" in line or "Instances" in line or "Calls" in line)
        ),
        None,
    )
    if header_index is None:
        raise ValueError("Nsight cuda_api_sum report has no CSV header")

    reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
    calls_column = next(
        (
            name
            for name in ("Num Calls", "Instances", "Calls")
            if name in (reader.fieldnames or [])
        ),
        None,
    )
    if calls_column is None:
        raise ValueError("Nsight cuda_api_sum report has no call-count column")

    cuda_api_calls = 0
    cuda_graph_launch_calls = 0
    for row in reader:
        if row.get(calls_column) is None or row.get("Name") is None:
            continue
        calls = int(float(row[calls_column]))
        cuda_api_calls += calls
        if GRAPH_LAUNCH_NAME.fullmatch(row["Name"]):
            cuda_graph_launch_calls += calls
    return {
        "nsys_cuda_api_calls": cuda_api_calls,
        "nsys_cuda_graph_launch_calls": cuda_graph_launch_calls,
    }


def summarize_profiles(
    profiles: Sequence[Mapping[str, int]],
) -> dict[str, object]:
    """Aggregate process coverage and CUDA API launch share."""
    graph_calls = [profile["nsys_cuda_graph_launch_calls"] for profile in profiles]
    total_api_calls = sum(profile["nsys_cuda_api_calls"] for profile in profiles)
    total_graph_calls = sum(graph_calls)
    profiles_with_graphs = sum(count > 0 for count in graph_calls)
    profile_count = len(profiles)
    return {
        "nsys_profile_count": profile_count,
        "nsys_profiles_with_cuda_graph_launches": profiles_with_graphs,
        "nsys_process_coverage_pct": round(
            100 * profiles_with_graphs / profile_count if profile_count else 0.0,
            6,
        ),
        "nsys_total_cuda_api_calls": total_api_calls,
        "nsys_total_cuda_graph_launch_calls": total_graph_calls,
        "nsys_cuda_graph_launch_share_of_cuda_api_calls_pct": round(
            100 * total_graph_calls / total_api_calls if total_api_calls else 0.0,
            6,
        ),
        "nsys_cuda_graph_launch_calls_min": min(graph_calls, default=0),
        "nsys_cuda_graph_launch_calls_median": (
            statistics.median(graph_calls) if graph_calls else 0
        ),
        "nsys_cuda_graph_launch_calls_max": max(graph_calls, default=0),
    }


def _profile_counts(profile: Path, nsys: str) -> tuple[Path, dict[str, int]]:
    result = subprocess.run(
        [
            nsys,
            "stats",
            "--report",
            "cuda_api_sum",
            "--format",
            "csv",
            str(profile),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return profile, parse_cuda_api_summary(result.stdout)


def _parse_label(value: str) -> tuple[str, Path]:
    label, separator, directory = value.partition("=")
    if not separator or not label or not directory:
        raise argparse.ArgumentTypeError("--label must use NAME=PROFILE_DIRECTORY")
    path = Path(directory)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"profile directory does not exist: {path}")
    return label, path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label",
        action="append",
        required=True,
        type=_parse_label,
        metavar="NAME=PROFILE_DIRECTORY",
    )
    parser.add_argument("--nsys", default="nsys")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")

    payload: dict[str, object] = {}
    for label, directory in args.label:
        profile_paths = sorted(directory.rglob("*.nsys-rep"))
        if not profile_paths:
            parser.error(f"no .nsys-rep profiles found under {directory}")
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            results = list(
                executor.map(
                    lambda profile: _profile_counts(profile, args.nsys),
                    profile_paths,
                )
            )
        counts = [profile_counts for _, profile_counts in results]
        summary: dict[str, object] = summarize_profiles(counts)
        summary["profiles"] = [
            {"path": str(path), **profile_counts} for path, profile_counts in results
        ]
        payload[label] = summary

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output_json)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
