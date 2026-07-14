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

"""Measure same-device temporal overlap between NCCL A2A and expert GEMMs."""

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
A2A_KERNEL_REGEX = r"\bnccl\w*(?:AllToAll|SendRecv)\w*\b"
FUSED_EXPERT_GEMM_REGEX = r"BlockScaledMoEGroupedGemm\w*Kernel(?:_|\b)"
BASELINE_EXPERT_GEMM_REGEX = r"(?:^|[^A-Za-z0-9])nvjet_sm100_[A-Za-z0-9_]*"
A2A_KERNEL_PATTERN = re.compile(A2A_KERNEL_REGEX, re.IGNORECASE)
EXPERT_GEMM_PATTERNS = (
    re.compile(FUSED_EXPERT_GEMM_REGEX, re.IGNORECASE),
    re.compile(BASELINE_EXPERT_GEMM_REGEX, re.IGNORECASE),
)
LIMITATIONS = [
    "Kernel classification is limited to bounded regexes "
    f"A2A={A2A_KERNEL_REGEX!r}, fused_expert_gemm={FUSED_EXPERT_GEMM_REGEX!r}, "
    f"baseline_expert_gemm={BASELINE_EXPERT_GEMM_REGEX!r}; unmatched kernels "
    "are excluded.",
    "Temporal overlap is computed from unioned kernel intervals on the same "
    "Nsight deviceId; stream and rank causality are not inferred.",
]


class AnalyzerError(RuntimeError):
    """Raised when profile evidence cannot be interpreted unambiguously."""


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    sha256: str
    device: int
    inode: int
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class KernelSchema:
    kernel_table: str
    start_column: str
    end_column: str
    device_column: str
    string_id_columns: tuple[str, ...]
    direct_name_column: str | None
    string_table: str | None
    string_id_column: str | None
    string_value_column: str | None


Interval = tuple[int, int]
IntervalsByDevice = dict[int, list[Interval]]


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _snapshot_regular_file(path: Path, label: str) -> FileSnapshot:
    absolute = _absolute_path(path)
    try:
        path_stat = absolute.lstat()
    except FileNotFoundError as error:
        raise AnalyzerError(f"{label} is missing: {absolute}") from error
    if stat.S_ISLNK(path_stat.st_mode):
        raise AnalyzerError(f"{label} must not be a symbolic link: {absolute}")
    if not stat.S_ISREG(path_stat.st_mode):
        raise AnalyzerError(f"{label} must be a regular file: {absolute}")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        raise AnalyzerError(f"unable to open {label}: {absolute}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AnalyzerError(f"{label} must be a regular file: {absolute}")
        if (path_stat.st_dev, path_stat.st_ino) != (before.st_dev, before.st_ino):
            raise AnalyzerError(f"{label} changed while it was opened: {absolute}")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise AnalyzerError(
            f"{label} changed while its digest was computed: {absolute}"
        )
    return FileSnapshot(
        path=absolute,
        sha256=digest.hexdigest(),
        device=before.st_dev,
        inode=before.st_ino,
        size=before.st_size,
        mtime_ns=before.st_mtime_ns,
    )


def _require_unchanged(expected: FileSnapshot, label: str) -> None:
    observed = _snapshot_regular_file(expected.path, label)
    if observed != expected:
        raise AnalyzerError(f"{label} changed during analysis: {expected.path}")


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _table_names(connection: sqlite3.Connection) -> list[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    )
    names = [row[0] for row in rows]
    if any(not isinstance(name, str) or not name for name in names):
        raise AnalyzerError("SQLite schema contains an invalid table name")
    return names


def _table_columns(connection: sqlite3.Connection, table: str) -> dict[str, str]:
    rows = connection.execute(f"PRAGMA table_info({_quote_identifier(table)})")
    columns: dict[str, str] = {}
    for row in rows:
        name = row[1]
        if not isinstance(name, str) or not name:
            raise AnalyzerError(f"SQLite table {table!r} has an invalid column name")
        normalized = name.casefold()
        if normalized in columns:
            raise AnalyzerError(
                f"SQLite table {table!r} has ambiguous case-insensitive columns"
            )
        columns[normalized] = name
    return columns


def _discover_kernel_candidate(
    connection: sqlite3.Connection,
    table: str,
) -> KernelSchema | None:
    if "kernel" not in table.casefold():
        return None
    columns = _table_columns(connection, table)
    if not {"start", "end"}.issubset(columns):
        return None

    device_aliases = [
        columns[name] for name in ("deviceid", "device_id", "device") if name in columns
    ]
    if len(device_aliases) > 1:
        raise AnalyzerError(f"ambiguous kernel device columns in table {table!r}")
    if not device_aliases:
        return None

    string_id_columns = tuple(
        columns[name] for name in ("shortname", "demangledname") if name in columns
    )
    direct_name_columns = [
        columns[name] for name in ("name", "kernelname") if name in columns
    ]
    if len(direct_name_columns) > 1 or (direct_name_columns and string_id_columns):
        raise AnalyzerError(f"ambiguous kernel name columns in table {table!r}")
    if not direct_name_columns and not string_id_columns:
        return None

    return KernelSchema(
        kernel_table=table,
        start_column=columns["start"],
        end_column=columns["end"],
        device_column=device_aliases[0],
        string_id_columns=string_id_columns,
        direct_name_column=direct_name_columns[0] if direct_name_columns else None,
        string_table=None,
        string_id_column=None,
        string_value_column=None,
    )


def _discover_schema(connection: sqlite3.Connection) -> KernelSchema:
    tables = _table_names(connection)
    candidates = [
        candidate
        for table in tables
        if (candidate := _discover_kernel_candidate(connection, table)) is not None
    ]
    if not candidates:
        raise AnalyzerError("no unambiguous kernel interval table found")
    if len(candidates) != 1:
        raise AnalyzerError(
            "ambiguous kernel interval tables: "
            + ", ".join(sorted(candidate.kernel_table for candidate in candidates))
        )
    schema = candidates[0]
    if schema.direct_name_column is not None:
        return schema

    string_candidates: list[tuple[str, str, str]] = []
    for table in tables:
        if "string" not in table.casefold():
            continue
        columns = _table_columns(connection, table)
        if "id" in columns and "value" in columns:
            string_candidates.append((table, columns["id"], columns["value"]))
    if not string_candidates:
        raise AnalyzerError("no string ID table found for kernel name columns")
    if len(string_candidates) != 1:
        raise AnalyzerError(
            "ambiguous string ID tables: "
            + ", ".join(sorted(table for table, _, _ in string_candidates))
        )
    string_table, id_column, value_column = string_candidates[0]
    return KernelSchema(
        kernel_table=schema.kernel_table,
        start_column=schema.start_column,
        end_column=schema.end_column,
        device_column=schema.device_column,
        string_id_columns=schema.string_id_columns,
        direct_name_column=None,
        string_table=string_table,
        string_id_column=id_column,
        string_value_column=value_column,
    )


def _kernel_query(schema: KernelSchema) -> str:
    kernel = _quote_identifier(schema.kernel_table)
    start = _quote_identifier(schema.start_column)
    end = _quote_identifier(schema.end_column)
    device = _quote_identifier(schema.device_column)
    if schema.direct_name_column is not None:
        name_expression = f"k.{_quote_identifier(schema.direct_name_column)}"
        join = ""
    else:
        if (
            schema.string_table is None
            or schema.string_id_column is None
            or schema.string_value_column is None
            or not schema.string_id_columns
        ):
            raise AnalyzerError("incomplete kernel string ID mapping")
        references = ", ".join(
            f"k.{_quote_identifier(column)}" for column in schema.string_id_columns
        )
        reference_expression = (
            references
            if len(schema.string_id_columns) == 1
            else f"COALESCE({references})"
        )
        string_table = _quote_identifier(schema.string_table)
        string_id = _quote_identifier(schema.string_id_column)
        string_value = _quote_identifier(schema.string_value_column)
        join = (
            f" LEFT JOIN {string_table} AS s ON s.{string_id} = {reference_expression}"
        )
        name_expression = f"s.{string_value}"
    return (
        f"SELECT k.{start}, k.{end}, k.{device}, {name_expression} "
        f"FROM {kernel} AS k{join} ORDER BY k.{device}, k.{start}, k.{end}"
    )


def _validate_string_id_mapping(
    connection: sqlite3.Connection,
    schema: KernelSchema,
) -> None:
    if schema.string_table is None or schema.string_id_column is None:
        return
    table = _quote_identifier(schema.string_table)
    id_column = _quote_identifier(schema.string_id_column)
    duplicate = connection.execute(
        f"SELECT {id_column} FROM {table} "
        f"GROUP BY {id_column} HAVING COUNT(*) != 1 LIMIT 1"
    ).fetchone()
    if duplicate is not None:
        raise AnalyzerError(
            f"ambiguous string ID mapping in table {schema.string_table!r}"
        )


def _require_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AnalyzerError(f"{label} must be an integer")
    return value


def _load_matching_intervals(
    connection: sqlite3.Connection,
    schema: KernelSchema,
) -> tuple[IntervalsByDevice, IntervalsByDevice, int, int]:
    a2a: IntervalsByDevice = {}
    expert_gemm: IntervalsByDevice = {}
    a2a_count = 0
    expert_gemm_count = 0
    for start_value, end_value, device_value, name_value in connection.execute(
        _kernel_query(schema)
    ):
        start = _require_integer(start_value, "kernel start")
        end = _require_integer(end_value, "kernel end")
        device = _require_integer(device_value, "kernel device")
        if end <= start:
            raise AnalyzerError(
                f"non-positive kernel interval on device {device}: [{start}, {end})"
            )
        if device < 0:
            raise AnalyzerError(f"kernel device must be non-negative: {device}")
        if not isinstance(name_value, str) or not name_value:
            raise AnalyzerError(
                "kernel name could not be resolved through the SQLite schema"
            )

        interval = (start, end)
        if A2A_KERNEL_PATTERN.search(name_value):
            a2a.setdefault(device, []).append(interval)
            a2a_count += 1
        if any(pattern.search(name_value) for pattern in EXPERT_GEMM_PATTERNS):
            expert_gemm.setdefault(device, []).append(interval)
            expert_gemm_count += 1
    return a2a, expert_gemm, a2a_count, expert_gemm_count


def _merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    merged: list[Interval] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        previous_start, previous_end = merged[-1]
        merged[-1] = (previous_start, max(previous_end, end))
    return merged


def _union_by_device(intervals: IntervalsByDevice) -> IntervalsByDevice:
    return {device: _merge_intervals(values) for device, values in intervals.items()}


def _duration(intervals: IntervalsByDevice) -> int:
    return sum(end - start for values in intervals.values() for start, end in values)


def _intersection_duration(left: IntervalsByDevice, right: IntervalsByDevice) -> int:
    duration = 0
    for device in left.keys() & right.keys():
        left_intervals = left[device]
        right_intervals = right[device]
        left_index = 0
        right_index = 0
        while left_index < len(left_intervals) and right_index < len(right_intervals):
            left_start, left_end = left_intervals[left_index]
            right_start, right_end = right_intervals[right_index]
            duration += max(0, min(left_end, right_end) - max(left_start, right_start))
            if left_end <= right_end:
                left_index += 1
            else:
                right_index += 1
    return duration


def _analyze_sqlite(path: Path, source_profile_sha256: str) -> dict[str, object]:
    sqlite_snapshot = _snapshot_regular_file(path, "SQLite profile")
    uri = sqlite_snapshot.path.as_uri() + "?mode=ro&immutable=1"
    try:
        with sqlite3.connect(uri, uri=True) as connection:
            connection.execute("PRAGMA query_only = ON")
            schema = _discover_schema(connection)
            _validate_string_id_mapping(connection, schema)
            a2a, expert_gemm, a2a_count, expert_gemm_count = _load_matching_intervals(
                connection, schema
            )
    except sqlite3.Error as error:
        raise AnalyzerError(
            f"unable to query Nsight SQLite profile: {error}"
        ) from error
    _require_unchanged(sqlite_snapshot, "SQLite profile")

    a2a_union = _union_by_device(a2a)
    expert_gemm_union = _union_by_device(expert_gemm)
    a2a_duration = _duration(a2a_union)
    expert_gemm_duration = _duration(expert_gemm_union)
    overlap_duration = _intersection_duration(a2a_union, expert_gemm_union)
    a2a_ratio = overlap_duration / a2a_duration if a2a_duration > 0 else 0.0
    gemm_ratio = (
        overlap_duration / expert_gemm_duration if expert_gemm_duration > 0 else 0.0
    )
    verified = (
        a2a_count > 0
        and expert_gemm_count > 0
        and overlap_duration > 0
        and math.isfinite(a2a_ratio)
        and a2a_ratio > 0
        and math.isfinite(gemm_ratio)
        and gemm_ratio > 0
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "source_profile_sha256": source_profile_sha256,
        "a2a_interval_count": a2a_count,
        "expert_gemm_interval_count": expert_gemm_count,
        "overlap_duration_ns": overlap_duration,
        "a2a_overlap_ratio": a2a_ratio,
        "gemm_overlap_ratio": gemm_ratio,
        "temporal_overlap_verified": verified,
        "limitations": LIMITATIONS,
    }


def _prepare_output_path(path: Path) -> Path:
    absolute = _absolute_path(path)
    try:
        parent = absolute.parent.resolve(strict=True)
    except FileNotFoundError as error:
        raise AnalyzerError(
            f"output directory is missing: {absolute.parent}"
        ) from error
    if not parent.is_dir():
        raise AnalyzerError(f"output parent must be a directory: {parent}")
    output = parent / absolute.name
    if os.path.lexists(output):
        raise AnalyzerError(f"refusing to overwrite output path: {output}")
    return output


def _write_json_atomic(output: Path, analysis: dict[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            json.dump(analysis, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    except FileExistsError as error:
        raise AnalyzerError(f"refusing to overwrite output path: {output}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _export_and_analyze(
    source: FileSnapshot,
    output_root: Path,
    nsys_bin: str,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(
        prefix=".a2a-temporal-overlap-", dir=output_root
    ) as temporary_directory:
        exported = Path(temporary_directory) / "profile.sqlite"
        command = [
            nsys_bin,
            "export",
            "-t",
            "sqlite",
            "-o",
            str(exported),
            str(source.path),
        ]
        try:
            subprocess.run(
                command,
                check=True,
                shell=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError as error:
            raise AnalyzerError(f"nsys executable is missing: {nsys_bin}") from error
        except subprocess.CalledProcessError as error:
            detail = error.stderr.strip() or error.stdout.strip() or str(error)
            raise AnalyzerError(f"nsys export failed: {detail}") from error
        _require_unchanged(source, "source profile")
        return _analyze_sqlite(exported, source.sha256)


def analyze_profile(profile: Path, output: Path, nsys_bin: str) -> Path:
    output_path = _prepare_output_path(output)
    source = _snapshot_regular_file(profile, "source profile")
    suffix = source.path.name.casefold()
    if suffix.endswith(".nsys-rep"):
        analysis = _export_and_analyze(source, output_path.parent, nsys_bin)
    elif suffix.endswith(".sqlite"):
        analysis = _analyze_sqlite(source.path, source.sha256)
        _require_unchanged(source, "source profile")
    else:
        raise AnalyzerError(
            "source profile must have a .nsys-rep or .sqlite filename suffix"
        )
    _write_json_atomic(output_path, analysis)
    return output_path


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure same-device temporal overlap between NCCL A2A/SendRecv and "
            "expert grouped-GEMM kernel intervals."
        )
    )
    parser.add_argument(
        "profile", type=Path, help="Input .nsys-rep or exported .sqlite"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New JSON output path"
    )
    parser.add_argument(
        "--nsys-bin",
        default=os.environ.get("CUSTOM_NSYS_EXE", "nsys"),
        help="nsys executable used only for .nsys-rep input",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        output = analyze_profile(args.profile, args.output, args.nsys_bin)
    except (AnalyzerError, OSError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
