# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Derive a vLLM DynamicSD schedule from a complete offline profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ProfileRow:
    """One measured batch-size and draft-length latency point."""

    batch_size: int
    k: int
    median_itl_ms: float
    completed_batches: int


@dataclass(frozen=True, slots=True)
class DynamicProfile:
    """Validated immutable DynamicSD profile and its exact identity."""

    source_path: Path
    source_sha256: str
    model_key: str
    target_revision: str
    drafter_revision: str
    runtime_vllm: str
    cuda_graph_mode: str
    dataset_name: str
    dataset_revision: str
    prompt_template_sha256: str
    temperature: float
    top_p: float
    max_model_len: int
    max_num_batched_tokens: int
    max_num_seqs: int
    target_tensor_parallel_size: int
    draft_tensor_parallel_size: int
    num_batches_per_point: int
    batch_sizes: tuple[int, ...]
    k_values: tuple[int, ...]
    acceptance_rate_per_pos: tuple[float, ...]
    rows: tuple[ProfileRow, ...]

    def row_map(self) -> dict[tuple[int, int], ProfileRow]:
        """Index the complete profile grid by ``(batch_size, K)``."""
        return {(row.batch_size, row.k): row for row in self.rows}


@dataclass(frozen=True, slots=True)
class ScheduleRange:
    """One inclusive scheduler batch-size range and its selected K."""

    start_batch: int
    end_batch: int
    k: int


@dataclass(frozen=True, slots=True)
class DerivedSchedule:
    """Dense K decisions compressed into vLLM-compatible ranges."""

    max_num_speculative_tokens: int
    minimum_goodput_gain: float
    selected_k_by_batch: Mapping[int, int]
    ranges: tuple[ScheduleRange, ...]


G_PROFILE_KEYS = {
    "schema_version",
    "calibration_status",
    "model_key",
    "target_revision",
    "drafter_revision",
    "runtime_vllm",
    "cuda_graph_mode",
    "dataset_name",
    "dataset_revision",
    "prompt_template_sha256",
    "temperature",
    "top_p",
    "max_model_len",
    "max_num_batched_tokens",
    "max_num_seqs",
    "target_tensor_parallel_size",
    "draft_tensor_parallel_size",
    "num_batches_per_point",
    "batch_sizes",
    "k_values",
    "acceptance_rate_per_pos",
    "rows",
}
G_REQUIRED_K_VALUES = tuple(range(6))


def _require_int(value: object, name: str, *, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"DynamicSD profile {name} must be an integer >= {minimum}")
    return value


def _require_float(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"DynamicSD profile {name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"DynamicSD profile {name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"DynamicSD profile {name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"DynamicSD profile {name} must be <= {maximum}")
    return result


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"DynamicSD profile {name} must be a non-empty string")
    return value


def _strict_int_list(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"DynamicSD profile {name} must be a non-empty list")
    items = tuple(_require_int(item, name, minimum=0) for item in value)
    if tuple(sorted(set(items))) != items:
        raise ValueError(f"DynamicSD profile {name} must be unique and sorted")
    return items


def load_profile(path: Path) -> DynamicProfile:
    """Load a complete K0-K5 profile and reject ambiguous data."""
    source_path = path.resolve()
    content = source_path.read_bytes()
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("DynamicSD profile must be a JSON object")
    if set(payload) != G_PROFILE_KEYS:
        missing = sorted(G_PROFILE_KEYS - set(payload))
        unknown = sorted(set(payload) - G_PROFILE_KEYS)
        raise ValueError(
            f"DynamicSD profile schema mismatch: missing={missing}, unknown={unknown}"
        )
    if _require_int(payload["schema_version"], "schema_version") != 1:
        raise ValueError("DynamicSD profile schema_version must be 1")
    if payload["calibration_status"] != "complete":
        raise ValueError("DynamicSD profile calibration_status must be complete")

    model_key = _require_string(payload["model_key"], "model_key")
    target_revision = _require_string(payload["target_revision"], "target_revision")
    drafter_revision = _require_string(
        payload["drafter_revision"], "drafter_revision"
    )
    for name, revision in (
        ("target_revision", target_revision),
        ("drafter_revision", drafter_revision),
    ):
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError(f"DynamicSD profile {name} must be a full hex digest")
    prompt_template_sha256 = _require_string(
        payload["prompt_template_sha256"], "prompt_template_sha256"
    )
    if re.fullmatch(r"[0-9a-f]{64}", prompt_template_sha256) is None:
        raise ValueError(
            "DynamicSD profile prompt_template_sha256 must be a full SHA-256 digest"
        )
    dataset_revision = _require_string(
        payload["dataset_revision"], "dataset_revision"
    )
    if re.fullmatch(r"[0-9a-f]{40}", dataset_revision) is None:
        raise ValueError(
            "DynamicSD profile dataset_revision must be a full hex digest"
        )

    batch_sizes = _strict_int_list(payload["batch_sizes"], "batch_sizes")
    if batch_sizes[0] != 1:
        raise ValueError("DynamicSD profile must include batch size 1")
    max_num_seqs = _require_int(payload["max_num_seqs"], "max_num_seqs")
    if batch_sizes[-1] != max_num_seqs:
        raise ValueError(
            "DynamicSD profile largest batch size must equal max_num_seqs"
        )
    k_values = _strict_int_list(payload["k_values"], "k_values")
    if k_values != G_REQUIRED_K_VALUES:
        raise ValueError("DynamicSD profile must measure every K0 through K5")

    raw_acceptance = payload["acceptance_rate_per_pos"]
    if not isinstance(raw_acceptance, list) or len(raw_acceptance) != k_values[-1]:
        raise ValueError(
            "DynamicSD profile acceptance_rate_per_pos must contain five values"
        )
    acceptance = tuple(
        _require_float(item, "acceptance_rate_per_pos", minimum=0.0, maximum=1.0)
        for item in raw_acceptance
    )

    num_batches = _require_int(
        payload["num_batches_per_point"], "num_batches_per_point"
    )
    raw_rows = payload["rows"]
    if not isinstance(raw_rows, list):
        raise ValueError("DynamicSD profile rows must be a list")
    rows: list[ProfileRow] = []
    seen: set[tuple[int, int]] = set()
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict) or set(raw_row) != {
            "batch_size",
            "k",
            "median_itl_ms",
            "completed_batches",
        }:
            raise ValueError("DynamicSD profile row schema is invalid")
        batch_size = _require_int(raw_row["batch_size"], "row batch_size")
        k = _require_int(raw_row["k"], "row K", minimum=0)
        key = (batch_size, k)
        if key in seen:
            raise ValueError(f"DynamicSD profile contains duplicate grid cell {key}")
        seen.add(key)
        completed_batches = _require_int(
            raw_row["completed_batches"], "row completed_batches"
        )
        if completed_batches < num_batches:
            raise ValueError(
                f"DynamicSD profile grid cell {key} has fewer completed batches"
            )
        median_itl_ms = _require_float(
            raw_row["median_itl_ms"], "row median_itl_ms", minimum=0.0
        )
        if median_itl_ms == 0.0:
            raise ValueError("DynamicSD profile row median_itl_ms must be positive")
        rows.append(
            ProfileRow(
                batch_size=batch_size,
                k=k,
                median_itl_ms=median_itl_ms,
                completed_batches=completed_batches,
            )
        )
    expected = {(batch_size, k) for batch_size in batch_sizes for k in k_values}
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(
            f"DynamicSD profile must contain the complete grid: "
            f"missing={missing}, extra={extra}"
        )

    return DynamicProfile(
        source_path=source_path,
        source_sha256=hashlib.sha256(content).hexdigest(),
        model_key=model_key,
        target_revision=target_revision,
        drafter_revision=drafter_revision,
        runtime_vllm=_require_string(payload["runtime_vllm"], "runtime_vllm"),
        cuda_graph_mode=_require_string(
            payload["cuda_graph_mode"], "cuda_graph_mode"
        ),
        dataset_name=_require_string(payload["dataset_name"], "dataset_name"),
        dataset_revision=dataset_revision,
        prompt_template_sha256=prompt_template_sha256,
        temperature=_require_float(payload["temperature"], "temperature"),
        top_p=_require_float(payload["top_p"], "top_p", minimum=0.0, maximum=1.0),
        max_model_len=_require_int(payload["max_model_len"], "max_model_len"),
        max_num_batched_tokens=_require_int(
            payload["max_num_batched_tokens"], "max_num_batched_tokens"
        ),
        max_num_seqs=max_num_seqs,
        target_tensor_parallel_size=_require_int(
            payload["target_tensor_parallel_size"], "target_tensor_parallel_size"
        ),
        draft_tensor_parallel_size=_require_int(
            payload["draft_tensor_parallel_size"], "draft_tensor_parallel_size"
        ),
        num_batches_per_point=num_batches,
        batch_sizes=batch_sizes,
        k_values=k_values,
        acceptance_rate_per_pos=acceptance,
        rows=tuple(sorted(rows, key=lambda row: (row.batch_size, row.k))),
    )


def _interpolate_itl(profile: DynamicProfile, batch_size: int, k: int) -> float:
    rows = profile.row_map()
    exact = rows.get((batch_size, k))
    if exact is not None:
        return exact.median_itl_ms
    lower = max(item for item in profile.batch_sizes if item < batch_size)
    upper = min(item for item in profile.batch_sizes if item > batch_size)
    lower_itl = rows[(lower, k)].median_itl_ms
    upper_itl = rows[(upper, k)].median_itl_ms
    ratio = (batch_size - lower) / (upper - lower)
    return lower_itl + ratio * (upper_itl - lower_itl)


def _compress_ranges(selected: Mapping[int, int]) -> tuple[ScheduleRange, ...]:
    if not selected:
        raise ValueError("DynamicSD selected-K map must not be empty")
    ordered = sorted(selected.items())
    ranges: list[ScheduleRange] = []
    start_batch, current_k = ordered[0]
    previous_batch = start_batch
    for batch_size, k in ordered[1:]:
        if batch_size != previous_batch + 1:
            raise ValueError("DynamicSD selected-K map must be contiguous")
        if k != current_k:
            ranges.append(ScheduleRange(start_batch, previous_batch, current_k))
            start_batch = batch_size
            current_k = k
        previous_batch = batch_size
    ranges.append(ScheduleRange(start_batch, previous_batch, current_k))
    return tuple(ranges)


def derive_schedule(
    profile: DynamicProfile,
    *,
    minimum_goodput_gain: float = 0.0,
) -> DerivedSchedule:
    """Select K by the upstream ``accepted length / median ITL`` rule."""
    if not math.isfinite(minimum_goodput_gain) or minimum_goodput_gain < 0.0:
        raise ValueError("minimum_goodput_gain must be finite and non-negative")
    accepted_lengths = {
        k: 1.0 + sum(profile.acceptance_rate_per_pos[:k])
        for k in profile.k_values
    }
    selected: dict[int, int] = {}
    for batch_size in range(1, profile.max_num_seqs + 1):
        goodput = {
            k: accepted_lengths[k] / _interpolate_itl(profile, batch_size, k)
            for k in profile.k_values
        }
        best_k = max(profile.k_values, key=lambda k: (goodput[k], -k))
        if goodput[best_k] < goodput[0] * (1.0 + minimum_goodput_gain):
            best_k = 0
        selected[batch_size] = best_k
    return DerivedSchedule(
        max_num_speculative_tokens=profile.k_values[-1],
        minimum_goodput_gain=minimum_goodput_gain,
        selected_k_by_batch=selected,
        ranges=_compress_ranges(selected),
    )


def write_schedule(
    profile: DynamicProfile,
    schedule: DerivedSchedule,
    output: Path,
) -> None:
    """Write a deterministic schema-v2 schedule linked to the raw profile."""
    payload: dict[str, Any] = {
        "schema_version": 2,
        "calibration_status": "calibrated",
        "model_key": profile.model_key,
        "target_revision": profile.target_revision,
        "drafter_revision": profile.drafter_revision,
        "source_runtime_vllm": profile.runtime_vllm,
        "target_runtime_vllm": profile.runtime_vllm,
        "target_cuda_graph_mode": profile.cuda_graph_mode,
        "profile_sha256": profile.source_sha256,
        "max_num_speculative_tokens": schedule.max_num_speculative_tokens,
        "selection_metric": "accepted_length_over_median_itl",
        "minimum_goodput_gain": schedule.minimum_goodput_gain,
        "ranges": [
            [item.start_batch, item.end_batch, item.k] for item in schedule.ranges
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-goodput-gain", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    profile = load_profile(args.profile)
    schedule = derive_schedule(
        profile, minimum_goodput_gain=args.minimum_goodput_gain
    )
    write_schedule(profile, schedule, args.output)
    print(f"profile_sha256={profile.source_sha256}")
    print(f"schedule={args.output.resolve()}")
    print(
        "ranges="
        + json.dumps(
            [
                [item.start_batch, item.end_batch, item.k]
                for item in schedule.ranges
            ],
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
