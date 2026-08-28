#!/usr/bin/env python3
"""Immutable experiment contract for the vLLM 0.28 Nemotron BF16 matrix."""

from __future__ import annotations

from typing import Any


_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 128, 512]
_METHOD_ORDER = [
    "baseline",
    "mtp_static_k1",
    "mtp_static_k2",
    "mtp_static_k3",
    "mtp_static_k4",
    "mtp_static_k5",
    "mtp_dynamic_max_k5",
]


def validate_dynamic_schedule(value: str, *, max_k: int) -> list[dict[str, int]]:
    """Parse a contiguous ``start:end:k`` schedule covering BS 1 through 512."""
    if max_k <= 0:
        raise ValueError("max_k must be positive")
    rows: list[dict[str, int]] = []
    expected_start = 1
    for raw_entry in value.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        try:
            start, end, k = (int(part) for part in entry.split(":"))
        except ValueError as exc:
            raise ValueError(
                f"invalid DynamicSD entry {entry!r}; expected start:end:k"
            ) from exc
        if not rows and start != 1:
            raise ValueError("the first DynamicSD range must start at batch size 1")
        if start != expected_start:
            raise ValueError(
                f"DynamicSD ranges must be contiguous; expected start={expected_start}"
            )
        if end < start:
            raise ValueError("DynamicSD range end must be >= start")
        if not 0 <= k <= max_k:
            raise ValueError(f"DynamicSD K must be between 0 and max K={max_k}")
        rows.append({"start": start, "end": end, "k": k})
        expected_start = end + 1
    if not rows:
        raise ValueError("DynamicSD schedule must not be empty")
    if rows[-1]["end"] != 512:
        raise ValueError("DynamicSD schedule must end at batch size 512")
    return rows


def build_contract_matrix() -> dict[str, Any]:
    """Return the reviewed model, workload, method, and runner contract."""
    return {
        "schema_version": 1,
        "runtime_release": {
            "vllm_version": "0.28.0",
            "vllm_branch": "release",
            "vllm_commit": "2cf0a69",
        },
        "models": [
            {
                "key": "super",
                "name": "NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
                "checkpoint_revision": (
                    "d51eab0d1f979ebc26b546e634a04f450d99158e"
                ),
                "runtime_topology": {
                    "tensor_parallel_size": 2,
                    "node_count": 1,
                    "data_parallel_size": 1,
                    "enable_expert_parallel": False,
                },
            },
            {
                "key": "ultra",
                "name": "NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
                "checkpoint_revision": (
                    "624ba927cfbef0427354998700de3d51173c8c04"
                ),
                "runtime_topology": {
                    "tensor_parallel_size": 8,
                    "node_count": 2,
                    "data_parallel_size": 1,
                    "enable_expert_parallel": True,
                },
            },
        ],
        "shapes": [
            {"key": "isl1k_osl10k", "isl": 1000, "osl": 10000},
            {"key": "isl10k_osl1k", "isl": 10000, "osl": 1000},
        ],
        "batch_sizes": list(_BATCH_SIZES),
        "method_order": list(_METHOD_ORDER),
        "dynamic_schedule": validate_dynamic_schedule(
            "1:4:5,5:16:3,17:64:2,65:128:1,129:512:0",
            max_k=5,
        ),
        "cuda_graph_modes": {
            "mrv1_gate": "PIECEWISE",
            "mrv2_canary": "FULL_AND_PIECEWISE",
        },
    }
