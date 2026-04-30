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

import hashlib
import json
import math
import os
import re
import threading
import time
from collections.abc import Mapping
from typing import Any

import torch

_DEBUG_ENV = "NRL_NEMOTRON_VL_DEBUG"
_DEBUG_DIR_ENV = "NRL_NEMOTRON_VL_DEBUG_DIR"
_DEBUG_FILE_ENV = "NRL_NEMOTRON_VL_DEBUG_FILE"
_DUMP_TENSORS_ENV = "NRL_NEMOTRON_VL_DEBUG_DUMP_TENSORS"
_RUN_LABEL_ENV = "NRL_NEMOTRON_VL_RUN_LABEL"
_SAMPLE_ID_ENV = "NRL_NEMOTRON_VL_FIXTURE_SAMPLE_ID"

_FILE_LOCK = threading.Lock()
_WRITE_ERROR_REPORTED = False
_SANITIZE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def debug_enabled() -> bool:
    return os.environ.get(_DEBUG_ENV, "0") == "1"


def dump_tensors_enabled() -> bool:
    return os.environ.get(_DUMP_TENSORS_ENV, "0") == "1"


def get_run_label(default: str | None = None) -> str | None:
    return os.environ.get(_RUN_LABEL_ENV, default)


def get_sample_id(default: str | None = None) -> str | None:
    return os.environ.get(_SAMPLE_ID_ENV, default)


def stable_hash(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        raw = value
    elif isinstance(value, str):
        raw = value.encode("utf-8")
    elif torch.is_tensor(value):
        raw = _tensor_hash_bytes(value)
    else:
        raw = json.dumps(
            _json_safe(value),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def tensor_summary(name: str, tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None

    detached = tensor.detach()
    cpu_tensor = detached.cpu()
    flat = cpu_tensor.reshape(-1)
    sample = _sample_tensor(flat)
    stats_tensor = _stats_tensor(sample)

    summary: dict[str, Any] = {
        "name": name,
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": int(detached.numel()),
        "sample_numel": int(sample.numel()),
        "sample_hash": stable_hash(sample),
        "first_values": _tensor_preview_values(sample),
    }

    if sample.numel() == 0:
        summary.update(
            {
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "sum": None,
                "l2": None,
            }
        )
    else:
        summary.update(
            {
                "min": float(stats_tensor.min().item()),
                "max": float(stats_tensor.max().item()),
                "mean": float(stats_tensor.mean().item()),
                "std": float(stats_tensor.std(unbiased=False).item()),
                "sum": float(stats_tensor.sum().item()),
                "l2": float(torch.linalg.vector_norm(stats_tensor).item()),
            }
        )

    if dump_tensors_enabled():
        summary["tensor_values"] = _tensor_values_for_json(cpu_tensor)

    return summary


def write_stage(stage: str, payload: Mapping[str, Any] | None = None) -> None:
    if not debug_enabled():
        return

    debug_dir = os.path.abspath(
        os.environ.get(_DEBUG_DIR_ENV, "/tmp/nrl_nemotron_vl_debug")
    )
    record: dict[str, Any] = {
        "stage": stage,
        "timestamp_ms": int(time.time() * 1000),
        "pid": os.getpid(),
        "run_label": get_run_label("unknown"),
        "repo": get_run_label("unknown"),
        "sample_id": get_sample_id(),
    }

    if payload is not None:
        record.update(_json_safe(payload))

    try:
        os.makedirs(debug_dir, exist_ok=True)
        output_path = _debug_output_path(debug_dir)
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        with _FILE_LOCK:
            with open(output_path, "a", encoding="utf-8") as handle:
                handle.write(line)
                handle.write("\n")
    except Exception as exc:
        _report_write_error_once(exc)


def _debug_output_path(debug_dir: str) -> str:
    explicit = os.environ.get(_DEBUG_FILE_ENV)
    if explicit:
        if os.path.isabs(explicit):
            return explicit
        return os.path.join(debug_dir, explicit)

    run_label = get_run_label("run") or "run"
    safe_run_label = _SANITIZE_FILENAME_RE.sub("_", run_label).strip("._")
    if not safe_run_label:
        safe_run_label = "run"
    return os.path.join(debug_dir, f"{safe_run_label}-pid{os.getpid()}.jsonl")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value

    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value

    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)

    if isinstance(value, torch.Size):
        return list(value)

    if isinstance(value, type):
        return value.__name__

    if torch.is_tensor(value):
        return tensor_summary("tensor", value)

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    if hasattr(value, "tolist") and not isinstance(value, (bytes, bytearray)):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass

    return str(value)


def _sample_tensor(tensor: torch.Tensor, max_values: int = 256) -> torch.Tensor:
    if tensor.numel() <= max_values:
        return tensor.contiguous()

    indices = torch.linspace(
        0,
        tensor.numel() - 1,
        steps=max_values,
        device=tensor.device,
        dtype=torch.float64,
    ).round().to(torch.long)
    return tensor.index_select(0, indices).contiguous()


def _stats_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.bool:
        return tensor.to(torch.float32)
    if torch.is_floating_point(tensor):
        return tensor.to(torch.float32)
    return tensor.to(torch.float32)


def _tensor_preview_values(tensor: torch.Tensor, limit: int = 8) -> list[Any]:
    preview = tensor[:limit]
    if preview.dtype == torch.bool:
        return [bool(item) for item in preview.tolist()]
    if torch.is_floating_point(preview):
        return [float(item) for item in preview.to(torch.float32).tolist()]
    return [int(item) for item in preview.to(torch.int64).tolist()]


def _tensor_values_for_json(tensor: torch.Tensor) -> Any:
    cpu_tensor = tensor.detach().cpu()
    if cpu_tensor.dtype == torch.bool:
        return cpu_tensor.tolist()
    if torch.is_floating_point(cpu_tensor):
        return cpu_tensor.to(torch.float32).tolist()
    return cpu_tensor.to(torch.int64).tolist()


def _tensor_hash_bytes(tensor: torch.Tensor) -> bytes:
    sample = _sample_tensor(tensor.detach().cpu().reshape(-1))
    if sample.dtype == torch.bool:
        sample = sample.to(torch.uint8)
    elif torch.is_floating_point(sample):
        sample = sample.to(torch.float32)
    else:
        sample = sample.to(torch.int64)
    return sample.contiguous().numpy().tobytes()


def _report_write_error_once(exc: Exception) -> None:
    global _WRITE_ERROR_REPORTED
    if _WRITE_ERROR_REPORTED:
        return
    _WRITE_ERROR_REPORTED = True
    print(f"[NRL_NEMOTRON_VL_DEBUG] failed to write debug record: {exc}", flush=True)
