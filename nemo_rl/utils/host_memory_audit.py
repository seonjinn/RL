"""Opt-in, read-only host-memory accounting for the September OOM investigation.

This diagnostic does not copy tensor data, collect garbage, trim allocators,
reset peaks, or synchronize CUDA. It is not a general-purpose profiler.
"""

import json
import os
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StorageRecord:
    device: str
    address: int
    nbytes: int
    pinned: bool


def summarize_storages(
    groups: Mapping[str, Iterable[StorageRecord]],
) -> dict[str, dict[str, int]]:
    seen: set[tuple[str, int]] = set()
    result: dict[str, dict[str, int]] = {}
    for name, records in groups.items():
        counts = {
            "cpu_unique_bytes": 0,
            "cuda_unique_bytes": 0,
            "pinned_unique_bytes": 0,
            "shared_bytes": 0,
        }
        local_seen: set[tuple[str, int]] = set()
        for record in records:
            key = record.device, record.address
            if record.nbytes == 0 or key in local_seen:
                continue
            local_seen.add(key)
            if key in seen:
                counts["shared_bytes"] += record.nbytes
                continue
            seen.add(key)
            if record.device == "cpu":
                counts["cpu_unique_bytes"] += record.nbytes
                if record.pinned:
                    counts["pinned_unique_bytes"] += record.nbytes
            elif record.device.startswith("cuda"):
                counts["cuda_unique_bytes"] += record.nbytes
        result[name] = counts
    return result


def parse_kib_fields(text: str) -> dict[str, int]:
    fields = {}
    for line in text.splitlines():
        parts = line.split()
        if (
            len(parts) == 3
            and parts[0].endswith(":")
            and parts[1].isdigit()
            and parts[2] == "kB"
        ):
            fields[parts[0][:-1]] = int(parts[1]) * 1024
    return fields


def _tensor_records(tree: Any) -> Iterable[StorageRecord]:
    import torch

    if isinstance(tree, torch.Tensor):
        if tree.device.type not in ("cpu", "cuda"):
            return
        storage = tree.untyped_storage()
        yield StorageRecord(
            str(tree.device),
            storage.data_ptr(),
            storage.nbytes(),
            tree.is_pinned() if tree.device.type == "cpu" else False,
        )
    elif isinstance(tree, Mapping):
        for value in tree.values():
            yield from _tensor_records(value)
    elif isinstance(tree, (list, tuple)):
        for value in tree:
            yield from _tensor_records(value)


def audit_host_memory(worker: Any, phase: str, temporary: Any = None) -> None:
    if os.environ.get("NRL_HOST_MEMORY_AUDIT") != "1":
        return
    import torch

    record: dict[str, Any] = {
        "phase": phase,
        "pid": os.getpid(),
        "rank": getattr(worker, "rank", None),
        "timestamp_ns": time.time_ns(),
        "source": __file__,
    }
    groups: dict[str, Any] = {
        "reference": getattr(worker, "reference_state_dict", None)
    }
    model = getattr(worker, "model", None)
    backup = []
    for attr in ("buffers", "expert_parallel_buffers"):
        buffers = getattr(model, attr, None)
        if isinstance(buffers, (list, tuple)):
            backup.extend(getattr(buffer, "param_data_cpu", None) for buffer in buffers)
    groups["policy_pinned_backup"] = backup
    optimizer = getattr(worker, "optimizer", None)
    if optimizer is not None:
        try:
            from megatron.core.optimizer import ChainedOptimizer

            groups["optimizer_state"] = (
                optimizer.state
                if isinstance(optimizer, ChainedOptimizer)
                else optimizer._get_state()
            )
        except (ImportError, AttributeError, RuntimeError) as exc:
            record["optimizer_inventory_error"] = type(exc).__name__
    groups["temporary_reference_swap"] = temporary
    record["storages"] = summarize_storages(
        {name: _tensor_records(value) for name, value in groups.items()}
    )
    for name in ("smaps_rollup", "status"):
        try:
            record[name] = parse_kib_fields(Path(f"/proc/self/{name}").read_text())
        except OSError as exc:
            record[f"{name}_error"] = type(exc).__name__
    try:
        relative = next(
            line.split(":", 2)[2]
            for line in Path("/proc/self/cgroup").read_text().splitlines()
            if line.startswith("0::")
        )
        cgroup = Path("/sys/fs/cgroup") / relative.lstrip("/")
        record["cgroup"] = {
            name: (cgroup / name).read_text().strip()
            for name in (
                "memory.current",
                "memory.peak",
                "memory.max",
                "memory.events",
                "memory.stat",
            )
            if (cgroup / name).is_file()
        }
    except (OSError, StopIteration) as exc:
        record["cgroup_error"] = type(exc).__name__
    host_stats = getattr(torch.cuda.memory, "host_memory_stats", None)
    if callable(host_stats):
        try:
            record["host_allocator"] = host_stats()
        except (RuntimeError, TypeError) as exc:
            record["host_allocator_error"] = type(exc).__name__
    else:
        record["host_allocator"] = "unavailable"
    if torch.cuda.is_initialized():
        record["cuda_allocated_bytes"] = torch.cuda.memory_allocated()
        record["cuda_reserved_bytes"] = torch.cuda.memory_reserved()
    print("[NRL_HOST_MEMORY_AUDIT] " + json.dumps(record, sort_keys=True), flush=True)
