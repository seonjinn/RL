"""Read-only CuMem inventory for the isolated vLLM initialization probe."""

import json
import os

import torch
from vllm.device_allocator.cumem import CuMemAllocator
from vllm.v1.worker.gpu_worker import Worker
from vllm.utils.mem_utils import MemorySnapshot

from cumem_accounting import unmapped_idle_bytes


def log_accounting(label: str) -> None:
    allocator = CuMemAllocator.instance
    if allocator is None:
        print(f"CUMEM_DIAG {label}: allocator absent", flush=True)
        return
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    entries = dict(allocator.pointer_to_data)
    rows = []
    seen: set[int] = set()
    for tag, (pool, _pluggable) in list(allocator.allocator_and_pools.items()):
        if id(pool) in seen:
            continue
        seen.add(id(pool))
        segments = [s for s in pool.snapshot() if s["device"] == device]
        missing = [s for s in segments if s["address"] not in entries]
        idle = [s for s in missing if s["allocated_size"] == s["active_size"] == 0]
        rows.append(
            {
                "tag": tag,
                "segment_count": len(segments),
                "snapshot_reserved_bytes": sum(s["total_size"] for s in segments),
                "snapshot_allocated_bytes": sum(s["allocated_size"] for s in segments),
                "missing_segment_count": len(missing),
                "missing_idle_bytes": sum(s["total_size"] for s in idle),
                "missing_nonidle_bytes": sum(
                    s["total_size"]
                    for s in missing
                    if s["allocated_size"] != 0 or s["active_size"] != 0
                ),
            }
        )
    stats = torch.cuda.memory_stats(device)
    print(
        "CUMEM_DIAG "
        + json.dumps(
            {
                "label": label,
                "device": device,
                "physical_used_bytes": total - free,
                "allocated_bytes": stats["allocated_bytes.all.current"],
                "reserved_bytes": stats["reserved_bytes.all.current"],
                "pools": rows,
            },
            sort_keys=True,
        ),
        flush=True,
    )


class ProbeWorker(Worker):
    correct_accounting: bool = False

    def determine_available_memory(self) -> int:
        log_accounting("before_profile")
        original_measure = MemorySnapshot.measure

        def corrected_measure(snapshot: MemorySnapshot) -> None:
            original_measure(snapshot)
            allocator = CuMemAllocator.instance
            if allocator is None:
                return
            device = snapshot.device_.index
            if device is None:
                raise ValueError("Snapshot must identify an explicit GPU")
            segments = []
            seen: set[int] = set()
            for pool, _ in list(allocator.allocator_and_pools.values()):
                if id(pool) not in seen:
                    seen.add(id(pool))
                    segments.extend(pool.snapshot())
            correction = unmapped_idle_bytes(
                segments, set(allocator.pointer_to_data), device, snapshot.torch_memory
            )
            original_reserved = snapshot.torch_memory
            snapshot.torch_memory -= correction
            snapshot.non_torch_memory = snapshot.cuda_memory - snapshot.torch_memory
            print(
                "CUMEM_CORRECTION "
                + json.dumps(
                    {
                        "device": device,
                        "original_reserved": original_reserved,
                        "correction": correction,
                        "corrected_reserved": snapshot.torch_memory,
                        "corrected_non_torch": snapshot.non_torch_memory,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        if self.correct_accounting or os.environ.get("AUDIT_CORRECT_CUMEM") == "1":
            # Only during awake initialization in this isolated probe. Not a
            # general sleep/wake or multi-threaded allocator correction.
            MemorySnapshot.measure = corrected_measure
        try:
            return super().determine_available_memory()
        finally:
            MemorySnapshot.measure = original_measure
            log_accounting("after_profile")


class CorrectedProbeWorker(ProbeWorker):
    """Explicit worker selection for the full diagnostic, not a recipe default."""

    correct_accounting = True
