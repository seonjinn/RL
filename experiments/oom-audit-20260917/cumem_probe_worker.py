"""Read-only CuMem inventory for the isolated vLLM initialization probe."""

import json

import torch
from vllm.device_allocator.cumem import CuMemAllocator
from vllm.v1.worker.gpu_worker import Worker


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
    def determine_available_memory(self) -> int:
        log_accounting("before_profile")
        try:
            return super().determine_available_memory()
        finally:
            log_accounting("after_profile")
