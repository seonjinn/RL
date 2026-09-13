"""Isolated GB200 diagnostic; malloc_trim is not a production offload fix."""

import argparse
import ctypes
import gc
import json
import os
import resource
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from probe_optimizer_cpu_copy import load_move_optimizer


def memory() -> dict[str, int]:
    result = {}
    for filename, keys in (
        ("/proc/self/status", {"VmRSS", "RssAnon", "VmHWM"}),
        ("/proc/meminfo", {"MemAvailable", "Mlocked", "Unevictable"}),
    ):
        for line in Path(filename).read_text().splitlines():
            name, _, value = line.partition(":")
            if name in keys:
                result[name + "_bytes"] = int(value.split()[0]) * 1024
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result.update(minor_faults=usage.ru_minflt, major_faults=usage.ru_majflt)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trim", action="store_true")
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--tensors", type=int, default=256)
    parser.add_argument("--tensor-mib", type=int, default=48)
    parser.add_argument("--pinned-mib", type=int, default=16384)
    args = parser.parse_args()
    if min(args.rounds, args.tensors, args.tensor_mib) <= 0 or args.pinned_mib < 0:
        parser.error("sizes and rounds must be positive; pinned-mib may be zero")
    rank = int(os.environ.get("SLURM_LOCALID", "0"))
    torch.cuda.set_device(rank)
    if "GB200" not in torch.cuda.get_device_name():
        raise RuntimeError("This diagnostic requires GB200")
    os.environ["NRL_HOST_STORAGE_DIAGNOSTICS"] = "1"
    trim = None
    if args.trim:
        trim = ctypes.CDLL(None).malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
    # Uneven pinned blocks model rounding, not a complete vLLM sleep lifecycle.
    pinned = []
    remaining = args.pinned_mib
    while remaining:
        size = min(65, remaining)
        pinned.append(torch.empty(size * 1024**2, dtype=torch.uint8, pin_memory=True))
        pinned[-1].fill_(19)
        remaining -= size
    elements = args.tensor_mib * 1024**2 // 4
    states = {index: {"exp_avg_sq": torch.full((elements,), float(index % 29), device="cuda")}
              for index in range(args.tensors)}
    worker = SimpleNamespace(optimizer=SimpleNamespace(_get_state=lambda: states))
    move = load_move_optimizer()
    print(json.dumps({"event": "config", "rank": rank, "torch": torch.__version__,
                      "args": vars(args), "memory": memory()}), flush=True)
    for cycle in range(args.rounds):
        torch.cuda.synchronize()
        before = memory()
        start = time.monotonic()
        move(worker, "cpu")
        offload_s = time.monotonic() - start
        offloaded = memory()
        start = time.monotonic()
        move(worker, "cuda")
        torch.cuda.synchronize()
        onload_s = time.monotonic() - start
        onloaded = memory()
        gc.collect()
        start = time.monotonic()
        trim_result = trim(0) if trim is not None else None
        trim_s = time.monotonic() - start
        returned = memory()
        for index, state in states.items():
            value = state["exp_avg_sq"]
            expected = float(index % 29 + cycle)
            if not torch.all(value == expected).item():
                raise AssertionError(f"roundtrip mismatch rank={rank} cycle={cycle} tensor={index}")
            value.add_(1)
        del value
        print(json.dumps({"event": "cycle", "rank": rank, "cycle": cycle,
                          "offload_s": offload_s, "onload_s": onload_s,
                          "trim_s": trim_s, "trim_result": trim_result,
                          "before": before, "offloaded": offloaded,
                          "onloaded": onloaded, "returned": returned,
                          "exact_values": True}), flush=True)
    assert all(torch.all(block == 19).item() for block in pinned)
    print(json.dumps({"event": "passed", "rank": rank, "rounds": args.rounds}), flush=True)


if __name__ == "__main__":
    main()
