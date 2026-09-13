"""Isolated GPU lifecycle checks for the CuMem retirement experiment."""

import gc
import subprocess
import sys

import torch
from vllm.device_allocator.cumem import CuMemAllocator


def check(case: str) -> None:
    torch.cuda.set_device(0)
    owner = CuMemAllocator.get_instance()
    with owner.use_memory_pool("weights"):
        weight = torch.full((1024 * 1024,), 3.0, device="cuda")
        temporary = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        del temporary
    gc.collect()
    torch.cuda.synchronize()
    print(f"{case}: pool retired, handles={len(owner.pointer_to_data)}", flush=True)
    if case == "asleep-free":
        owner.sleep(offload_tags=("weights",))
        del weight
        gc.collect()
        torch.cuda.empty_cache()
    else:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                output = weight * 2
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = weight * 2
        for value in (5, 7, 11):
            weight.fill_(value)
            owner.sleep(offload_tags=("weights",))
            owner.wake_up(tags=["weights"])
            graph.replay()
            torch.cuda.synchronize()
            assert bool(torch.all(output == value * 2))
        graph.reset()
        del graph, output, weight
        gc.collect()
        torch.cuda.empty_cache()
    assert not owner.pointer_to_data, list(owner.pointer_to_data)
    owner.close()
    owner.close()
    print(f"PASS {case}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        check(sys.argv[1])
    else:
        failed = []
        for case in ("graph-replay", "asleep-free"):
            result = subprocess.run(
                [sys.executable, __file__, case],
                timeout=120,
                capture_output=True,
                text=True,
            )
            print(result.stdout, end="", flush=True)
            print(result.stderr, end="", file=sys.stderr, flush=True)
            print(f"RESULT {case}: {result.returncode}", flush=True)
            if result.returncode or "CUDA Error:" in result.stdout + result.stderr:
                failed.append(case)
        sys.exit(bool(failed))
