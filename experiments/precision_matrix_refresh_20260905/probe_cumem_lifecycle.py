"""Isolated GPU lifecycle checks for the CuMem retirement experiment."""

import gc
import importlib.util
import os
import subprocess
import sys

import torch

if extension := os.environ.get("NRL_CUMEM_EXTENSION"):
    import vllm

    assert "vllm.cumem_allocator" not in sys.modules
    spec = importlib.util.spec_from_file_location("vllm.cumem_allocator", extension)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["vllm.cumem_allocator"] = module
    print(f"Experimental allocator extension: {module.__file__}", flush=True)

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
    if case == "live-close":
        owner.close()
        assert bool(torch.all(weight == 3))
        del weight
        gc.collect()
        torch.cuda.empty_cache()
    elif case == "exception-reuse":
        try:
            with owner.use_memory_pool("interrupted"):
                other = torch.full((1024,), 7.0, device="cuda")
                raise ValueError("intentional pool interruption")
        except ValueError as error:
            assert str(error) == "intentional pool interruption"
        assert owner.current_tag == owner.default_tag
        assert bool(torch.all(other == 7))
        with owner.use_memory_pool("subsequent"):
            replacement = torch.full((1024,), 11.0, device="cuda")
        owner.sleep(offload_tags=("weights", "interrupted", "subsequent"))
        owner.wake_up()
        assert bool(torch.all(weight == 3))
        assert bool(torch.all(other == 7))
        assert bool(torch.all(replacement == 11))
        del weight, other, replacement
        gc.collect()
        torch.cuda.empty_cache()
    elif case == "asleep-free":
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
        for case in ("graph-replay", "asleep-free", "live-close", "exception-reuse"):
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
