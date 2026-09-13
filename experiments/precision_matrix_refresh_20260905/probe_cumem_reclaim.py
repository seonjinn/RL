"""Exercise allocator-owned pool retirement without manual unmapping."""

import gc
import json

import torch

from vllm.device_allocator.cumem import (
    CuMemAllocator,
    use_memory_pool_with_allocator,
)


def snapshot(label: str) -> dict[str, int | str]:
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    result = {
        "label": label,
        "physical": total - free,
        "reserved": torch.cuda.memory_reserved(),
        "allocated": torch.cuda.memory_allocated(),
    }
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    torch.cuda.set_device(0)
    owner = CuMemAllocator.get_instance()
    wrappers = []
    live_tensors = []
    for iteration in range(3):
        owner.current_tag = "weights"
        with use_memory_pool_with_allocator(
            owner.python_malloc_callback, owner.python_free_callback
        ) as data:
            live = torch.full(
                (64 * 1024 * 1024,), iteration + 1, dtype=torch.uint8, device="cuda"
            )
            temporary = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
            del temporary
        before = snapshot(f"before-retirement-{iteration}")
        wrappers.append(data[1])
        del data
        gc.collect()
        after = snapshot(f"after-retirement-{iteration}")
        assert before["reserved"] - after["reserved"] >= 256 * 1024 * 1024
        assert before["physical"] - after["physical"] >= 250 * 1024 * 1024
        assert bool(torch.all(live == iteration + 1))
        live_tensors.append(live)
        owner.sleep(offload_tags=("weights",))
        owner.wake_up(tags=["weights"])
        for index, tensor in enumerate(live_tensors):
            assert bool(torch.all(tensor == index + 1))
        del tensor
    del live
    live_tensors.clear()
    gc.collect()
    torch.cuda.empty_cache()
    assert not owner.pointer_to_data, owner.pointer_to_data.keys()
    wrappers.clear()
    gc.collect()
    snapshot("all-freed")
    print(
        "PASS: three retire/sleep/wake cycles preserve live values and free all handles",
        flush=True,
    )


if __name__ == "__main__":
    main()
