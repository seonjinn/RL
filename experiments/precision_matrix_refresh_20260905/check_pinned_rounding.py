"""Dedicated GPU probe; does not modify a live worker's allocator settings."""

import json
import os
import subprocess
import sys


PROBE = r"""
import gc
import json
import torch

torch.cuda.init()
before = torch.cuda.host_memory_stats()
buffers = [torch.empty(n * 1024 * 1024, dtype=torch.uint8, pin_memory=True)
           for n in (17, 129)]
allocated = torch.cuda.host_memory_stats()
for value in (7, 19, 31):
    for source in buffers:
        source.fill_(value)
        device = source.to('cuda', non_blocking=True)
        restored = device.to('cpu', non_blocking=True)
        torch.cuda.synchronize()
        assert torch.equal(source, restored)
del source, restored, device, buffers
gc.collect()
after = torch.cuda.host_memory_stats()
keys = ('allocated_bytes.current', 'active_bytes.current')
print(json.dumps({'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
                  'requested_bytes': 146 * 1024 * 1024, 'roundtrip_equal': True,
                  'allocation_delta': {k: allocated[k] - before[k] for k in keys},
                  'after_release': {k: after[k] for k in keys}}))
"""


def main() -> None:
    results = {}
    for name, config in (
        ("default", ""),
        ("exact_large", "pinned_max_round_threshold_mb:16"),
    ):
        env = os.environ.copy()
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        env.pop("PYTORCH_ALLOC_CONF", None)
        if config:
            env["PYTORCH_ALLOC_CONF"] = config
        child = subprocess.run(
            [sys.executable, "-c", PROBE], env=env, capture_output=True,
            text=True, timeout=180, check=True,
        )
        results[name] = json.loads(child.stdout)
        print(json.dumps({name: results[name]}), flush=True)
    assert (results["exact_large"]["allocation_delta"]["allocated_bytes.current"]
            < results["default"]["allocation_delta"]["allocated_bytes.current"])


if __name__ == "__main__":
    main()
