"""Run inside each image-native worker interpreter on one allocated GPU node."""

import importlib
import importlib.metadata
import json
import sys
from pathlib import Path

import torch

role = sys.argv[1]
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == 4, torch.cuda.device_count()
for device in range(4):
    value = torch.ones(8, device=f"cuda:{device}")
    assert value.sum().item() == 8
    del value
modules = ["nemo_rl", "ray", "torch"]
if role == "policy":
    modules += [
        "megatron.core",
        "megatron.bridge",
        "transformer_engine.pytorch",
        "deep_ep",
    ]
if role == "generation":
    modules += ["vllm", "flashinfer"]
    from vllm.platforms import current_platform

    assert current_platform.is_cuda()
print(
    json.dumps(
        {
            "role": role,
            "python": sys.executable,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "modules": {
                name: str(importlib.import_module(name).__file__) for name in modules
            },
        },
        sort_keys=True,
    )
)

if role == "policy":
    # Load only the diagnostic module from the host checkout, leaving framework
    # imports image-native for this gate. Training verifies host-source imports separately.
    import importlib.util

    source = Path(__file__).resolve().parents[2] / "nemo_rl/utils/host_memory_audit.py"
    spec = importlib.util.spec_from_file_location("host_memory_audit", source)
    audit = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = audit
    spec.loader.exec_module(audit)
    cpu = torch.zeros(1024, pin_memory=True)
    records = list(audit._tensor_records([cpu, cpu[:1], {"alias": cpu}]))
    counts = audit.summarize_storages({"cpu": records})["cpu"]
    assert counts["cpu_unique_bytes"] == 4096, counts
    assert counts["pinned_unique_bytes"] == 4096, counts
    import os
    from types import SimpleNamespace

    os.environ["NRL_HOST_MEMORY_AUDIT"] = "1"
    audit.audit_host_memory(
        SimpleNamespace(rank=0, reference_state_dict={"test": cpu}), "smoke"
    )
    print("HOST_MEMORY_AUDIT_REAL_TENSOR_PASS", flush=True)
