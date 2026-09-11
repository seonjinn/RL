"""Record the installed worker runtime before any M2N dependency changes."""

import importlib
import importlib.metadata
import json
import platform
import sys
import traceback
from typing import Any


def main() -> None:
    result: dict[str, Any] = {
        "python": sys.executable,
        "machine": platform.machine(),
        "sys_path": sys.path,
        "packages": {},
        "modules": {},
    }
    for name in (
        "torch",
        "vllm",
        "nccl4py",
        "nccl-extensions",
        "nvidia-nccl-cu12",
        "nvidia-nccl-cu13",
    ):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    for name in ("torch", "nccl", "nccl.m2n"):
        try:
            module = importlib.import_module(name)
            entry = {"file": str(module.__file__)}
            result["modules"][name] = entry
            if name == "torch":
                entry["cuda"] = module.version.cuda
                entry["nccl"] = module.cuda.nccl.version()
                entry["gpu"] = module.cuda.get_device_name()
            elif name == "nccl":
                entry["loaded_versions"] = str(module.get_version())
            else:
                entry["reshard"] = callable(getattr(module, "reshard", None))
            result["modules"][name] = entry
        except Exception:
            result["modules"].setdefault(name, {})["error"] = traceback.format_exc()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
