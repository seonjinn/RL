"""Record the installed worker runtime before any M2N dependency changes."""

import argparse
import importlib
import importlib.metadata
import json
import platform
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-native", action="store_true")
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
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
    for name in ("torch", "nccl.core", "nccl.m2n"):
        try:
            module = importlib.import_module(name)
            entry = {"file": str(module.__file__)}
            result["modules"][name] = entry
            if name == "torch":
                entry["cuda"] = module.version.cuda
                entry["nccl_build_version"] = module.cuda.nccl.version()
                entry["gpu"] = module.cuda.get_device_name()
            elif name == "nccl.core":
                loaded = module.get_version().libnccl
                entry["loaded_version"] = str(loaded.version)
                entry["loaded_path"] = str(loaded.path)
            elif name == "nccl.m2n":
                entry["reshard"] = callable(getattr(module, "reshard", None))
            result["modules"][name] = entry
        except Exception:
            result["modules"].setdefault(name, {})["error"] = traceback.format_exc()
    result["nccl_mappings"] = sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "/libnccl.so" in line
        }
    )
    print(json.dumps(result, indent=2), flush=True)
    if args.require_native:
        assert args.baseline is not None
        baseline = json.loads(args.baseline.read_text())
        for name in ("torch", "vllm"):
            assert result["packages"][name] == baseline["packages"][name], name
        assert (
            result["modules"]["torch"]["file"] == baseline["modules"]["torch"]["file"]
        )
        assert result["modules"]["nccl.core"]["loaded_version"] == "2.30.7"
        assert result["modules"]["nccl.m2n"]["reshard"]
        overlay = Path(os.environ["M2N_BINDINGS_ROOT"]).resolve()
        for name in ("nccl.core", "nccl.m2n"):
            assert (
                Path(result["modules"][name]["file"]).resolve().is_relative_to(overlay)
            ), name
        assert (
            Path(result["modules"]["nccl.core"]["loaded_path"])
            .resolve()
            .is_relative_to(overlay)
        )
        assert result["nccl_mappings"]
        assert all(
            Path(path).resolve().is_relative_to(overlay)
            for path in result["nccl_mappings"]
        )


if __name__ == "__main__":
    main()
