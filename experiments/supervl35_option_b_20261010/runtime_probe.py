"""Record installed worker dependencies and bounded input-media accessibility."""

import importlib.metadata
import json
import os
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse


def media_paths(value: object) -> set[str]:
    paths: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in ("image_url", "video_url", "url") and isinstance(child, str):
                if child.startswith("file://"):
                    paths.add(unquote(urlparse(child).path))
                elif child.startswith("/"):
                    paths.add(child)
            paths.update(media_paths(child))
    elif isinstance(value, list):
        for child in value:
            paths.update(media_paths(child))
    return paths


def main() -> None:
    versions = {}
    for package in (
        "torch",
        "vllm",
        "ray",
        "transformers",
        "transformer-engine",
        "flashinfer-python",
        "nvidia-cutlass-dsl",
        "nemo-lens",
        "torchcodec",
        "nemo-gym",
        "megatron-core",
        "megatron-bridge",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    report = {"python": sys.executable, "versions": versions}
    report["cpu_affinity_count"] = len(os.sched_getaffinity(0))
    if versions["torch"]:
        import torch

        report["cuda_devices"] = torch.cuda.device_count()
        report["cuda_tensor_sum"] = torch.ones(4, device="cuda").sum().item()
    sample = Path(sys.argv[2])
    paths: set[str] = set()
    for line in sample.read_text().splitlines()[:30]:
        try:
            paths.update(media_paths(json.loads(line)))
        except json.JSONDecodeError:
            continue
    report["media_readable"] = {
        path: os.access(path, os.R_OK) for path in sorted(paths)
    }
    output = Path(sys.argv[1]) / (Path(sys.executable).parents[1].name + ".json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "media_readable"}
        )
    )
    print(f"Media accessibility: {sum(report['media_readable'].values())}/{len(paths)}")


if __name__ == "__main__":
    main()
