#!/usr/bin/env python3
"""Write an atomic one-GPU vLLM 0.29 runtime provenance receipt."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parsed = parser.parse_args()

    import torch
    import vllm

    if vllm.__version__ != "0.29.0":
        raise ValueError(f"expected vLLM 0.29.0, found {vllm.__version__}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("smoke requires exactly one visible CUDA device")
    try:
        import flashinfer

        flashinfer_version = getattr(flashinfer, "__version__", "unknown")
    except ImportError as exc:
        raise ValueError("vLLM 0.29 image is missing FlashInfer") from exc

    receipt = {
        "status": "complete",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "host": platform.node(),
        "machine": platform.machine(),
        "image": parsed.image,
        "image_sha256": parsed.image_sha256,
        "source_commit": parsed.source_commit,
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "flashinfer_version": flashinfer_version,
        "cuda_available": True,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0),
    }
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = parsed.output.with_suffix(parsed.output.suffix + ".partial")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(parsed.output)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

