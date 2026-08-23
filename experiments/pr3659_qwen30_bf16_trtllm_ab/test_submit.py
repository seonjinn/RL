from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "experiments/pr3659_qwen30_bf16_trtllm_ab/submit_oci_hsg.sh"


def render(backend: str) -> str:
    env = os.environ | {
        "ACTION": "render",
        "BACKEND": backend,
        "MAX_STEPS": "20",
    }
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_triton_contract() -> None:
    output = render("triton")

    assert "training_precision=bf16" in output
    assert "rollout_precision=bf16" in output
    assert "moe_backend=triton" in output
    assert "refit_transport=nccl_reshard" in output
    assert "max_steps=20" in output
    assert "cuda_graphs=enabled" in output


def test_trtllm_contract() -> None:
    output = render("flashinfer_trtllm")

    assert "moe_backend=flashinfer_trtllm" in output
    assert "native_layerwise_refit=enabled" in output
    assert "reference_logprobs=enabled" in output
