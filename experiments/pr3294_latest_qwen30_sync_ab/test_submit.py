from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "experiments/pr3294_latest_qwen30_sync_ab/submit_oci_hsg.sh"


def render(arm: str) -> str:
    env = os.environ | {"ACTION": "render", "ARM": arm, "MAX_STEPS": "20"}
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_bf16_render_uses_matched_sync_contract() -> None:
    output = render("bf16")

    assert "rollout_precision=bfloat16" in output
    assert "moe_backend=triton" in output
    assert "max_steps=20" in output
    assert "cuda_graphs=enabled" in output
    assert "reference_logprobs=enabled" in output


def test_mxfp8_render_enables_full_pr3294_path() -> None:
    output = render("mxfp8")

    assert "rollout_precision=mxfp8" in output
    assert "moe_backend=flashinfer_trtllm" in output
    assert "refit_prequantize=true" in output
    assert "persistent_ipc_buffers=true" in output
    assert "batched_moe_shuffle=true" in output
    assert "loader_route_cache=true" in output
