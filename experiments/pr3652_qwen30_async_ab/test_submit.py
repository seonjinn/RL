from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "experiments/pr3652_qwen30_async_ab/submit_oci_hsg.sh"


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


def test_bf16_async_contract() -> None:
    output = render("bf16")

    assert "mode=async_disaggregated_nccl_reshard" in output
    assert "rollout_precision=bfloat16" in output
    assert "refit_transport=nccl_reshard" in output
    assert "cuda_graphs=enabled" in output
    assert "reference_logprobs=enabled" in output


def test_mxfp8_uses_receiver_side_conversion() -> None:
    output = render("mxfp8")

    assert "rollout_precision=mxfp8" in output
    assert "quantization_scope=routed_experts_only" in output
    assert "refit_prequantize=false" in output
    assert "moe_backend=flashinfer_trtllm" in output


def test_optional_overrides_cannot_merge_with_next_argument() -> None:
    source = LAUNCHER.read_text()

    assert "${MXFP8_OVERRIDES}  loss_fn" not in source
    assert "${MXFP8_OVERRIDE_ARGS} \\\\" in source
