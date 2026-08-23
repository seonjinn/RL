from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "experiments/pr3652_qwen235_async_ab/submit_oci_hsg.sh"


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


def test_bf16_contract() -> None:
    output = render("bf16")

    assert "model=Qwen3-235B-A22B" in output
    assert "mode=async_disaggregated_nccl_reshard" in output
    assert "nodes=32" in output
    assert "generation_nodes=16" in output
    assert "rollout_precision=bfloat16" in output
    assert "cuda_graphs=enabled" in output
    assert "reference_logprobs=enabled" in output


def test_mxfp8_contract() -> None:
    output = render("mxfp8")

    assert "rollout_precision=mxfp8" in output
    assert "quantization_scope=routed_experts_only" in output
    assert "moe_backend=flashinfer_trtllm" in output
    assert "refit_prequantize=false" in output


def test_optional_settings_use_hydra_append() -> None:
    source = LAUNCHER.read_text()

    assert "++policy.generation.vllm_cfg.refit_prequantize=false" in source
    assert "${MXFP8_OVERRIDE_ARGS} \\\\" in source
