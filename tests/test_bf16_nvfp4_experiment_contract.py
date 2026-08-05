# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERF_SCRIPT_DIR = PROJECT_ROOT / "tests/test_suites/llm/performance"
PERF_CONFIG_DIR = PROJECT_ROOT / "examples/configs/recipes/llm/performance"


def test_bf16_control_recipe_contract() -> None:
    recipe = (PERF_CONFIG_DIR / "grpo-qwen3-30ba3b-4n4g-bf16-control.yaml").read_text()

    assert "defaults: ./grpo-qwen3-30ba3b-4n4g.yaml" in recipe
    assert "force_on_policy_ratio: false" in recipe
    assert "use_importance_sampling_correction: true" in recipe
    assert "real_quant: false" in recipe
    assert "quant_cfg: null" in recipe


def test_bf16_control_smoke_script_contract() -> None:
    script = (PERF_SCRIPT_DIR / "grpo-qwen3-30ba3b-4n4g-bf16-control.sh").read_text()

    assert "NUM_NODES=2" in script
    assert "GPUS_PER_NODE=8" in script
    assert "SEGMENT_SIZE=2" in script
    assert "MAX_STEPS=2" in script
    assert "checkpointing.enabled=false" in script
    assert 'len(data["train/loss"]) == 2' in script
    assert (
        'len(data["timing/train/prepare_for_generation/transfer_and_update_weights"]) == 2'
        in script
    )
    assert "VllmQuantInternalWorkerExtension" in script


def test_gcp_nrt_submitter_supports_bf16_control() -> None:
    submitter = (
        PROJECT_ROOT / "experiments/bf16-nvfp4-rollout/submit_gcp_nrt.sh"
    ).read_text()

    assert "bf16)" in submitter
    assert "grpo-qwen3-30ba3b-4n4g-bf16-control.sh" in submitter


def test_gcp_nrt_submitter_forwards_max_steps() -> None:
    submitter = (
        PROJECT_ROOT / "experiments/bf16-nvfp4-rollout/submit_gcp_nrt.sh"
    ).read_text()

    assert "MAX_STEPS=${MAX_STEPS:-2}" in submitter
    assert "export MAX_STEPS=${MAX_STEPS}" in submitter
    assert "max_steps=${MAX_STEPS}" in submitter
