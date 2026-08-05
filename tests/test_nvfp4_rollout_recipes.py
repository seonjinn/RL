# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import pytest

from nemo_rl.models.generation.vllm.utils import resolve_generation_worker_cls
from nemo_rl.models.policy.utils import resolve_policy_worker_cls
from nemo_rl.utils.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERF_CONFIG_DIR = PROJECT_ROOT / "examples/configs/recipes/llm/performance"
PERF_SCRIPT_DIR = PROJECT_ROOT / "tests/test_suites/llm/performance"

POLICY_WORKER = (
    "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
)
VLLM_WORKER = "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"
QUANT_VLLM_WORKER = (
    "nemo_rl.modelopt.models.generation.vllm_quant_worker.VllmQuantGenerationWorker"
)
QWEN3_30BA3B_REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"

NVFP4_ROLLOUT_CASES = {
    "grpo-qwen3-30ba3b-4n4g-nvfp4-w4a16-rollout": (
        "examples/modelopt/quant_configs/nvfp4_experts_weightonly.yaml",
        False,
        "marlin",
    ),
    "grpo-qwen3-30ba3b-4n4g-nvfp4-w4a4-rollout": (
        "examples/modelopt/quant_configs/nvfp4_experts.yaml",
        True,
        "flashinfer_trtllm",
    ),
}


@pytest.mark.parametrize(
    (
        "recipe_name",
        "expected_quant_cfg",
        "has_calibration_path",
        "expected_moe_backend",
    ),
    [
        (recipe_name, expected_quant_cfg, has_calibration_path)
        for recipe_name, (
            expected_quant_cfg,
            has_calibration_path,
            expected_moe_backend,
        ) in NVFP4_ROLLOUT_CASES.items()
    ],
)
def test_nvfp4_rollout_recipe_contract(
    recipe_name: str,
    expected_quant_cfg: str,
    has_calibration_path: bool,
    expected_moe_backend: str,
) -> None:
    config = load_config(PERF_CONFIG_DIR / f"{recipe_name}.yaml")
    policy = config["policy"]
    generation = policy["generation"]

    assert policy["megatron_cfg"]["enabled"] is True
    assert policy["dtensor_cfg"]["enabled"] is False
    assert generation["backend"] == "vllm"
    assert policy["quant_cfg"] is None
    assert config["loss_fn"]["force_on_policy_ratio"] is False
    assert config["loss_fn"]["use_importance_sampling_correction"] is True
    assert generation["real_quant"] is True
    assert generation["quant_cfg"] == expected_quant_cfg
    assert generation["vllm_kwargs"]["moe_backend"] == expected_moe_backend
    assert generation["vllm_kwargs"]["revision"] == QWEN3_30BA3B_REVISION
    assert "*.shared_expert.*" in generation["real_quant_ignore"]
    assert "*.shared_experts.*" in generation["real_quant_ignore"]
    assert config["cluster"]["num_nodes"] == 4
    assert config["cluster"]["gpus_per_node"] == 4

    policy_worker_cls = (
        POLICY_WORKER
        if policy["megatron_cfg"]["enabled"]
        else "nemo_rl.models.policy.workers.dtensor_policy_worker.DTensorPolicyWorker"
    )
    generation_worker_cls = (
        VLLM_WORKER
        if generation["backend"] == "vllm"
        else "nemo_rl.models.generation.sglang.sglang_worker.SglangGenerationWorker"
    )
    assert resolve_policy_worker_cls(policy_worker_cls, policy) == policy_worker_cls
    assert (
        resolve_generation_worker_cls(generation_worker_cls, generation)
        == QUANT_VLLM_WORKER
    )

    if has_calibration_path:
        assert generation["real_quant_calibration_path"] is None


@pytest.mark.parametrize("recipe_name", NVFP4_ROLLOUT_CASES)
def test_nvfp4_rollout_smoke_script_contract(recipe_name: str) -> None:
    script = (PERF_SCRIPT_DIR / f"{recipe_name}.sh").read_text()

    assert "NUM_NODES=2" in script
    assert "GPUS_PER_NODE=8" in script
    assert "SEGMENT_SIZE=${SCHEDULER_SEGMENT_SIZE:-2}" in script
    assert "Legacy refit requires SCHEDULER_SEGMENT_SIZE=2" in script
    assert "NCCL-Reshard requires SCHEDULER_SEGMENT_SIZE=1" in script
    assert "WANDB_API_KEY must be exported" in script
    assert "grpo.max_num_steps=$MAX_STEPS" in script
    assert "checkpointing.enabled=false" in script
    assert 'len(data["train/loss"]) == 2' in script
    assert (
        'len(data["timing/train/prepare_for_generation/transfer_and_update_weights"]) == 2'
        in script
    )
