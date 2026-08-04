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

POLICY_WORKER = (
    "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
)
VLLM_WORKER = "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"
QUANT_VLLM_WORKER = (
    "nemo_rl.modelopt.models.generation.vllm_quant_worker.VllmQuantGenerationWorker"
)

NVFP4_ROLLOUT_CASES = {
    "grpo-qwen3-30ba3b-4n4g-nvfp4-w4a16-rollout": (
        "examples/modelopt/quant_configs/nvfp4_experts_weightonly.yaml",
        False,
    ),
    "grpo-qwen3-30ba3b-4n4g-nvfp4-w4a4-rollout": (
        "examples/modelopt/quant_configs/nvfp4_experts.yaml",
        True,
    ),
}


@pytest.mark.parametrize(
    ("recipe_name", "expected_quant_cfg", "has_calibration_path"),
    [
        (recipe_name, expected_quant_cfg, has_calibration_path)
        for recipe_name, (
            expected_quant_cfg,
            has_calibration_path,
        ) in NVFP4_ROLLOUT_CASES.items()
    ],
)
def test_nvfp4_rollout_recipe_contract(
    recipe_name: str, expected_quant_cfg: str, has_calibration_path: bool
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
