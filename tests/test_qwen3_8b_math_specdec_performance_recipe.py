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

import os
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECIPE = (
    PROJECT_ROOT
    / "examples/configs/recipes/llm/performance/grpo-qwen3-8b-2n4g-specdec.yaml"
)
LAUNCHER = (
    PROJECT_ROOT / "tests/test_suites/llm/performance/grpo-qwen3-8b-2n4g-specdec.sh"
)


def _resolved_config(overrides: list[str] | None = None) -> dict:
    register_omegaconf_resolvers()
    config = load_config(RECIPE)
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    resolved = OmegaConf.to_container(config, resolve=True)
    assert isinstance(resolved, dict)
    return resolved


def test_recipe_preserves_qwen3_8b_math_performance_contract() -> None:
    config = _resolved_config()

    assert config["grpo"]["num_prompts_per_step"] == 64
    assert config["grpo"]["num_generations_per_prompt"] == 32
    assert config["policy"]["model_name"] == "Qwen/Qwen3-8B"
    assert config["policy"]["tokenizer"]["name"] == "Qwen/Qwen3-8B"
    assert config["policy"]["generation"]["stop_token_ids"] is None
    assert config["data"]["train"]["dataset_name"] == "OpenMathInstruct-2"
    assert config["data"]["default"]["processor"] == "math_hf_data_processor"
    assert config["data"]["default"]["env_name"] == "math"
    assert config["env"]["math"]["math_verify_impl"] == "hf_math_verify"
    assert config["cluster"]["gpus_per_node"] == 4
    assert config["cluster"]["num_nodes"] == 2
    assert config["cluster"]["segment_size"] == 2

    generation = config["policy"]["generation"]
    assert generation["vllm_cfg"]["enforce_eager"] is False
    assert generation["vllm_kwargs"].get("speculative_config") is None


@pytest.mark.parametrize("method", ["dflash", "dspark"])
def test_recipe_accepts_supported_fixed_drafter_overrides(method: str) -> None:
    config = _resolved_config(
        [
            f"++policy.generation.vllm_kwargs.speculative_config.method={method}",
            "++policy.generation.vllm_kwargs.speculative_config.model=/models/draft",
            "++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=7",
            "++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1",
        ]
    )

    assert config["policy"]["generation"]["vllm_kwargs"]["speculative_config"] == {
        "method": method,
        "model": "/models/draft",
        "num_speculative_tokens": 7,
        "draft_tensor_parallel_size": 1,
    }


def test_performance_launcher_resolves_its_paired_recipe() -> None:
    env = {**os.environ, "TEST_DRYRUN": "1"}

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "TEST_DRYRUN mode" in result.stdout
