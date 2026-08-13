# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
from typing import Any

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERFORMANCE_RECIPES = PROJECT_ROOT / "examples/configs/recipes/llm/performance"


def _load_raw(name: str) -> dict[str, Any]:
    value = yaml.safe_load((PERFORMANCE_RECIPES / name).read_text())
    assert isinstance(value, dict)
    return value


@pytest.mark.parametrize(
    ("model", "base_recipe"),
    (
        ("qwen3-30ba3b-4n4g", "./grpo-qwen3-30ba3b-4n4g.yaml"),
        ("qwen3-235b-16n4g", "./grpo-qwen3-235b-16n4g.yaml"),
    ),
)
def test_mxfp8_baseline_is_dependency_matched(model: str, base_recipe: str) -> None:
    baseline_name = f"grpo-{model}-megatron-mxfp8.yaml"
    config = _load_raw(baseline_name)

    assert config["defaults"] == base_recipe
    assert set(config) == {"defaults", "checkpointing", "policy", "logger"}
    megatron = config["policy"]["megatron_cfg"]
    assert megatron["moe_router_dtype"] == "fp32"
    assert megatron["cuda_graph_impl"] == "none"
    assert megatron["fp8_cfg"] == {
        "enabled": True,
        "fp8": "e4m3",
        "fp8_recipe": "mxfp8",
        "fp8_param": False,
    }
    assert megatron["model_overrides"] == {
        "use_transformer_engine_op_fuser": False,
        "moe_mlp_glu_interleave_size": None,
    }
    assert megatron["env_vars"]["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] == "0"


@pytest.mark.parametrize(
    "model", ("qwen3-30ba3b-4n4g", "qwen3-235b-16n4g")
)
def test_cutedsl_overlay_changes_only_the_fused_grouped_mlp_bundle(model: str) -> None:
    baseline_name = f"grpo-{model}-megatron-mxfp8.yaml"
    config = _load_raw(f"grpo-{model}-megatron-mxfp8-cutedsl.yaml")

    assert config["defaults"] == f"./{baseline_name}"
    assert set(config) == {"defaults", "checkpointing", "policy", "logger"}
    assert set(config["policy"]) == {"megatron_cfg"}
    megatron = config["policy"]["megatron_cfg"]
    assert megatron == {
        "model_overrides": {
            "use_transformer_engine_op_fuser": True,
            "moe_mlp_glu_interleave_size": 32,
        },
        "env_vars": {"NVTE_CUTEDSL_FUSED_GROUPED_MLP": "1"},
    }

