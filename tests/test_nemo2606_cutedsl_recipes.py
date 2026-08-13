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


def test_feature_contract_runner_reads_transformer_engine_distribution_version() -> None:
    runner = (PROJECT_ROOT / "tools/run_nemo2606_feature_tests.sbatch").read_text()

    assert "te.__version__" not in runner
    assert 'version("transformer-engine")' in runner


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


@pytest.mark.parametrize("model", ("qwen3-30ba3b-4n4g", "qwen3-235b-16n4g"))
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


@pytest.mark.parametrize(
    ("model", "dispatcher_overrides"),
    (
        (
            "qwen3-30ba3b-4n4g",
            {
                "moe_token_dispatcher_type": "flex",
                "moe_flex_dispatcher_backend": "hybridep",
                "moe_hybridep_num_sms": 32,
                "moe_hybridep_num_sms_preprocessing": 32,
            },
        ),
        (
            "qwen3-235b-16n4g",
            {"moe_hybridep_num_sms_preprocessing": 32},
        ),
    ),
)
def test_a2a_matched_baseline_has_compatible_eager_settings(
    model: str, dispatcher_overrides: dict[str, Any]
) -> None:
    cutedsl_name = f"grpo-{model}-megatron-mxfp8-cutedsl.yaml"
    config = _load_raw(f"grpo-{model}-megatron-mxfp8-cutedsl-a2a-matched.yaml")

    assert config["defaults"] == f"./{cutedsl_name}"
    assert set(config) == {"defaults", "checkpointing", "policy", "logger"}
    assert config["checkpointing"]["enabled"] is False
    assert config["policy"]["megatron_cfg"] == {
        "activation_checkpointing": False,
        "defer_fp32_logits": False,
        "env_vars": {"CUDA_DEVICE_MAX_CONNECTIONS": "32"},
        "overlap_moe_expert_parallel_comm": False,
        "high_priority_a2a_comm_stream": False,
        "delay_wgrad_compute": False,
        **dispatcher_overrides,
    }


@pytest.mark.parametrize("model", ("qwen3-30ba3b-4n4g", "qwen3-235b-16n4g"))
def test_a2a_overlay_changes_only_the_overlap_bundle(model: str) -> None:
    matched_name = f"grpo-{model}-megatron-mxfp8-cutedsl-a2a-matched.yaml"
    config = _load_raw(f"grpo-{model}-megatron-mxfp8-cutedsl-a2a.yaml")

    assert config["defaults"] == f"./{matched_name}"
    assert set(config) == {"defaults", "checkpointing", "policy", "logger"}
    assert config["policy"]["megatron_cfg"] == {
        "overlap_moe_expert_parallel_comm": True,
        "high_priority_a2a_comm_stream": True,
        "delay_wgrad_compute": True,
    }


@pytest.mark.parametrize(
    ("model", "base_model", "nodes", "generation_nodes"),
    (
        ("qwen3-30ba3b-8n4g", "qwen3-30ba3b-4n4g", 8, 4),
        ("qwen3-235b-32n4g", "qwen3-235b-16n4g", 32, 16),
    ),
)
def test_full_cuda_graph_has_dependency_matched_eager_baseline(
    model: str,
    base_model: str,
    nodes: int,
    generation_nodes: int,
) -> None:
    matched_name = f"grpo-{model}-megatron-mxfp8-cutedsl-full-cg-matched.yaml"
    config = _load_raw(matched_name)

    assert config["defaults"] == (
        f"./grpo-{base_model}-megatron-mxfp8-cutedsl-a2a-matched.yaml"
    )
    assert config["grpo"]["skip_reference_policy_logprobs_calculation"] is True
    assert config["grpo"]["seq_logprob_error_threshold"] == 1.0e30
    assert config["loss_fn"]["reference_policy_kl_penalty"] == 0.0
    assert config["policy"]["dynamic_batching"]["enabled"] is False
    assert config["policy"]["sequence_packing"] == {
        "enabled": False,
        "fuse_loss": False,
    }
    megatron = config["policy"]["megatron_cfg"]
    assert megatron["cuda_graph_impl"] == "none"
    assert megatron["moe_expert_rank_capacity_factor"] == 1.5
    assert megatron["moe_paged_stash"] is True
    assert megatron["offload_modules"] == []
    assert config["policy"]["generation"]["colocated"] == {
        "enabled": False,
        "resources": {"num_nodes": generation_nodes, "gpus_per_node": 4},
    }
    assert config["cluster"]["num_nodes"] == nodes


@pytest.mark.parametrize("model", ("qwen3-30ba3b-8n4g", "qwen3-235b-32n4g"))
def test_full_cuda_graph_and_combined_overlays_are_single_factor(model: str) -> None:
    full_name = f"grpo-{model}-megatron-mxfp8-cutedsl-full-cg.yaml"
    full = _load_raw(full_name)
    assert full["policy"]["megatron_cfg"] == {"cuda_graph_impl": "full_iteration"}

    combined = _load_raw(f"grpo-{model}-megatron-mxfp8-cutedsl-full-cg-a2a.yaml")
    assert combined["defaults"] == f"./{full_name}"
    assert combined["policy"]["megatron_cfg"] == {
        "overlap_moe_expert_parallel_comm": True,
        "high_priority_a2a_comm_stream": True,
        "delay_wgrad_compute": True,
    }
