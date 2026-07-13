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
from typing import Any, cast

import yaml
from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERFORMANCE_RECIPES = PROJECT_ROOT / "examples/configs/recipes/llm/performance"
BASE_RECIPE = PERFORMANCE_RECIPES / "grpo-qwen3-235b-16n4g.yaml"
RECIPE = PERFORMANCE_RECIPES / "grpo-qwen3-235b-16n4g-megatron-mxfp8-cutedsl.yaml"
SUITE_SCRIPT = (
    PROJECT_ROOT
    / "tests/test_suites/llm/performance"
    / "grpo-qwen3-235b-16n4g-megatron-mxfp8-cutedsl.sh"
)
GB200_SUITE = PROJECT_ROOT / "tests/test_suites/performance_gb200.txt"

register_omegaconf_resolvers()


def _load_resolved(path: Path) -> dict[str, Any]:
    config = OmegaConf.to_container(load_config(path), resolve=True)
    assert isinstance(config, dict)
    return cast(dict[str, Any], config)


def _difference_paths(left: object, right: object, path: str = "") -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences: set[str] = set()
        for key in set(left) | set(right):
            child_path = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                differences.add(child_path)
            else:
                differences.update(_difference_paths(left[key], right[key], child_path))
        return differences
    return set() if left == right else {path}


def test_qwen3_235b_cutedsl_overlay_is_policy_only() -> None:
    raw = yaml.safe_load(RECIPE.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)

    assert raw["defaults"] == "./grpo-qwen3-235b-16n4g.yaml"
    assert set(raw) == {"defaults", "checkpointing", "policy", "logger"}
    assert set(raw["policy"]) == {"megatron_cfg"}
    assert "generation" not in raw["policy"]
    assert (
        raw["policy"]["megatron_cfg"]["env_vars"]["NVTE_CUTEDSL_FUSED_GROUPED_MLP"]
        == "1"
    )


def test_qwen3_235b_cutedsl_recipe_is_registered_for_performance() -> None:
    assert SUITE_SCRIPT.is_file()
    completed = subprocess.run(
        ["bash", str(SUITE_SCRIPT)],
        cwd=PROJECT_ROOT,
        env={**os.environ, "TEST_DRYRUN": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    suite_entry = (
        "tests/test_suites/llm/performance/"
        "grpo-qwen3-235b-16n4g-megatron-mxfp8-cutedsl.sh"
    )
    entries = GB200_SUITE.read_text(encoding="utf-8").splitlines()
    assert entries.count(suite_entry) == 1


def test_qwen3_235b_policy_mxfp8_cutedsl_contract() -> None:
    config = _load_resolved(RECIPE)
    policy = config["policy"]
    assert isinstance(policy, dict)
    megatron = policy["megatron_cfg"]
    generation = policy["generation"]
    assert isinstance(megatron, dict)
    assert isinstance(generation, dict)

    cluster = config["cluster"]
    assert isinstance(cluster, dict)
    assert {
        key: cluster[key] for key in ("num_nodes", "gpus_per_node", "segment_size")
    } == {"num_nodes": 16, "gpus_per_node": 4, "segment_size": 16}
    assert policy["model_name"] == "Qwen/Qwen3-235B-A22B"
    assert policy["precision"] == "bfloat16"
    assert [
        megatron[key]
        for key in (
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
            "expert_model_parallel_size",
            "expert_tensor_parallel_size",
        )
    ] == [2, 4, 2, 16, 1]
    assert megatron["moe_grouped_gemm"] is True
    assert policy["train_global_batch_size"] == 512
    assert policy["train_micro_batch_size"] == 1
    assert policy["logprob_batch_size"] == 1
    assert policy["max_total_sequence_length"] == 8192
    assert policy["sequence_packing"]["enabled"] is True
    assert config["grpo"]["num_prompts_per_step"] == 16
    assert config["grpo"]["num_generations_per_prompt"] == 32

    vllm = generation["vllm_cfg"]
    assert generation["backend"] == "vllm"
    assert generation["colocated"]["enabled"] is True
    assert vllm["async_engine"] is True
    assert vllm["tensor_parallel_size"] == 8
    assert vllm["precision"] == "bfloat16"
    assert vllm["gpu_memory_utilization"] == 0.4
    assert generation["vllm_kwargs"]["moe_backend"] == "triton"
    assert "is_mx" not in vllm
    assert "quantization_ignored_layer_kws" not in vllm

    assert megatron["moe_router_dtype"] == "fp32"
    assert megatron["use_transformer_engine_op_fuser"] is True
    assert megatron["moe_mlp_glu_interleave_size"] == 32
    assert megatron["fp8_cfg"] == {
        **megatron["fp8_cfg"],
        "enabled": True,
        "fp8": "e4m3",
        "fp8_recipe": "mxfp8",
        "fp8_param": False,
    }
    assert megatron["env_vars"]["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] == "1"
    assert megatron["cuda_graph_impl"] == "none"
    assert megatron["overlap_moe_expert_parallel_comm"] is False
    assert megatron["high_priority_a2a_comm_stream"] is False
    assert megatron["delay_wgrad_compute"] is False


def test_qwen3_235b_cutedsl_overlay_preserves_bf16_rollout_and_base() -> None:
    base_before = _load_resolved(BASE_RECIPE)
    overlay = _load_resolved(RECIPE)
    base_after = _load_resolved(BASE_RECIPE)

    assert base_before == base_after
    assert overlay["policy"]["generation"] == base_before["policy"]["generation"]
    assert _difference_paths(base_before, overlay) == {
        "checkpointing.checkpoint_dir",
        "logger.log_dir",
        "logger.wandb.name",
        "policy.megatron_cfg.cuda_graph_impl",
        "policy.megatron_cfg.env_vars",
        "policy.megatron_cfg.fp8_cfg.enabled",
        "policy.megatron_cfg.fp8_cfg.fp8_recipe",
        "policy.megatron_cfg.moe_mlp_glu_interleave_size",
        "policy.megatron_cfg.moe_router_dtype",
        "policy.megatron_cfg.use_transformer_engine_op_fuser",
    }
