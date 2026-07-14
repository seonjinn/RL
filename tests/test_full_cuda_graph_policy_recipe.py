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
CONFIGS = PROJECT_ROOT / "examples/configs"
PERFORMANCE_RECIPES = CONFIGS / "recipes/llm/performance"
PARENT_NAME = "grpo-qwen3-30ba3b-1n4g-megatron-mxfp8-cutedsl"
RECIPE_NAME = "grpo-qwen3-30ba3b-2n4g-megatron-mxfp8-full-cg-noncolocated"
RECIPE = PERFORMANCE_RECIPES / f"{RECIPE_NAME}.yaml"
DISABLED_SUITE = PROJECT_ROOT / "tests/test_suites/disabled.txt"
LAUNCH = PROJECT_ROOT / "tools/launch"
FEATURE_LAUNCHERS = {
    "grpo-qwen3-30ba3b-1n4g-megatron-mxfp8-cutedsl": (1, 4, 3, None),
    "grpo-qwen3-30ba3b-1n4g-megatron-mxfp8-a2a-overlap": (1, 4, 3, None),
    "grpo-qwen3-30ba3b-2n4g-megatron-mxfp8-factorial": (2, 4, 3, 2),
    "grpo-qwen3-30ba3b-4n4g-megatron-mxfp8-cutedsl": (4, 4, 3, 4),
    RECIPE_NAME: (2, 4, 6, None),
}
DEFAULT_SUITES = (
    PROJECT_ROOT / "tests/test_suites/nightly.txt",
    PROJECT_ROOT / "tests/test_suites/release.txt",
    PROJECT_ROOT / "tests/test_suites/nightly_gb200.txt",
    PROJECT_ROOT / "tests/test_suites/release_gb200.txt",
    PROJECT_ROOT / "tests/test_suites/performance.txt",
    PROJECT_ROOT / "tests/test_suites/performance_gb200.txt",
)
EXEMPLARS = (
    CONFIGS / "grpo_math_1B.yaml",
    CONFIGS / "grpo_math_1B_megatron.yaml",
    PROJECT_ROOT / "tests/unit/reference_configs/grpo_math_1B.yaml",
)
MCORE_DISABLED_DEFAULTS = {
    "cuda_graph_impl": "none",
    "cuda_graph_warmup_steps": 3,
    "cuda_graph_use_single_mempool": True,
    "moe_expert_rank_capacity_factor": None,
    "moe_paged_stash": False,
    "moe_paged_stash_page_size": 64,
    "moe_paged_stash_buffer_size_factor_cuda": 1.10,
    "moe_paged_stash_buffer_size_factor_cpu": 0.0,
}

register_omegaconf_resolvers()


def _load_yaml(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    return cast(dict[str, Any], config)


def _load_resolved(path: Path) -> dict[str, Any]:
    config = OmegaConf.to_container(load_config(path), resolve=True)
    assert isinstance(config, dict)
    return cast(dict[str, Any], config)


def test_full_cuda_graph_defaults_are_explicit_and_mcore_compatible() -> None:
    for exemplar in EXEMPLARS:
        megatron = _load_yaml(exemplar)["policy"]["megatron_cfg"]
        assert {
            key: megatron[key] for key in MCORE_DISABLED_DEFAULTS
        } == MCORE_DISABLED_DEFAULTS


def test_full_cuda_graph_recipe_has_the_non_factorial_parent() -> None:
    raw = _load_yaml(RECIPE)

    assert raw["defaults"] == f"./{PARENT_NAME}.yaml"
    assert raw["cluster"]["segment_size"] is None
    assert raw["policy"]["megatron_cfg"]["expert_model_parallel_size"] == 4
    assert raw["policy"]["megatron_cfg"]["expert_model_parallel_size"] != 8


def test_full_cuda_graph_recipe_has_fixed_policy_shape_and_topology() -> None:
    config = _load_resolved(RECIPE)
    policy = config["policy"]
    megatron = policy["megatron_cfg"]
    cluster = config["cluster"]
    generation = policy["generation"]

    assert policy["model_name"] == "Qwen/Qwen3-30B-A3B"
    assert {
        key: cluster[key] for key in ("num_nodes", "gpus_per_node", "segment_size")
    } == {"num_nodes": 2, "gpus_per_node": 4, "segment_size": None}
    assert generation["backend"] == "vllm"
    assert generation["colocated"] == {
        "enabled": False,
        "resources": {"num_nodes": 1, "gpus_per_node": 4},
    }
    assert generation["vllm_cfg"]["async_engine"] is False
    assert generation["vllm_cfg"]["tensor_parallel_size"] == 1
    assert cluster["num_nodes"] - generation["colocated"]["resources"]["num_nodes"] == 1

    assert [
        megatron[key]
        for key in (
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
            "expert_tensor_parallel_size",
            "expert_model_parallel_size",
        )
    ] == [1, 1, 1, 1, 4]
    assert policy["train_global_batch_size"] == 4
    assert policy["train_micro_batch_size"] == 1
    assert policy["dynamic_batching"]["enabled"] is False
    assert policy["sequence_packing"]["enabled"] is False
    assert policy["max_total_sequence_length"] == 1024
    assert policy["make_sequence_length_divisible_by"] == 1024


def test_full_cuda_graph_recipe_has_capture_and_cutedsl_prerequisites() -> None:
    config = _load_resolved(RECIPE)
    megatron = config["policy"]["megatron_cfg"]

    assert megatron["cuda_graph_impl"] == "full_iteration"
    assert megatron["cuda_graph_warmup_steps"] == 3
    assert megatron["cuda_graph_use_single_mempool"] is True
    assert {
        key: megatron[key]
        for key in (
            "moe_expert_rank_capacity_factor",
            "moe_paged_stash",
            "moe_paged_stash_page_size",
            "moe_paged_stash_buffer_size_factor_cuda",
            "moe_paged_stash_buffer_size_factor_cpu",
        )
    } == {
        key: MCORE_DISABLED_DEFAULTS[key]
        for key in (
            "moe_expert_rank_capacity_factor",
            "moe_paged_stash",
            "moe_paged_stash_page_size",
            "moe_paged_stash_buffer_size_factor_cuda",
            "moe_paged_stash_buffer_size_factor_cpu",
        )
    }
    assert megatron["moe_grouped_gemm"] is True
    assert megatron["use_transformer_engine_op_fuser"] is True
    assert megatron["moe_mlp_glu_interleave_size"] == 32
    assert megatron["env_vars"]["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] == "1"
    assert megatron["fp8_cfg"] == {
        **megatron["fp8_cfg"],
        "enabled": True,
        "fp8": "e4m3",
        "fp8_recipe": "mxfp8",
        "fp8_param": False,
    }


def test_full_cuda_graph_recipe_skips_dynamic_policy_work() -> None:
    config = _load_resolved(RECIPE)
    grpo = config["grpo"]
    loss = config["loss_fn"]
    warmup_steps = config["policy"]["megatron_cfg"]["cuda_graph_warmup_steps"]

    assert grpo["async_grpo"]["enabled"] is False
    assert config["data_plane"]["enabled"] is False
    assert grpo["max_num_steps"] >= 6
    assert grpo["max_num_steps"] >= warmup_steps + 3
    assert grpo["skip_reference_policy_logprobs_calculation"] is True
    assert grpo["seq_logprob_error_threshold"] is None
    assert grpo["val_period"] == 0
    assert grpo["val_at_start"] is False
    assert grpo["val_at_end"] is False
    assert config["data"]["train"]["split_validation_size"] == 0.0
    assert config["data"]["validation"] is None
    assert loss["force_on_policy_ratio"] is True
    assert loss["reference_policy_kl_penalty"] == 0.0
    assert config["policy"]["train_global_batch_size"] == (
        grpo["num_prompts_per_step"] * grpo["num_generations_per_prompt"]
    )
    assert config["policy"]["megatron_cfg"]["overlap_moe_expert_parallel_comm"] is False


def test_nemo_2606_feature_launchers_are_disabled_pending_gpu_validation() -> None:
    disabled_entries = DISABLED_SUITE.read_text(encoding="utf-8").splitlines()

    for name, (nodes, gpus_per_node, steps, segment_size) in FEATURE_LAUNCHERS.items():
        suite_entry = f"tests/test_suites/llm/performance/{name}.sh"
        suite_script = PROJECT_ROOT / suite_entry
        recipe = _load_resolved(PERFORMANCE_RECIPES / f"{name}.yaml")

        assert suite_script.is_file()
        assert os.access(suite_script, os.X_OK)
        assert recipe["cluster"]["num_nodes"] == nodes
        assert recipe["cluster"]["gpus_per_node"] == gpus_per_node
        assert recipe["cluster"]["segment_size"] == segment_size
        assert disabled_entries.count(suite_entry) == 1
        for default_suite in DEFAULT_SUITES:
            assert (
                suite_entry
                not in default_suite.read_text(encoding="utf-8").splitlines()
            )

        script = suite_script.read_text(encoding="utf-8")
        assert f"NUM_NODES={nodes}" in script.splitlines()
        assert f"GPUS_PER_NODE={gpus_per_node}" in script.splitlines()
        assert f"STEPS_PER_RUN={steps}" in script.splitlines()
        assert f"MAX_STEPS={steps}" in script.splitlines()
        if segment_size is None:
            assert not any(
                line.startswith("SEGMENT_SIZE=") for line in script.splitlines()
            )
        else:
            assert f"SEGMENT_SIZE={segment_size}" in script.splitlines()

        syntax = subprocess.run(
            ["bash", "-n", str(suite_script)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert syntax.returncode == 0, syntax.stderr

        dry_run = subprocess.run(
            ["bash", str(suite_script)],
            cwd=PROJECT_ROOT,
            env={**os.environ, "TEST_DRYRUN": "1"},
            capture_output=True,
            text=True,
            check=False,
        )
        assert dry_run.returncode == 0, dry_run.stderr

        launch = subprocess.run(
            [str(LAUNCH), str(suite_script)],
            cwd=PROJECT_ROOT,
            env={
                **os.environ,
                "DRYRUN": "1",
                "HF_HOME": "...",
                "HF_DATASETS_CACHE": "...",
                "CONTAINER": "",
                "ACCOUNT": "",
                "PARTITION": "",
            },
            capture_output=True,
            text=True,
            check=False,
        )
        assert launch.returncode == 0, launch.stderr
