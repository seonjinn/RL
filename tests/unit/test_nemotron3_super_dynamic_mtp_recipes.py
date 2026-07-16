# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from pathlib import Path

from omegaconf import DictConfig

from nemo_rl.utils.config import load_config


RECIPE_DIR = Path(__file__).parents[2] / "examples/configs/recipes/llm/performance"
RECIPE_PREFIX = "grpo-nemotron3-super-120BA12B-32n4g"


def _load_recipe(suffix: str) -> DictConfig:
    return load_config(RECIPE_DIR / f"{RECIPE_PREFIX}-{suffix}.yaml")


def test_dynamic_mtp_recipes_preserve_super_performance_topology():
    configs = [
        _load_recipe("mtp-off"),
        _load_recipe("native-mtp-k5"),
        _load_recipe("dynamic-native-mtp-k5"),
    ]

    for config in configs:
        assert config.policy.model_name == (
            "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
        )
        assert config.policy.train_global_batch_size == 256
        assert config.grpo.num_prompts_per_step == 32
        assert config.grpo.num_generations_per_prompt == 8
        assert config.policy.max_total_sequence_length == 8192
        assert config.policy.megatron_cfg.tensor_model_parallel_size == 2
        assert config.policy.megatron_cfg.expert_model_parallel_size == 16
        assert config.policy.megatron_cfg.mtp_num_layers == 0
        assert config.policy.generation.vllm_cfg.tensor_parallel_size == 4
        assert config.policy.generation.vllm_cfg.enforce_eager is False
        assert config.policy.generation.vllm_kwargs.moe_backend == "triton"
        assert (
            config.policy.generation.vllm_kwargs.compilation_config.cudagraph_mode
            == "PIECEWISE"
        )
        assert config.checkpointing.enabled is False
        assert config.cluster.num_nodes == 32
        assert config.cluster.gpus_per_node == 4
        assert config.cluster.segment_size == 8


def test_static_native_mtp_uses_all_five_checkpoint_heads():
    config = _load_recipe("native-mtp-k5")
    speculative_config = config.policy.generation.vllm_kwargs.speculative_config

    assert speculative_config.method == "mtp"
    assert speculative_config.num_speculative_tokens == 5
    assert "num_speculative_tokens_per_batch_size" not in speculative_config


def test_dynamic_native_mtp_reduces_depth_as_concurrency_grows():
    config = _load_recipe("dynamic-native-mtp-k5")
    speculative_config = config.policy.generation.vllm_kwargs.speculative_config

    assert speculative_config.method == "mtp"
    assert speculative_config.num_speculative_tokens == 5
    assert speculative_config.num_speculative_tokens_per_batch_size == [
        [1, 64, 5],
        [65, 128, 3],
        [129, 256, 1],
    ]


def test_mtp_off_control_has_no_speculative_config():
    config = _load_recipe("mtp-off")

    assert "speculative_config" not in config.policy.generation.vllm_kwargs
