"""Compose the committed Q30 configs through the pinned product loader."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--source-root", type=Path, required=True)
parser.add_argument("--config", type=Path, action="append", required=True)
args = parser.parse_args()

sys.path.insert(0, str(args.source_root))
from nemo_rl.algorithms.grpo import MasterConfig  # noqa: E402
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


register_omegaconf_resolvers()
overrides: list[str] = []
composed: dict[str, object] = {}
for config_path in args.config:
    variant = config_path.stem.removeprefix("resolved-input-")
    drafter, cadence = variant.split("-", 1)
    config = parse_hydra_overrides(load_config(config_path), overrides)
    MasterConfig(**OmegaConf.to_container(config, resolve=True))
    generation = config.policy.generation
    assert config.grpo.max_num_steps == 200
    assert config.grpo.num_prompts_per_step == 64
    assert config.grpo.num_generations_per_prompt == 32
    assert config.grpo.val_period == 10
    assert config.grpo.async_grpo.enabled is False
    assert config.checkpointing.enabled is False
    assert config.data.shuffle is True
    assert config.data.train.dataset_name == "OpenMathInstruct-2"
    assert config.data.train.split_validation_size == 0.05
    assert config.data_plane.enabled is False
    assert config.policy.train_global_batch_size == 2048
    assert config.policy.max_total_sequence_length == 4096
    assert config.policy.offload_optimizer_for_refit is False
    assert generation.max_new_tokens == 4096
    assert generation.vllm_cfg.max_model_len == 4096
    assert generation.vllm_cfg.enforce_eager is False
    vllm_kwargs = OmegaConf.to_container(generation.vllm_kwargs, resolve=True)
    assert isinstance(vllm_kwargs, dict)
    assert set(vllm_kwargs) == {"moe_backend", "speculative_config"}
    assert vllm_kwargs["moe_backend"] == "triton"
    assert config.policy.megatron_cfg.tensor_model_parallel_size == 1
    assert config.policy.megatron_cfg.pipeline_model_parallel_size == 1
    assert config.policy.megatron_cfg.expert_model_parallel_size == 16
    assert config.policy.megatron_cfg.context_parallel_size == 1
    assert config.policy.megatron_cfg.sequence_parallel is False
    assert config.policy.sequence_packing.enabled is True
    assert config.policy.sequence_packing.fuse_loss is True
    assert config.policy.make_sequence_length_divisible_by == 1
    training_world_size = config.cluster.num_nodes * config.cluster.gpus_per_node
    assert training_world_size == 16
    assert (
        training_world_size
        % (
            config.policy.megatron_cfg.tensor_model_parallel_size
            * config.policy.megatron_cfg.expert_model_parallel_size
            * config.policy.megatron_cfg.pipeline_model_parallel_size
        )
        == 0
    )
    assert (
        training_world_size
        // (
            config.policy.megatron_cfg.tensor_model_parallel_size
            * config.policy.megatron_cfg.pipeline_model_parallel_size
            * config.policy.megatron_cfg.context_parallel_size
        )
        == 16
    )
    assert (
        training_world_size
        // (
            config.policy.megatron_cfg.tensor_model_parallel_size
            * config.policy.megatron_cfg.expert_model_parallel_size
            * config.policy.megatron_cfg.pipeline_model_parallel_size
        )
        == 1
    )
    assert generation.vllm_cfg.tensor_parallel_size == 1
    assert generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size == 1
    assert config.policy.draft.anchors_per_sample == 2
    assert config.policy.draft.mask_token_id == 151669
    assert config.policy.draft.target_hidden_state_layer_ids == [1, 12, 23, 34, 45]
    assert config.policy.draft.num_layers == 5
    if drafter == "dflash":
        assert config.policy.draft.gamma == 5
    if drafter == "dspark":
        assert config.policy.draft.block_size == 5
        assert config.policy.draft.markov_rank == 256
        assert config.policy.draft.markov_head_type == "vanilla"
        assert config.policy.draft.confidence_enabled is True
        assert config.policy.draft.confidence_with_markov is True
    schedule = config.policy.draft.update_schedule
    if cadence in {"fixed5", "fixed10", "fixed20"}:
        assert schedule.mode == "fixed"
        assert schedule.action == "sparse_update"
        assert schedule.fixed_interval == int(cadence.removeprefix("fixed"))
    elif cadence == "adaptive-v2":
        assert schedule.mode == "adaptive"
        assert schedule.action == "sparse_update"
        assert schedule.min_interval == 10
        assert schedule.max_interval == 40
        assert schedule.ewma_alpha == 0.2
        assert schedule.degradation_threshold == 0.03
        assert schedule.recovery_threshold == 0.01
        assert schedule.min_observations == 10
        assert schedule.max_burst_updates == 2
    else:
        raise ValueError(f"unknown cadence {cadence!r}")
    assert config.cadence_runtime.enabled is False
    composed[variant] = {
        "draft_model": config.policy.draft.model_name,
        "fixed_interval": schedule.fixed_interval,
        "performance_recipe_preserved": True,
    }
print(json.dumps(composed, sort_keys=True))
