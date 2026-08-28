"""Compose stable Q30 configs through the pinned product loader."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, action="append", required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.source_root))
    from nemo_rl.algorithms.grpo import MasterConfig  # noqa: PLC0415
    from nemo_rl.utils.config import (  # noqa: PLC0415
        load_config,
        parse_hydra_overrides,
        register_omegaconf_resolvers,
    )
    from omegaconf import OmegaConf  # noqa: PLC0415

    register_omegaconf_resolvers()
    overrides = [
        "logger.wandb_enabled=true",
        "logger.wandb.project=sna-specdec",
        "+logger.wandb.group=q30ba3b-fixed-vs-always-stable-200step-20260827",
        "logger.wandb.name=config-composition-probe",
    ]
    composed: dict[str, object] = {}
    for config_path in args.config:
        variant = config_path.stem.removeprefix("resolved-input-")
        drafter, training_mode = variant.split("-", 1)
        config = parse_hydra_overrides(load_config(config_path), overrides)
        validated = MasterConfig(**OmegaConf.to_container(config, resolve=True))
        generation = config.policy.generation
        draft = config.policy.draft

        assert config.grpo.max_num_steps == 200
        assert config.grpo.num_prompts_per_step == 64
        assert config.grpo.num_generations_per_prompt == 32
        assert config.grpo.val_period == 10
        assert config.checkpointing.enabled is False
        assert config.checkpointing.keep_top_k == 3
        assert config.grpo.async_grpo.enabled is False
        assert config.data.shuffle is True
        assert config.data.train.dataset_name == "OpenMathInstruct-2"
        assert config.data.train.split_validation_size == 0.05
        assert config.data_plane.enabled is False
        assert config.policy.train_global_batch_size == 2048
        assert config.policy.max_total_sequence_length == 4096
        assert config.policy.sequence_packing.enabled is True
        assert config.policy.sequence_packing.fuse_loss is True
        assert config.policy.make_sequence_length_divisible_by == 1
        assert generation.max_new_tokens == 4096
        assert generation.vllm_cfg.tensor_parallel_size == 1
        assert generation.vllm_cfg.max_model_len == 4096
        assert generation.vllm_cfg.enforce_eager is False
        vllm_kwargs = OmegaConf.to_container(generation.vllm_kwargs, resolve=True)
        assert isinstance(vllm_kwargs, dict)
        assert set(vllm_kwargs) == {"moe_backend", "speculative_config"}
        assert vllm_kwargs["moe_backend"] == "triton"
        assert config.logger.wandb_enabled is True
        assert config.logger.wandb.project == "sna-specdec"
        assert (
            config.logger.wandb.group
            == "q30ba3b-fixed-vs-always-stable-200step-20260827"
        )
        assert config.logger.wandb.name == "config-composition-probe"
        assert (
            validated.logger["wandb"].get("group")
            == "q30ba3b-fixed-vs-always-stable-200step-20260827"
        )
        assert generation.vllm_kwargs.speculative_config.method == drafter
        assert generation.vllm_kwargs.speculative_config.num_speculative_tokens == 5
        assert generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size == 1
        assert draft.speculator_type == drafter
        assert draft.anchors_per_sample == 2
        assert draft.mask_token_id == 151669
        assert draft.target_hidden_state_layer_ids == [1, 12, 23, 34, 45]
        assert draft.num_layers == 5
        assert not hasattr(draft, "update_schedule")
        if drafter == "dflash":
            assert draft.gamma == 5
        elif drafter == "dspark":
            assert draft.block_size == 5
            assert draft.markov_rank == 256
            assert draft.markov_head_type == "vanilla"
            assert draft.confidence_enabled is True
            assert draft.confidence_with_markov is True
        else:
            raise ValueError(f"unsupported drafter {drafter!r}")

        if training_mode == "fixed":
            assert draft.enabled is False
            assert draft.optimizer is None
        elif training_mode == "always":
            assert draft.enabled is True
            assert draft.optimizer.lr == 5e-6
            assert draft.optimizer.min_lr == 5e-7
            assert draft.optimizer.weight_decay == 0.01
        else:
            raise ValueError(f"unsupported training mode {training_mode!r}")

        megatron = config.policy.megatron_cfg
        assert megatron.tensor_model_parallel_size == 1
        assert megatron.pipeline_model_parallel_size == 1
        assert megatron.expert_model_parallel_size == 16
        assert megatron.context_parallel_size == 1
        assert megatron.sequence_parallel is False
        training_world_size = config.cluster.num_nodes * config.cluster.gpus_per_node
        assert training_world_size == 16
        assert (
            training_world_size
            % (
                megatron.tensor_model_parallel_size
                * megatron.expert_model_parallel_size
                * megatron.pipeline_model_parallel_size
            )
            == 0
        )
        composed[variant] = {
            "draft_model": draft.model_name,
            "draft_training_enabled": draft.enabled,
            "performance_recipe_preserved": True,
        }
    print(json.dumps(composed, sort_keys=True))


if __name__ == "__main__":
    main()
