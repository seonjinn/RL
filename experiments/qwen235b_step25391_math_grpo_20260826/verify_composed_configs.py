"""Compose and validate the Qwen3-235B Base Math GRPO benchmark arms."""

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
from nemo_rl.utils.config import (  # noqa: E402
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from omegaconf import OmegaConf  # noqa: E402


TARGET = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/"
    "hf_home/hub/models--Qwen--Qwen3-235B-A22B/snapshots/"
    "8efa61729e24bd65b1d152b5ab5409052aa80e65"
)
DRAFTER = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/"
    "modelopt-specdec/checkpoints/"
    "qwen3-235ba22b-base-nemotron-b8-s25391/dspark"
)


register_omegaconf_resolvers()
composed: dict[str, object] = {}
for config_path in args.config:
    arm = config_path.stem.removeprefix("resolved-input-")
    config = parse_hydra_overrides(load_config(config_path), [])
    MasterConfig(**OmegaConf.to_container(config, resolve=True))
    generation = config.policy.generation
    kwargs = OmegaConf.to_container(generation.vllm_kwargs, resolve=True)
    assert isinstance(kwargs, dict)
    assert config.grpo.max_num_steps == 20
    assert config.grpo.num_prompts_per_step == 16
    assert config.grpo.num_generations_per_prompt == 32
    assert config.grpo.async_grpo.enabled is False
    assert config.checkpointing.enabled is False
    assert config.data.train.dataset_name == "OpenMathInstruct-2"
    assert config.policy.model_name == TARGET
    assert config.policy.tokenizer.name == TARGET
    assert config.policy.train_global_batch_size == 512
    assert config.policy.max_total_sequence_length == 8192
    assert config.policy.sequence_packing.enabled is True
    assert config.policy.sequence_packing.fuse_loss is True
    assert config.policy.megatron_cfg.tensor_model_parallel_size == 2
    assert config.policy.megatron_cfg.pipeline_model_parallel_size == 4
    assert config.policy.megatron_cfg.context_parallel_size == 2
    assert config.policy.megatron_cfg.expert_model_parallel_size == 16
    assert generation.vllm_cfg.tensor_parallel_size == 8
    assert generation.vllm_cfg.max_model_len == 8192
    assert generation.max_new_tokens == 1024
    assert kwargs["max_num_batched_tokens"] == 8192
    assert kwargs["max_num_seqs"] == 32
    assert kwargs["moe_backend"] == "triton"
    assert kwargs["disable_custom_all_reduce"] is True
    assert kwargs["compilation_config"]["cudagraph_mode"] == "FULL_AND_PIECEWISE"
    assert config.cluster.num_nodes == 32
    assert config.cluster.gpus_per_node == 4
    speculative = kwargs.get("speculative_config")
    if arm == "baseline":
        assert speculative is None
        k = 0
    else:
        k = int(arm.removeprefix("dspark_k"))
        assert k in {3, 5, 7}
        assert speculative["method"] == "dspark"
        assert speculative["model"] == DRAFTER
        assert speculative["num_speculative_tokens"] == k
        assert speculative["draft_tensor_parallel_size"] == 1
        assert speculative["attention_backend"] == "FLASH_ATTN"
    composed[arm] = {"fap": True, "num_speculative_tokens": k}

print(json.dumps(composed, sort_keys=True))
