"""Compose the committed Q30 configs through the exact df9 loader."""

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
from nemo_rl.utils.config import load_config, parse_hydra_overrides, register_omegaconf_resolvers  # noqa: E402


register_omegaconf_resolvers()
overrides = [
    "++policy.generation.vllm_kwargs.max_num_seqs=8",
    "++policy.generation.vllm_kwargs.compilation_config.backend=eager",
    "++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=PIECEWISE",
    "++policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes=[1,2,4,8,12,16,24,32,40,48]",
]
composed: dict[str, object] = {}
for config_path in args.config:
    config = parse_hydra_overrides(load_config(config_path), overrides)
    generation = config.policy.generation
    assert config.grpo.max_num_steps == 20
    assert config.grpo.async_grpo.enabled is False
    assert config.data.shuffle is False
    assert config.data.train.dataset_name == "OpenMathInstruct-2"
    assert config.policy.train_global_batch_size == 512
    assert config.policy.max_total_sequence_length == 8192
    assert generation.max_new_tokens == 1024
    assert generation.vllm_cfg.max_model_len == 8192
    assert generation.vllm_cfg.enforce_eager is False
    assert generation.vllm_kwargs.max_num_seqs == 8
    assert generation.vllm_kwargs.compilation_config.backend == "eager"
    assert generation.vllm_kwargs.compilation_config.cudagraph_mode == "PIECEWISE"
    assert generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes == [1, 2, 4, 8, 12, 16, 24, 32, 40, 48]
    assert config.policy.megatron_cfg.tensor_model_parallel_size == 1
    assert config.policy.megatron_cfg.pipeline_model_parallel_size == 1
    assert config.policy.megatron_cfg.expert_model_parallel_size == 16
    assert config.policy.draft.anchors_per_sample == 2
    assert config.policy.draft.mask_token_id == 151669
    assert config.policy.draft.target_hidden_state_layer_ids == [1, 12, 23, 34, 45]
    assert config.policy.draft.num_layers == 5
    if config_path.stem == "dflash":
        assert config.policy.draft.gamma == 5
    if config_path.stem == "dspark":
        assert config.policy.draft.block_size == 8
        assert config.policy.draft.markov_rank == 256
        assert config.policy.draft.markov_head_type == "vanilla"
        assert config.policy.draft.confidence_enabled is True
        assert config.policy.draft.confidence_with_markov is True
    composed[config_path.stem] = {
        "draft_model": config.policy.draft.model_name,
        "max_num_seqs": generation.vllm_kwargs.max_num_seqs,
    }
print(json.dumps(composed, sort_keys=True))
