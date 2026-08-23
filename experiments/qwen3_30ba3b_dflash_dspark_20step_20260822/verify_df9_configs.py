"""Compose the committed Q30 configs through the exact df9 loader."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--source-root", type=Path, required=True)
parser.add_argument("--config", type=Path, action="append", required=True)
parser.add_argument("--capture-sizes", type=json.loads, required=True)
args = parser.parse_args()

sys.path.insert(0, str(args.source_root))
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)  # noqa: E402


register_omegaconf_resolvers()
capture_sizes = args.capture_sizes
if (
    not isinstance(capture_sizes, list)
    or not capture_sizes
    or not all(isinstance(size, int) and size > 0 for size in capture_sizes)
    or capture_sizes != sorted(set(capture_sizes))
):
    raise SystemExit("capture sizes must be a sorted, unique list of positive integers")
overrides = [
    "++policy.generation.vllm_kwargs.max_num_seqs=8",
    "++policy.generation.vllm_kwargs.compilation_config.backend=eager",
    "++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=PIECEWISE",
    "++policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes="
    + json.dumps(capture_sizes, separators=(",", ":")),
]
composed: dict[str, object] = {}
for config_path in args.config:
    variant = config_path.stem.removeprefix("resolved-input-")
    config = parse_hydra_overrides(load_config(config_path), overrides)
    generation = config.policy.generation
    assert config.grpo.max_num_steps == 20
    assert config.grpo.val_period == 0
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
    assert (
        generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes
        == capture_sizes
    )
    assert config.policy.megatron_cfg.tensor_model_parallel_size == 2
    assert config.policy.megatron_cfg.pipeline_model_parallel_size == 1
    assert config.policy.megatron_cfg.expert_model_parallel_size == 8
    assert config.policy.megatron_cfg.context_parallel_size == 1
    assert config.policy.megatron_cfg.sequence_parallel is True
    assert config.policy.sequence_packing.enabled is True
    assert config.policy.make_sequence_length_divisible_by == 2
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
        == 8
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
    method = variant.split("-", maxsplit=1)[0]
    if method == "baseline":
        assert "speculative_config" not in generation.vllm_kwargs
        if "draft" in config.policy:
            assert config.policy.draft.enabled is False
        composed[variant] = {
            "draft_model": None,
            "max_num_seqs": generation.vllm_kwargs.max_num_seqs,
        }
        continue
    speculative = generation.vllm_kwargs.speculative_config
    assert speculative.draft_tensor_parallel_size == 1
    assert speculative.method == method
    if "-k" in variant:
        assert speculative.num_speculative_tokens == int(
            variant.rsplit("-k", maxsplit=1)[1]
        )
    assert config.policy.draft.anchors_per_sample == 2
    assert config.policy.draft.mask_token_id == 151669
    assert config.policy.draft.target_hidden_state_layer_ids == [1, 12, 23, 34, 45]
    assert config.policy.draft.num_layers == 5
    if method == "dflash":
        assert config.policy.draft.gamma == speculative.num_speculative_tokens
    if method == "dspark":
        assert config.policy.draft.block_size == 8
        assert config.policy.draft.markov_rank == 256
        assert config.policy.draft.markov_head_type == "vanilla"
        assert config.policy.draft.confidence_enabled is True
        assert config.policy.draft.confidence_with_markov is True
    composed[variant] = {
        "draft_model": config.policy.draft.model_name,
        "max_num_seqs": generation.vllm_kwargs.max_num_seqs,
    }
print(json.dumps(composed, sort_keys=True))
