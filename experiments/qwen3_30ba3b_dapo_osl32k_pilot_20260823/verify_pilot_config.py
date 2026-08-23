"""Compose and verify a committed OSL32K pilot config through the exact product."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--source-root", type=Path, required=True)
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--capture-sizes", type=json.loads, required=True)
args = parser.parse_args()

sys.path.insert(0, str(args.source_root))
from nemo_rl.utils.config import load_config, parse_hydra_overrides, register_omegaconf_resolvers


register_omegaconf_resolvers()
capture_sizes = args.capture_sizes
if capture_sizes != [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64]:
    raise SystemExit(f"invalid K7 capture sizes: {capture_sizes}")
overrides = [
    "++policy.generation.vllm_kwargs.max_num_seqs=8",
    "++policy.generation.vllm_kwargs.compilation_config.backend=eager",
    "++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=PIECEWISE",
    "++policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes=" + json.dumps(capture_sizes, separators=(",", ":")),
]
config = parse_hydra_overrides(load_config(args.config), overrides)
variant = args.config.stem.removeprefix("resolved-input-")
generation = config.policy.generation
assert config.grpo.max_num_steps == 2
assert config.grpo.num_prompts_per_step == 16
assert config.grpo.num_generations_per_prompt == 4
assert config.grpo.seed == 42
assert config.grpo.async_grpo.enabled is False
assert config.data.max_input_seq_length == 2048
assert config.data.shuffle is False
assert config.data.train.dataset_name == "ResponseDataset"
assert config.data.validation is None
assert config.data.default.prompt_file is None
assert config.policy.train_global_batch_size == 64
assert config.policy.train_micro_batch_size == 1
assert config.policy.logprob_batch_size == 1
assert config.policy.logprob_chunk_size == 2048
assert config.policy.max_total_sequence_length == 40960
assert config.policy.make_sequence_length_divisible_by == 8
assert config.policy.sequence_packing.enabled is True
megatron = config.policy.megatron_cfg
assert (megatron.tensor_model_parallel_size, megatron.pipeline_model_parallel_size, megatron.expert_model_parallel_size, megatron.context_parallel_size) == (2, 1, 8, 2)
assert megatron.sequence_parallel is True
assert megatron.activation_checkpointing is True
assert megatron.defer_fp32_logits is True
assert megatron.empty_unused_memory_level == 2
assert megatron.checkpoint.async_save is False
assert config.cluster.num_nodes * config.cluster.gpus_per_node == 16
assert generation.max_new_tokens == 32768
assert generation.vllm_cfg.tensor_parallel_size == 1
assert generation.vllm_cfg.max_model_len == 40960
assert generation.vllm_cfg.gpu_memory_utilization == 0.5
assert generation.vllm_cfg.enforce_eager is False
assert generation.vllm_kwargs.max_num_seqs == 8
assert generation.vllm_kwargs.compilation_config.cudagraph_mode == "PIECEWISE"
assert generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes == capture_sizes
if variant == "baseline-k0":
    assert "speculative_config" not in generation.vllm_kwargs
else:
    method = variant.split("-", 1)[0]
    assert generation.vllm_kwargs.speculative_config.method == method
    assert generation.vllm_kwargs.speculative_config.num_speculative_tokens == 7
    assert config.policy.draft.enabled is True
print(json.dumps({"variant": variant, "CONFIG_COMPOSE_GATE_PASS": True, "max_output_length": generation.max_new_tokens, "max_model_len": generation.vllm_cfg.max_model_len, "capture_sizes": capture_sizes}, sort_keys=True))
