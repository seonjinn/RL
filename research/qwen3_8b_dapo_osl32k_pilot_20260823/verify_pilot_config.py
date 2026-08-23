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
parser.add_argument("--static-only", action="store_true")
args = parser.parse_args()

capture_sizes = args.capture_sizes
if capture_sizes != [1, 2, 4, 6, 8, 12, 16, 18, 24, 30, 32, 36, 40, 42, 48, 56, 64]:
    raise SystemExit(f"invalid K5 capture sizes: {capture_sizes}")
variant = args.config.stem.removeprefix("resolved-input-")
raw = json.loads(args.config.read_text())
recipe = (
    "grpo-qwen3-8b-1n8g-megatron-dspark.yaml"
    if variant == "dspark-k5"
    else "grpo-qwen3-8b-1n8g-megatron-dflash.yaml"
)
expected_default = args.source_root / "examples/configs/recipes/llm" / recipe
assert raw["defaults"] == str(expected_default)
assert raw["grpo"] == {
    "max_num_steps": 2,
    "num_prompts_per_step": 2,
    "num_generations_per_prompt": 4,
    "val_period": 0,
    "seed": 42,
    "async_grpo": {"enabled": False},
}
assert raw["data"]["max_input_seq_length"] == 2048
assert raw["data"]["shuffle"] is False
assert raw["data"]["train"]["dataset_name"] == "ResponseDataset"
policy = raw["policy"]
assert policy["train_global_batch_size"] == 8
assert policy["train_micro_batch_size"] == 1
assert policy["logprob_batch_size"] == 1
assert policy["logprob_chunk_size"] == 2048
assert policy["max_total_sequence_length"] == 40960
assert policy["make_sequence_length_divisible_by"] == 8
assert policy["sequence_packing"] == {"enabled": False}
megatron_raw = policy["megatron_cfg"]
assert tuple(
    megatron_raw[key]
    for key in (
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
    )
) == (2, 1, 1)
assert megatron_raw["sequence_parallel"] is False
assert megatron_raw["activation_checkpointing"] is True
assert megatron_raw["defer_fp32_logits"] is True
generation_raw = policy["generation"]
assert generation_raw["max_new_tokens"] == 32768
assert generation_raw["vllm_cfg"] == {
    "tensor_parallel_size": 1,
    "max_model_len": 40960,
    "gpu_memory_utilization": 0.7,
    "enforce_eager": False,
}
if variant == "baseline-k0":
    assert policy["draft"] == {"enabled": False}
    assert generation_raw["vllm_kwargs"]["speculative_config"] is None
else:
    method = variant.split("-", 1)[0]
    speculative = generation_raw["vllm_kwargs"]["speculative_config"]
    assert speculative["method"] == method
    assert speculative["num_speculative_tokens"] == 5
    draft_raw = policy["draft"]
    assert draft_raw["speculator_type"] == method
    if method == "dflash":
        assert draft_raw["gamma"] == 5
        assert "block_size" not in draft_raw
    else:
        assert "gamma" not in draft_raw
        assert draft_raw["block_size"] == 7
if args.static_only:
    print(
        json.dumps(
            {"variant": variant, "STATIC_CONFIG_GATE_PASS": True}, sort_keys=True
        )
    )
    raise SystemExit(0)

sys.path.insert(0, str(args.source_root))
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


register_omegaconf_resolvers()
overrides = [
    "++policy.generation.vllm_kwargs.max_num_seqs=8",
    "++policy.generation.vllm_kwargs.compilation_config.backend=eager",
    "++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=PIECEWISE",
    "++policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes="
    + json.dumps(capture_sizes, separators=(",", ":")),
]
config = parse_hydra_overrides(load_config(args.config), overrides)
generation = config.policy.generation
assert config.grpo.max_num_steps == 2
assert config.grpo.num_prompts_per_step == 2
assert config.grpo.num_generations_per_prompt == 4
assert config.grpo.seed == 42
assert config.grpo.async_grpo.enabled is False
assert config.data.max_input_seq_length == 2048
assert config.data.shuffle is False
assert config.data.train.dataset_name == "ResponseDataset"
assert config.data.validation is None
assert config.data.default.prompt_file is None
assert config.policy.train_global_batch_size == 8
assert config.policy.train_micro_batch_size == 1
assert config.policy.logprob_batch_size == 1
assert config.policy.logprob_chunk_size == 2048
assert config.policy.max_total_sequence_length == 40960
assert config.policy.make_sequence_length_divisible_by == 8
assert config.policy.sequence_packing.enabled is False
megatron = config.policy.megatron_cfg
assert (
    megatron.tensor_model_parallel_size,
    megatron.pipeline_model_parallel_size,
    megatron.context_parallel_size,
) == (2, 1, 1)
assert megatron.sequence_parallel is False
assert megatron.activation_checkpointing is True
assert megatron.defer_fp32_logits is True
assert megatron.empty_unused_memory_level == 2
assert megatron.checkpoint.async_save is False
assert config.cluster.num_nodes * config.cluster.gpus_per_node == 4
assert generation.max_new_tokens == 32768
assert generation.vllm_cfg.tensor_parallel_size == 1
assert generation.vllm_cfg.max_model_len == 40960
assert generation.vllm_cfg.gpu_memory_utilization == 0.7
assert generation.vllm_cfg.enforce_eager is False
assert generation.vllm_kwargs.max_num_seqs == 8
assert generation.vllm_kwargs.compilation_config.cudagraph_mode == "PIECEWISE"
assert (
    generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes == capture_sizes
)
if variant == "baseline-k0":
    assert config.policy.draft.enabled is False
    assert generation.vllm_kwargs.speculative_config is None
else:
    method = variant.split("-", 1)[0]
    assert generation.vllm_kwargs.speculative_config.method == method
    assert generation.vllm_kwargs.speculative_config.num_speculative_tokens == 5
    assert config.policy.draft.enabled is True
print(
    json.dumps(
        {
            "variant": variant,
            "CONFIG_COMPOSE_GATE_PASS": True,
            "max_output_length": generation.max_new_tokens,
            "max_model_len": generation.vllm_cfg.max_model_len,
            "capture_sizes": capture_sizes,
        },
        sort_keys=True,
    )
)
