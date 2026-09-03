"""Compose and validate the Qwen3-235B Base Math GRPO benchmark arms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


TARGET = "Qwen/Qwen3-235B-A22B"
DSPARK_DRAFTER = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/"
    "modelopt-specdec/checkpoints/"
    "qwen3-235ba22b-base-nemotron-b8-s25391/dspark"
)
EAGLE3_DRAFTER = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/"
    "hf_home/hub/models--nvidia--Qwen3-235B-A22B-Eagle3/snapshots/"
    "33f3c01ce807376d1171301b9a148b1b28f239ba"
)
DEFAULT_SMALL_CAPTURE_SIZES = [
    1,
    2,
    4,
    8,
    16,
    24,
    32,
    40,
    48,
    56,
    64,
    72,
    80,
    88,
    96,
    104,
    112,
    120,
    128,
    136,
    144,
    152,
    160,
    168,
    176,
    184,
    192,
    200,
    208,
    216,
    224,
    232,
    240,
    248,
    256,
    272,
    288,
    304,
    320,
    336,
    352,
    368,
    384,
    400,
    416,
    432,
    448,
    464,
    480,
    496,
    512,
]
SMALL_CAPTURE_SIZES_BY_K = {
    3: [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128],
    5: [1, 2, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 160, 192],
    7: [1, 2, 4, 7, 8, 14, 16, 28, 32, 56, 64, 112, 128, 224, 256],
}


def expected_capture_sizes(arm: str) -> list[int]:
    """Return the exact expanded CUDA Graph capture ladder for an arm."""
    if not arm.endswith("_cg2048"):
        raise ValueError(f"arm does not select expanded graphs: {arm}")
    base_arm = arm.removesuffix("_cg2048")
    if base_arm == "baseline":
        return sorted({*DEFAULT_SMALL_CAPTURE_SIZES, 1024, 2048})
    k = int(base_arm.rsplit("_k", maxsplit=1)[1])
    if k not in SMALL_CAPTURE_SIZES_BY_K:
        raise ValueError(f"unsupported expanded arm: {arm}")
    verifier_anchors = {
        (k + 1) * concurrency
        for concurrency in (64, 128, 256, 512)
        if (k + 1) * concurrency <= 2048
    }
    return sorted({*SMALL_CAPTURE_SIZES_BY_K[k], *verifier_anchors})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, action="append", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.source_root))
    from nemo_rl.algorithms.grpo import MasterConfig
    from nemo_rl.utils.config import (
        load_config,
        parse_hydra_overrides,
        register_omegaconf_resolvers,
    )
    from omegaconf import OmegaConf

    register_omegaconf_resolvers()
    composed: dict[str, dict[str, Any]] = {}
    for config_path in args.config:
        arm = config_path.stem.removeprefix("resolved-input-")
        base_arm = arm.removesuffix("_cg2048")
        graph_profile = "expanded_2048" if arm != base_arm else "default_small"
        config = parse_hydra_overrides(load_config(config_path), [])
        MasterConfig(**OmegaConf.to_container(config, resolve=True))
        generation = config.policy.generation
        kwargs = OmegaConf.to_container(generation.vllm_kwargs, resolve=True)
        assert isinstance(kwargs, dict)
        assert config.grpo.max_num_steps == 20
        assert config.grpo.num_prompts_per_step == 16
        assert config.grpo.num_generations_per_prompt == 32
        assert config.grpo.async_grpo.enabled is False
        assert config.grpo.val_period == 10
        assert config.grpo.max_val_samples == 16
        assert config.grpo.val_batch_size == 5
        assert config.checkpointing.enabled is True
        assert config.data.shuffle is True
        assert config.data.train.dataset_name == "OpenMathInstruct-2"
        assert config.data.train.split_validation_size == 0.05
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
        assert generation.max_new_tokens == 8192
        assert kwargs["moe_backend"] == "triton"
        assert config.cluster.num_nodes == 32
        assert config.cluster.gpus_per_node == 4
        speculative = kwargs.get("speculative_config")
        if base_arm == "baseline":
            assert speculative is None
            k = 0
        else:
            method, k_text = base_arm.rsplit("_k", maxsplit=1)
            k = int(k_text)
            assert (method, k) in {("dspark", 3), ("dspark", 5), ("dspark", 7), ("eagle3", 3)}
            assert isinstance(speculative, dict)
            assert speculative["method"] == method
            expected_drafter = DSPARK_DRAFTER if method == "dspark" else EAGLE3_DRAFTER
            assert speculative["model"] == expected_drafter
            assert speculative["num_speculative_tokens"] == k
            assert speculative["draft_tensor_parallel_size"] == 1
            if method == "dspark":
                assert speculative["attention_backend"] == "FLASH_ATTN"
            else:
                assert "attention_backend" not in speculative
        if graph_profile == "expanded_2048":
            compilation = kwargs.get("compilation_config")
            assert isinstance(compilation, dict)
            assert compilation.get("cudagraph_capture_sizes") == expected_capture_sizes(
                arm
            )
        composed[arm] = {
            "base_arm": base_arm,
            "graph_profile": graph_profile,
            "cudagraph_mode_source": "official-performance-recipe",
            "num_speculative_tokens": k,
        }

    print(json.dumps(composed, sort_keys=True))


if __name__ == "__main__":
    main()
