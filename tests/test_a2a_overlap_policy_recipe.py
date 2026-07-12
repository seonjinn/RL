from pathlib import Path

from omegaconf import OmegaConf

from nemo_rl.models.policy import MegatronConfig
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECIPE_NAME = "grpo-qwen3-30ba3b-1n4g-megatron-mxfp8-a2a-overlap"
RECIPE_PATH = (
    PROJECT_ROOT / "examples/configs/recipes/llm/performance" / f"{RECIPE_NAME}.yaml"
)

register_omegaconf_resolvers()


def test_ep_a2a_overlap_fields_are_typed_policy_config() -> None:
    optional_keys = MegatronConfig.__optional_keys__

    assert "overlap_moe_expert_parallel_comm" in optional_keys
    assert "high_priority_a2a_comm_stream" in optional_keys
    assert "delay_wgrad_compute" in optional_keys


def test_ep_a2a_overlap_smoke_recipe_contract() -> None:
    config = OmegaConf.to_container(load_config(RECIPE_PATH), resolve=True)
    assert isinstance(config, dict)

    policy = config["policy"]
    megatron_cfg = policy["megatron_cfg"]
    cluster = config["cluster"]

    assert policy["model_name"] == "Qwen/Qwen3-30B-A3B"
    assert cluster["num_nodes"] == 1
    assert cluster["gpus_per_node"] == 4
    assert policy["precision"] == "bfloat16"
    assert policy["train_global_batch_size"] >= 8
    assert policy["train_micro_batch_size"] == 1
    assert policy["sequence_packing"]["enabled"] is False
    assert policy["dynamic_batching"]["enabled"] is False
    assert not policy.get("router_replay", {}).get("enabled", False)
    assert not policy.get("draft", {}).get("enabled", False)
    assert megatron_cfg["tensor_model_parallel_size"] == 1
    assert megatron_cfg["pipeline_model_parallel_size"] == 1
    assert megatron_cfg["context_parallel_size"] == 1
    assert megatron_cfg["expert_model_parallel_size"] == 4
    assert megatron_cfg["moe_token_dispatcher_type"] == "alltoall"
    assert megatron_cfg["moe_shared_expert_overlap"] is False
    assert megatron_cfg["activation_checkpointing"] is False
    assert megatron_cfg["mtp_num_layers"] == 0
    assert megatron_cfg["defer_fp32_logits"] is False
    assert megatron_cfg["use_fused_linear_logprobs"] is False
    assert megatron_cfg["overlap_moe_expert_parallel_comm"] is True
    assert megatron_cfg["high_priority_a2a_comm_stream"] is True
    assert megatron_cfg["delay_wgrad_compute"] is True

    data_parallel_size = (
        cluster["num_nodes"]
        * cluster["gpus_per_node"]
        // (
            megatron_cfg["tensor_model_parallel_size"]
            * megatron_cfg["pipeline_model_parallel_size"]
            * megatron_cfg["context_parallel_size"]
        )
    )
    num_local_microbatches = (
        policy["train_global_batch_size"]
        // data_parallel_size
        // policy["train_micro_batch_size"]
    )
    assert num_local_microbatches >= 2

    rollout_batch_size = (
        config["grpo"]["num_prompts_per_step"]
        * config["grpo"]["num_generations_per_prompt"]
    )
    assert rollout_batch_size == policy["train_global_batch_size"]
