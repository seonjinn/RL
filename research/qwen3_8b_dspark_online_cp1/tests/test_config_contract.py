from pathlib import Path

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


ROOT = Path(__file__).parents[3]
CONFIG = ROOT / "research/qwen3_8b_dspark_online_cp1/config.yaml"
CAPTURE_SIZES = [
    1,
    2,
    4,
    6,
    8,
    10,
    12,
    16,
    18,
    20,
    24,
    28,
    30,
    32,
    36,
    40,
    42,
    48,
    50,
    56,
    60,
    64,
    70,
    80,
    96,
    128,
    160,
    192,
    224,
    256,
    288,
    320,
]


def test_online_dspark_config_matches_cp1_k7_contract() -> None:
    register_omegaconf_resolvers()
    config = load_config(CONFIG)

    assert config.grpo.max_num_steps == 1000
    assert config.grpo.seed == 42
    assert config.grpo.num_prompts_per_step == 8
    assert config.grpo.num_generations_per_prompt == 4
    assert config.policy.model_name == "Qwen/Qwen3-8B"
    assert config.policy.max_total_sequence_length == 4096
    assert config.policy.train_global_batch_size == 32
    assert config.policy.train_micro_batch_size == 1
    assert config.policy.sequence_packing.enabled is False
    assert config.policy.megatron_cfg.tensor_model_parallel_size == 2
    assert config.policy.megatron_cfg.pipeline_model_parallel_size == 1
    assert config.policy.megatron_cfg.context_parallel_size == 1
    assert config.policy.megatron_cfg.sequence_parallel is False
    assert config.policy.draft.speculator_type == "dspark"
    assert config.policy.draft.enabled is True
    assert config.policy.draft.block_size == 7
    assert config.policy.draft.draft_vocab_size is None
    assert config.policy.draft.markov_rank == 256
    assert config.policy.draft.update_probe_enabled is True
    assert config.policy.generation.max_new_tokens == 1024
    assert config.policy.generation.vllm_cfg.tensor_parallel_size == 1
    assert config.policy.generation.vllm_cfg.enforce_eager is False
    assert config.policy.generation.vllm_kwargs.speculative_config.method == "dspark"
    assert (
        config.policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens
        == 7
    )
    assert (
        config.policy.generation.vllm_kwargs.compilation_config.cudagraph_mode
        == "PIECEWISE"
    )
    assert (
        config.policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes
        == CAPTURE_SIZES
    )
    assert config.data.max_input_seq_length == 2048
    assert config.data.train.dataset_name == "DAPOMath17K"
    assert config.data.train.seed == 42
    assert config.logger.wandb.project == "sna-nemo-rl-online-drafter"
    assert config.logger.wandb.config.draft_training_enabled is True
    assert config.logger.wandb.config.draft_refit_enabled is True
    assert config.cluster.num_nodes == 1
    assert config.cluster.gpus_per_node == 4
