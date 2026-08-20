from pathlib import Path

import pytest

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from research.qwen3_8b_dflash_online_cp1.validate_gate import (
    validate_gate,
    validate_validation_history,
)


ROOT = Path(__file__).parents[3]
CONFIG = ROOT / "research/qwen3_8b_dflash_online_cp1/config.yaml"
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
UPDATE_PROOF = (
    "draft_update_probe=complete grad_l2=0.25 "
    "checksum_before=10 checksum_after=10.125 delta=0.125\n"
)
REFIT_PROOF = (
    "draft_refit_manifest=draft_count=17\n"
    "draft_refit_load=complete\n"
    "draft_refit_finalize=complete\n"
)
TWO_STEP_PROOF = 2 * (UPDATE_PROOF + REFIT_PROOF)


def test_online_dflash_recipe_resolves_to_the_cp1_gate_contract() -> None:
    register_omegaconf_resolvers()
    config = load_config(CONFIG)

    assert config.grpo.max_num_steps == 1000
    assert config.grpo.seed == 42
    assert config.grpo.num_prompts_per_step == 8
    assert config.grpo.num_generations_per_prompt == 4
    assert config.grpo.val_at_end is True
    assert config.policy.model_name == "Qwen/Qwen3-8B"
    assert config.policy.max_total_sequence_length == 4096
    assert config.policy.train_global_batch_size == 32
    assert config.policy.train_micro_batch_size == 1
    assert config.policy.megatron_cfg.tensor_model_parallel_size == 2
    assert config.policy.megatron_cfg.pipeline_model_parallel_size == 1
    assert config.policy.megatron_cfg.context_parallel_size == 1
    assert config.policy.megatron_cfg.sequence_parallel is False
    assert config.policy.draft.speculator_type == "dflash"
    assert config.policy.draft.enabled is True
    assert config.policy.draft.gamma == 7
    assert config.policy.draft.loss_weight == 1.0
    assert config.policy.draft.optimizer.lr == 1.0e-6
    assert config.policy.draft.optimizer.min_lr == 1.0e-6
    assert config.policy.draft.optimizer.weight_decay == 0.1
    assert config.policy.draft.update_probe_enabled is True
    assert config.policy.generation.max_new_tokens == 1024
    assert config.policy.generation.vllm_cfg.tensor_parallel_size == 1
    assert config.policy.generation.vllm_cfg.enforce_eager is False
    assert config.policy.generation.vllm_kwargs.speculative_config.method == "dflash"
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
    assert config.data.train.dataset_name == "DAPOMath17K"
    assert config.logger.wandb_enabled is True
    assert config.logger.wandb.project == "nemo-rl-specdec-eval"
    assert config.logger.wandb.config.oracle_run_id == "tbosl9uz"
    assert config.logger.wandb.config.draft_training_enabled is True
    assert config.logger.wandb.config.draft_refit_enabled is True
    assert config.cluster.num_nodes == 1
    assert config.cluster.gpus_per_node == 4


def test_gate_accepts_complete_online_draft_evidence() -> None:
    validate_gate(
        {
            "train/draft_grad_norm": {"1": 0.25},
            "train/draft_loss": {"1": 1.5},
            "train/vllm/spec_acceptance_rate": {"1": 0.41},
        },
        TWO_STEP_PROOF,
    )


def test_gate_rejects_one_step_without_post_refit_generation() -> None:
    metrics = {
        "train/draft_grad_norm": {"1": 0.25},
        "train/draft_loss": {"1": 1.5},
        "train/vllm/spec_acceptance_rate": {"1": 0.41},
    }
    log_text = UPDATE_PROOF + REFIT_PROOF

    with pytest.raises(RuntimeError, match="at least two"):
        validate_gate(metrics, log_text)


def test_gate_requires_initial_and_final_validation_metrics() -> None:
    validate_validation_history(
        [
            {"_step": 0, "validation/accuracy": 0.0, "validation/avg_length": 128},
            {"_step": 2, "validation/accuracy": 0.25, "validation/avg_length": 256},
        ]
    )

    with pytest.raises(RuntimeError, match="initial and final"):
        validate_validation_history(
            [
                {
                    "_step": 0,
                    "validation/accuracy": 0.0,
                    "validation/avg_length": 128,
                }
            ]
        )


@pytest.mark.parametrize(
    ("metrics", "log_text", "message"),
    [
        (
            {
                "train/draft_grad_norm": {"1": 0.0},
                "train/draft_loss": {"1": 1.5},
                "train/vllm/spec_acceptance_rate": {"1": 0.41},
            },
            TWO_STEP_PROOF,
            "draft_grad_norm",
        ),
        (
            {
                "train/draft_grad_norm": {"1": 0.25},
                "train/draft_loss": {"1": 1.5},
                "train/vllm/spec_acceptance_rate": {"1": 0.41},
            },
            2
            * (
                "draft_update_probe=complete grad_l2=0.25 "
                "checksum_before=10 checksum_after=10 delta=0\n" + REFIT_PROOF
            ),
            "draft update",
        ),
        (
            {
                "train/draft_grad_norm": {"1": 0.25},
                "train/draft_loss": {"1": 1.5},
                "train/vllm/spec_acceptance_rate": {"1": 0.41},
            },
            2
            * (
                UPDATE_PROOF + "draft_refit_manifest=draft_count=17\n"
                "draft_refit_load=complete\n"
            ),
            "draft_refit_finalize",
        ),
        (
            {
                "train/draft_grad_norm": {"1": 0.25},
                "train/draft_loss": {"1": 1.5},
                "train/vllm/spec_acceptance_rate": {"1": 0.41},
            },
            2
            * (
                UPDATE_PROOF + "draft_refit_manifest=draft_count=17\n"
                "draft_refit_finalize=complete\n"
            ),
            "draft_refit_load",
        ),
    ],
)
def test_gate_rejects_missing_online_draft_evidence(
    metrics: dict[str, dict[str, float]], log_text: str, message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        validate_gate(metrics, log_text)
