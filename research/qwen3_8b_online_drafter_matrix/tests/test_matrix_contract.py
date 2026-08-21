import importlib.util
from pathlib import Path
import sys

import pytest
import yaml

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


ROOT = Path(__file__).parents[1]


def _contract():
    spec = importlib.util.spec_from_file_location(
        "matrix_contract", ROOT / "runtime_contract.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("arm", "draft_enabled", "method", "k"),
    [
        ("baseline", False, None, None),
        ("dflash-fixed-k5", False, "dflash", 5),
        ("dflash-fixed-k7", False, "dflash", 7),
        ("dflash-k5", True, "dflash", 5),
        ("dflash-k7", True, "dflash", 7),
    ],
)
def test_arm_configs_are_matched(
    arm: str, draft_enabled: bool, method: str | None, k: int | None
) -> None:
    contract = _contract()
    config = yaml.safe_load((ROOT / f"{arm}.yaml").read_text())

    contract.validate_arm_config(
        arm,
        config,
        expected_draft_enabled=draft_enabled,
        expected_method=method,
        expected_k=k,
    )


def test_runner_and_submitter_are_fail_closed() -> None:
    runner = (ROOT / "run_oci_hsg.sbatch").read_text()
    submitter = (ROOT / "submit_chain.sh").read_text()

    for marker in (
        "sequence_packing.enabled=false",
        "megatron_cfg.sequence_parallel=false",
        "train_global_batch_size=32",
        "num_prompts_per_step=8",
        "num_generations_per_prompt=4",
        "cudagraph_mode: PIECEWISE",
    ):
        assert marker in runner
    assert "sbatch --test-only" in submitter
    assert "afterok:" in submitter
    assert "segment1 04:00:00 350" in submitter
    assert "segment2 04:00:00 700" in submitter
    assert "segment3 04:00:00 1000" in submitter
    assert "sna-nemo-rl-online-drafter" in submitter


def test_unknown_arm_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported matrix arm"):
        _contract().arm_contract("unknown")


@pytest.mark.parametrize(
    ("arm", "draft_enabled", "method", "k"),
    [
        ("baseline", False, None, None),
        ("dflash-fixed-k5", False, "dflash", 5),
        ("dflash-fixed-k7", False, "dflash", 7),
        ("dflash-k5", True, "dflash", 5),
        ("dflash-k7", True, "dflash", 7),
    ],
)
def test_resolved_recipe_preserves_the_fair_matrix(
    arm: str, draft_enabled: bool, method: str | None, k: int | None
) -> None:
    register_omegaconf_resolvers()
    config = load_config(ROOT / f"{arm}.yaml")

    assert config.grpo.seed == 42
    assert config.grpo.num_prompts_per_step == 8
    assert config.grpo.num_generations_per_prompt == 4
    assert config.policy.train_global_batch_size == 32
    assert config.policy.sequence_packing.enabled is False
    assert config.policy.megatron_cfg.tensor_model_parallel_size == 2
    assert config.policy.megatron_cfg.context_parallel_size == 1
    assert config.policy.megatron_cfg.sequence_parallel is False
    assert config.policy.draft.enabled is draft_enabled
    speculative = config.policy.generation.vllm_kwargs.speculative_config
    if method is None:
        assert speculative is None
    else:
        assert speculative.method == method
        assert speculative.num_speculative_tokens == k
        assert config.policy.draft.gamma == k
    assert config.policy.generation.vllm_kwargs.compilation_config.cudagraph_mode == (
        "PIECEWISE"
    )
    assert config.data.train.dataset_name == "DAPOMath17K"
    assert config.logger.wandb.project == "sna-nemo-rl-online-drafter"
    assert config.logger.wandb.config.draft_training_enabled is draft_enabled
    assert config.logger.wandb.config.draft_refit_enabled is draft_enabled
    assert config.logger.wandb.config.speculator_type == method
    assert config.logger.wandb.config.k == k
    assert config.logger.wandb.config.sequence_packing is False
    assert config.logger.wandb.config.sequence_parallel is False
