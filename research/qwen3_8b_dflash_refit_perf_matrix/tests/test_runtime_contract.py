import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).parents[3]
EXPERIMENT_DIR = ROOT / "research/qwen3_8b_dflash_refit_perf_matrix"


def _module(name: str) -> ModuleType:
    path = EXPERIMENT_DIR / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("shape", "gbs", "mbs", "prompts"),
    [
        ("gbs32_mbs1", 32, 1, 8),
        ("gbs64_mbs1", 64, 1, 16),
        ("gbs64_mbs2", 64, 2, 16),
    ],
)
def test_each_shape_expands_to_a_matched_fixed_online_pair(
    shape: str, gbs: int, mbs: int, prompts: int
) -> None:
    contract = _module("runtime_contract.py")

    fixed = contract.resolve_cell(f"{shape}_fixed")
    online = contract.resolve_cell(f"{shape}_online")

    assert (fixed.gbs, fixed.mbs, fixed.prompts, fixed.generations) == (
        gbs,
        mbs,
        prompts,
        4,
    )
    assert (online.gbs, online.mbs, online.prompts, online.generations) == (
        gbs,
        mbs,
        prompts,
        4,
    )
    assert fixed.arm == "fixed"
    assert online.arm == "online"
    assert fixed.logprob_mbs == online.logprob_mbs == 1


def test_runtime_overrides_keep_logprob_mbs_one_and_bind_measurement_window() -> None:
    contract = _module("runtime_contract.py")

    overrides = contract.runtime_overrides(
        contract.resolve_cell("gbs64_mbs2_online"),
        target_snapshot="/lustre/target/b968",
        drafter_snapshot="/lustre/draft/9b414",
        scratch_root="/raid/scratch/matrix/123/online",
        wandb_run_id="run123",
        expected_head="a" * 40,
    )

    assert "grpo.max_num_steps=30" in overrides
    assert "grpo.num_prompts_per_step=16" in overrides
    assert "grpo.num_generations_per_prompt=4" in overrides
    assert "policy.train_global_batch_size=64" in overrides
    assert "policy.train_micro_batch_size=2" in overrides
    assert "policy.logprob_batch_size=1" in overrides
    assert "policy.sequence_packing.enabled=false" in overrides
    assert "++logger.wandb.config.performance_window=steps_5_through_29" in overrides


def test_unknown_cell_fails_before_scheduling() -> None:
    contract = _module("runtime_contract.py")

    with pytest.raises(ValueError, match="Unsupported matrix cell"):
        contract.resolve_cell("gbs128_mbs8_online")
