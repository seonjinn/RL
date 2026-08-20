import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]


def _load_resume_contract() -> ModuleType:
    module_path = EXPERIMENT_DIR / "resume_contract.py"
    spec = importlib.util.spec_from_file_location("dflash_resume_contract", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("previous_step", "target_step"), [(1, 350), (350, 700), (700, 1000)]
)
def test_resume_endpoint_contract(previous_step: int, target_step: int) -> None:
    contract = _load_resume_contract()

    assert contract.validate_transition(previous_step, target_step) == (
        previous_step,
        target_step,
    )


@pytest.mark.parametrize(
    ("previous_step", "target_step"), [(0, 350), (1, 700), (350, 1000), (700, 900)]
)
def test_non_chain_resume_transition_fails_loudly(
    previous_step: int, target_step: int
) -> None:
    contract = _load_resume_contract()

    with pytest.raises(ValueError, match="1 -> 350 -> 700 -> 1000"):
        contract.validate_transition(previous_step, target_step)


def _write_complete_checkpoint(checkpoint_root: Path, step: int) -> Path:
    step_dir = checkpoint_root / f"step_{step}"
    (step_dir / "policy" / "weights").mkdir(parents=True)
    (step_dir / "policy" / "optimizer").mkdir(parents=True)
    (step_dir / "train_dataloader.pt").write_bytes(b"dataloader")
    (step_dir / "training_info.json").write_text(
        json.dumps(
            {
                "current_step": step,
                "total_steps": step,
                "consumed_samples": 8 * step,
            }
        )
    )
    return step_dir


def test_checkpoint_preflight_requires_exact_complete_predecessor(
    tmp_path: Path,
) -> None:
    contract = _load_resume_contract()
    checkpoint_root = tmp_path / "checkpoints"
    expected = _write_complete_checkpoint(checkpoint_root, 350)

    assert contract.validate_checkpoint(checkpoint_root, expected_step=350) == expected


@pytest.mark.parametrize(
    "mutation",
    ["later_step", "wrong_total", "wrong_consumed", "missing_optimizer", "tmp_step"],
)
def test_checkpoint_preflight_rejects_incomplete_or_ambiguous_state(
    tmp_path: Path, mutation: str
) -> None:
    contract = _load_resume_contract()
    checkpoint_root = tmp_path / "checkpoints"
    step_dir = _write_complete_checkpoint(checkpoint_root, 350)

    if mutation == "later_step":
        _write_complete_checkpoint(checkpoint_root, 351)
    elif mutation == "wrong_total":
        info = json.loads((step_dir / "training_info.json").read_text())
        info["total_steps"] = 349
        (step_dir / "training_info.json").write_text(json.dumps(info))
    elif mutation == "wrong_consumed":
        info = json.loads((step_dir / "training_info.json").read_text())
        info["consumed_samples"] = 0
        (step_dir / "training_info.json").write_text(json.dumps(info))
    elif mutation == "missing_optimizer":
        (step_dir / "policy" / "optimizer").rmdir()
    else:
        (checkpoint_root / "tmp_step_351").mkdir()

    with pytest.raises(ValueError):
        contract.validate_checkpoint(checkpoint_root, expected_step=350)


def test_wandb_run_id_is_recovered_from_successful_gate_log(tmp_path: Path) -> None:
    contract = _load_resume_contract()
    train_log = tmp_path / "train.log"
    train_log.write_text(
        "wandb: View run at https://wandb.ai/nvidia/project/runs/abc123xyz\n"
    )

    assert contract.extract_wandb_run_id(train_log) == "abc123xyz"


def test_resume_runner_reuses_checkpoint_and_wandb_identity() -> None:
    runner = (EXPERIMENT_DIR / "run_resume_oci_hsg.sbatch").read_text()

    assert '${CHECKPOINT_DIR}' in runner
    assert 'grpo.max_num_steps=\'${TARGET_TOTAL_STEPS}\'' in runner
    assert "checkpointing.save_period=100" in runner
    assert "logger.wandb.id='${wandb_run_id}'" in runner
    assert "logger.wandb.resume=must" in runner
    assert "resume_contract.py" in runner
    assert "--expected-step '${EXPECTED_PREVIOUS_STEP}'" in runner
    assert "--expected-step '${TARGET_TOTAL_STEPS}'" in runner
