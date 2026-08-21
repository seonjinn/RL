import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[3]
PROFILE_DIR = ROOT / "research/qwen3_8b_dflash_policy_profile"


def _module(name: str):
    path = PROFILE_DIR / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_profile_runner_is_policy_only_and_bounded() -> None:
    script = (PROFILE_DIR / "run_oci_hsg.sbatch").read_text()

    assert "NRL_NSYS_WORKER_PATTERNS=megatron_policy_worker" in script
    assert "NRL_NSYS_PROFILE_STEP_RANGE=3:6" in script
    assert "NRL_NSYS_EXTRA_OPTIONS=" in script
    assert "--write-nsys-options" in script
    assert r"NRL_NSYS_EXTRA_OPTIONS=\"\$(cat" in script
    assert "grpo.max_num_steps=6" in script
    assert "logger.wandb_enabled=false" in script
    assert "logger.tensorboard_enabled=false" in script
    assert "checkpointing.enabled=false" in script
    assert "vllm_generation_worker" not in script


def test_profile_runner_preserves_mars_storage_layout() -> None:
    script = (PROFILE_DIR / "run_oci_hsg.sbatch").read_text()

    assert '[[ "${REMOTE_REPO}" == /home/* ]]' in script
    assert '[[ "${FINAL_DIR}" == /lustre/* ]]' in script
    assert 'readonly scratch_root="/raid/scratch/p/${SLURM_JOB_ID}"' in script
    assert 'readonly ray_root="/raid/scratch/r/${SLURM_JOB_ID}"' in script
    assert 'find "${scratch_root}" "${ray_root}"' in script
    assert "*.nsys-rep" in script


@pytest.mark.parametrize(
    ("arm", "config_name", "probe_enabled"),
    [
        ("fixed-control", "qwen3_8b_dflash_fixed_dense_control", False),
        ("online-current", "qwen3_8b_dflash_online_cp1", True),
        ("online-probe-off", "qwen3_8b_dflash_online_cp1", False),
    ],
)
def test_arm_contract(
    arm: str,
    config_name: str,
    probe_enabled: bool,
) -> None:
    contract = _module("runtime_contract.py")

    profile = contract.resolve_arm(arm)

    assert profile.config_name == config_name
    assert profile.update_probe_enabled is probe_enabled


def test_unknown_arm_fails_loudly() -> None:
    contract = _module("runtime_contract.py")

    with pytest.raises(ValueError, match="Unsupported profile arm"):
        contract.resolve_arm("online-typo")


def test_profile_receipt_requires_reports(tmp_path: Path) -> None:
    contract = _module("runtime_contract.py")

    with pytest.raises(RuntimeError, match="no Nsight reports"):
        contract.validate_profile_receipt(tmp_path)

    (tmp_path / "policy-rank-0.nsys-rep").write_bytes(b"profile")
    reports = contract.validate_profile_receipt(tmp_path)

    assert reports == [tmp_path / "policy-rank-0.nsys-rep"]


def test_nsys_options_are_serialized_as_valid_json(tmp_path: Path) -> None:
    contract = _module("runtime_contract.py")
    output_path = tmp_path / "nsys-options.json"

    contract.write_nsys_options(output_path, Path("/raid/scratch/p/123/policy_%p"))

    assert json.loads(output_path.read_text()) == {
        "cpuctxsw": "none",
        "cuda-graph-trace": "node",
        "o": "/raid/scratch/p/123/policy_%p",
    }
