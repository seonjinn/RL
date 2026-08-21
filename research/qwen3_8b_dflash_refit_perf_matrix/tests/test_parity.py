import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).parents[3]
EXPERIMENT_DIR = ROOT / "research/qwen3_8b_dflash_refit_perf_matrix"
ONLINE_CONFIG = ROOT / "research/qwen3_8b_dflash_online_cp1/config.yaml"
FIXED_CONFIG = ROOT / "research/qwen3_8b_dflash_fixed_dense_control/config.yaml"


def _module(name: str) -> ModuleType:
    path = EXPERIMENT_DIR / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("shape", ["gbs32_mbs1", "gbs64_mbs1", "gbs64_mbs2"])
def test_resolved_pair_changes_only_trainer_owned_draft_state(shape: str) -> None:
    parity = _module("parity.py")

    report = parity.resolve_pair(
        shape=shape,
        online_config=ONLINE_CONFIG,
        fixed_config=FIXED_CONFIG,
        target_snapshot="/lustre/target/b968",
        drafter_snapshot="/lustre/draft/9b414",
        expected_head="a" * 40,
    )

    assert report["status"] == "passed"
    assert report["shape"] == shape
    assert report["unexpected_differences"] == []
    assert set(report["allowed_differences"]) == {
        "logger.wandb.config.ab_arm",
        "logger.wandb.config.draft_refit_enabled",
        "logger.wandb.config.draft_training_enabled",
        "logger.wandb.config.fixed_public_drafter",
        "logger.wandb.config.matrix_cell",
        "logger.wandb.name",
        "logger.wandb.tags",
        "policy.draft.enabled",
        "policy.draft.optimizer",
    }
    assert len(report["common_fingerprint"]) == 64


def test_proof_validation_rejects_a_proof_for_another_shape(tmp_path: Path) -> None:
    parity = _module("parity.py")
    proof = tmp_path / "resolved-parity.json"
    proof.write_text(
        json.dumps(
            {
                "status": "passed",
                "shape": "gbs32_mbs1",
                "expected_head": "a" * 40,
                "container_sha256": "b" * 64,
                "unexpected_differences": [],
                "common_fingerprint": "c" * 64,
            }
        )
    )

    with pytest.raises(ValueError, match="shape"):
        parity.validate_proof(
            proof,
            shape="gbs64_mbs1",
            expected_head="a" * 40,
            container_sha256="b" * 64,
        )


def test_proof_validation_rejects_an_incomplete_allowed_delta(tmp_path: Path) -> None:
    parity = _module("parity.py")
    proof = tmp_path / "resolved-parity.json"
    proof.write_text(
        json.dumps(
            {
                "status": "passed",
                "shape": "gbs32_mbs1",
                "expected_head": "a" * 40,
                "container_sha256": "b" * 64,
                "allowed_differences": ["policy.draft.enabled"],
                "unexpected_differences": [],
                "common_fingerprint": "c" * 64,
            }
        )
    )

    with pytest.raises(ValueError, match="allowed_differences"):
        parity.validate_proof(
            proof,
            shape="gbs32_mbs1",
            expected_head="a" * 40,
            container_sha256="b" * 64,
        )
