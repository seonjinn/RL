import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest


EXPERIMENT_DIR = Path(__file__).parents[1]


def _contract():
    path = EXPERIMENT_DIR / "resume_contract.py"
    spec = importlib.util.spec_from_file_location("online_resume_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkpoint(root: Path, step: int) -> None:
    step_dir = root / f"step_{step}"
    weights = step_dir / "policy" / "weights"
    weights.mkdir(parents=True)
    (weights / "latest_train_state.pt").write_bytes(b"state")
    (step_dir / "train_dataloader.pt").write_bytes(b"loader")
    (step_dir / "training_info.json").write_text(
        json.dumps(
            {"current_step": step, "total_steps": step, "consumed_samples": 8 * step}
        )
    )
    (step_dir / "config.yaml").write_text("grpo:\n  max_num_steps: 1000\n")


def test_checkpoint_keeps_one_horizon_and_reaches_declared_milestone(
    tmp_path: Path,
) -> None:
    contract = _contract()
    root = tmp_path / "checkpoints"
    _checkpoint(root, 117)

    assert contract.latest_step(root) == 117
    contract.validate_checkpoint(root, expected_step=117)
    contract.validate_progress(10, 117, required_min_step=100)


@pytest.mark.parametrize("current", [11, 99])
def test_progress_fails_when_a_segment_misses_its_milestone(current: int) -> None:
    with pytest.raises(ValueError, match="milestone"):
        _contract().validate_progress(10, current, required_min_step=100)


def test_progress_requires_forward_progress() -> None:
    with pytest.raises(ValueError, match="invalid progress"):
        _contract().validate_progress(10, 10, required_min_step=100)


def test_manifest_binds_fresh_wandb_and_exact_composition(tmp_path: Path) -> None:
    contract = _contract()
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    manifest_path = tmp_path / "gate-manifest.json"
    identity = {
        "git_sha": "a" * 40,
        "checkpoint_root": checkpoint_root,
        "wandb_run_id": "fresh123",
        "target_revision": "b968",
        "drafter_revision": "9b414",
        "container_sha256": "6940",
    }

    contract.write_manifest(manifest_path, **identity)
    manifest = contract.validate_manifest(
        manifest_path,
        **{key: value for key, value in identity.items() if key != "wandb_run_id"},
    )
    assert manifest["wandb_run_id"] == "fresh123"
    assert manifest["oracle_run_id"] == "tbosl9uz"


def test_science_chain_uses_one_fresh_run_after_the_smoke_gate(
    tmp_path: Path,
) -> None:
    sbatch_log = tmp_path / "sbatch.log"
    sbatch = tmp_path / "sbatch"
    sbatch.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$SBATCH_CALL_LOG"\n'
        'case " $* " in *" --test-only "*) echo forecast >&2 ;; *) echo 123 ;; esac\n'
    )
    sbatch.chmod(0o755)
    root = EXPERIMENT_DIR.parents[1]
    environment = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "SBATCH_CALL_LOG": str(sbatch_log),
        "REMOTE_REPO": str(root),
        "EXPECTED_HEAD": "b" * 40,
        "SOURCE_GATE_SHA": "a" * 40,
        "FINAL_DIR": "/lustre/fake-online-science",
        "CONTAINER": "/lustre/fake.sqsh",
        "TARGET_SNAPSHOT": "/lustre/target/b968",
        "DRAFTER_SNAPSHOT": "/lustre/draft/9b414",
        "SBATCH_ACCOUNT": "test-account",
        "WANDB_API_KEY": "test-only-placeholder",
    }

    result = subprocess.run(
        ["bash", EXPERIMENT_DIR / "submit_chain.sh", "science", "6379582"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    calls = sbatch_log.read_text().splitlines()
    assert len(calls) == 6
    run_ids = {
        field.removeprefix("WANDB_RUN_ID=")
        for call in calls
        for field in call.split(",")
        if field.startswith("WANDB_RUN_ID=")
    }
    assert len(run_ids) == 1
    assert "afterok:6379582" in calls[0]
    assert "WANDB_RESUME=allow" in calls[0]
    assert all("WANDB_RESUME=must" in call for call in calls[2:])
    assert all("SOURCE_GATE_SHA=" + "a" * 40 in call for call in calls)
    assert all("WANDB_PROJECT=sna-nemo-rl-online-drafter" in call for call in calls)
    assert "wandb.ai/nvidia/sna-nemo-rl-online-drafter/runs/" in result.stdout
