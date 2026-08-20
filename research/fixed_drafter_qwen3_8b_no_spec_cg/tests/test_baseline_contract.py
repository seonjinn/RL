import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


EXPERIMENT_DIR = Path(__file__).parents[1]
DFLASH_DIR = EXPERIMENT_DIR.parent / "fixed_drafter_qwen3_8b_dflash"


def _load_module():
    module_path = EXPERIMENT_DIR / "contract.py"
    spec = importlib.util.spec_from_file_location("no_spec_contract", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_baseline_changes_only_specdec_and_provenance() -> None:
    contract = _load_module()

    result = contract.validate_config(EXPERIMENT_DIR / "config.yaml")

    assert result == {
        "method": "no-specdec",
        "target_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "speculative_config": None,
        "enforce_eager": False,
        "cudagraph_mode": "PIECEWISE",
        "cudagraph_capture_sizes": [
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
        ],
        "seed": 42,
        "dataset": "DAPOMath17K",
        "prompts_per_step": 8,
        "generations_per_prompt": 4,
        "global_batch_size": 32,
        "micro_batch_size": 1,
        "training_tp": 2,
        "training_pp": 1,
        "training_cp": 1,
        "generation_tp": 1,
        "max_new_tokens": 1024,
        "max_total_sequence_length": 4096,
        "wandb_project": "sna-nemo-rl-fixed-drafter",
        "wandb_group": "qwen3-8b-dflash-fixed-drafter-k-sweep",
    }


def test_runners_keep_one_horizon_and_one_wandb_identity() -> None:
    gate = (EXPERIMENT_DIR / "run_oci_hsg.sbatch").read_text()
    resume = (EXPERIMENT_DIR / "run_resume_oci_hsg.sbatch").read_text()

    for runner in (gate, resume):
        assert "grpo.max_num_steps='${TRAINING_HORIZON_STEPS}'" in runner
        assert "checkpointing.checkpoint_must_save_by" in runner
        assert "Capturing CUDA graphs (PIECEWISE)" in runner
        assert "Graph capturing finished" in runner
        assert "DRAFTER_SNAPSHOT" not in runner
        assert "logger.wandb.config.stage_steps='${TRAINING_HORIZON_STEPS}'" in runner
    assert "+logger.wandb.id='${wandb_run_id}'" in resume
    assert "+logger.wandb.resume=must" in resume


def test_submitter_builds_one_gate_and_arm_local_resume_chain(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    call_log = tmp_path / "sbatch-calls.txt"
    counter = tmp_path / "counter.txt"
    counter.write_text("9100")
    (fake_bin / "sbatch").write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        'echo "$*" >> "${SBATCH_CALL_LOG}"\n'
        'if [[ " $* " == *" --test-only "* ]]; then exit 0; fi\n'
        'next=$(( $(cat "${SBATCH_COUNTER}") + 1 ))\n'
        'echo "${next}" > "${SBATCH_COUNTER}"\n'
        'echo "${next}"\n'
    )
    (fake_bin / "sbatch").chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "SBATCH_CALL_LOG": str(call_log),
        "SBATCH_COUNTER": str(counter),
        "REMOTE_REPO": str(tmp_path / "repo"),
        "EXPECTED_HEAD": "a" * 40,
        "RUN_ROOT": str(tmp_path / "runs"),
        "CONTAINER": str(tmp_path / "container.sqsh"),
        "TARGET_SNAPSHOT": str(tmp_path / "target"),
        "WANDB_API_KEY": "test-only-placeholder",
    }
    result = subprocess.run(
        ["bash", str(EXPERIMENT_DIR / "submit_chain.sh")],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    calls = call_log.read_text().splitlines()
    assert len(calls) == 10
    assert "--dependency" not in calls[0]
    assert "--dependency" not in calls[1]
    for call_index, dependency in ((2, 9101), (4, 9102), (6, 9103), (8, 9104)):
        assert f"--dependency=afterok:{dependency}" in calls[call_index]


@pytest.mark.parametrize(
    "bad_key", ["speculative_config", "enforce_eager", "target_revision"]
)
def test_contract_rejects_specdec_or_eager_baseline(
    tmp_path: Path, bad_key: str
) -> None:
    config = (EXPERIMENT_DIR / "config.yaml").read_text()
    if bad_key == "speculative_config":
        config = config.replace("speculative_config: null", "speculative_config: {}")
    elif bad_key == "enforce_eager":
        config = config.replace(
            "  generation:\n    vllm_kwargs:",
            "  generation:\n    vllm_cfg:\n      enforce_eager: true\n    vllm_kwargs:",
        )
    else:
        config = config.replace(
            "experiment:\n",
            "experiment:\n  target_revision: wrong-target-revision\n",
        )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config)

    with pytest.raises(ValueError, match=bad_key):
        _load_module().validate_config(
            config_path,
            reference_path=DFLASH_DIR / "config_k5.yaml",
        )
