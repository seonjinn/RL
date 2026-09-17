import subprocess
from pathlib import Path

import pytest


LAUNCHER = Path(__file__).resolve().parents[3] / "experiments/q30_dapo_concurrency_20260912/cluster_setup/submit_native_baseline.sh"


@pytest.mark.parametrize("site,partition", [("lyris", "gb200"), ("ptyche", "batch")])
def test_native_plan_preserves_recipe_and_topology(site: str, partition: str) -> None:
    result = subprocess.run(["bash", str(LAUNCHER), site, "--plan"], capture_output=True, text=True, check=True)
    assert f"--partition={partition}" in result.stdout
    for argument in ("--nodes=4", "--segment=4", "GPUS_PER_NODE=4", "grpo.max_num_steps=1", "flashinfer_trtllm", "enforce_eager=false", "/opt/nemo_rl_venv/bin/python", "cd /opt/nemo-rl"):
        assert argument in result.stdout
    assert "num_prompts_per_step=" not in result.stdout
    assert "max_num_seqs=" not in result.stdout
    assert "NRL_FORCE_REBUILD_VENVS=true" not in result.stdout
    assert "--gres" not in result.stdout
    assert "DEDICATED_RAY_HEAD=0" in result.stdout


def test_native_plan_rejects_unknown_site() -> None:
    result = subprocess.run(["bash", str(LAUNCHER), "oci", "--plan"], capture_output=True, text=True)
    assert result.returncode != 0
