import subprocess
import shlex
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


@pytest.mark.parametrize("site", ["lyris", "ptyche"])
def test_conversion_checkpoint_is_shared_separately_from_local_cache(site: str) -> None:
    result = subprocess.run(["bash", str(LAUNCHER), site, "--plan"], capture_output=True, text=True, check=True)
    exported = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    assert exported.get("NRL_MEGATRON_CHECKPOINT_DIR", "").startswith("/lustre/")
    assert exported.get("HF_HOME", "").startswith("/raid/scratch/")
    assert exported["NRL_MEGATRON_CHECKPOINT_DIR"] != exported["HF_HOME"]


def test_generated_preflight_python_retains_valid_quoting() -> None:
    result = subprocess.run(["bash", str(LAUNCHER), "lyris", "--plan"], capture_output=True, text=True, check=True)
    commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith("/usr/local/bin/python-MegatronPolicyWorker -c ")]
    assert len(commands) == 1
    compile(commands[0][2], "<preflight>", "exec")
