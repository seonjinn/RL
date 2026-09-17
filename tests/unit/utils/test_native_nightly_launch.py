import subprocess
import shlex
import os
import sys
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


def test_preflight_after_launcher_scrubs_slurm_environment(tmp_path: Path) -> None:
    result = subprocess.run(["bash", str(LAUNCHER), "lyris", "--plan"], capture_output=True, text=True, check=True)
    line = next(line for line in result.stdout.splitlines() if line.startswith("/usr/local/bin/python-MegatronPolicyWorker -c "))
    code = shlex.split(line)[2]
    for node in range(4):
        (tmp_path / f".mount-check-node{node}").touch()
    # Only the unavailable GPU library import is replaced; execute the real emitted preflight.
    bootstrap = "import os,sys,types; m=types.ModuleType('nemo_rl.models.policy.utils'); m.get_megatron_checkpoint_dir=lambda: os.environ['NRL_MEGATRON_CHECKPOINT_DIR']; sys.modules['nemo_rl.models.policy.utils']=m; "
    clean_env = {key: value for key, value in os.environ.items() if not key.startswith("SLURM_")}
    clean_env["NRL_MEGATRON_CHECKPOINT_DIR"] = str(tmp_path)
    checked = subprocess.run([sys.executable, "-c", bootstrap + code], env=clean_env, capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr
