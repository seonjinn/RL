import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (
    REPO_ROOT
    / "experiments"
    / "eagle3_online"
    / "submit_nemorl_online_draft_specdec.sh"
)
IMPORT_SMOKE = REPO_ROOT / "scripts" / "submit_nemorl_import_smoke.sh"


def test_precluster_dry_run_omits_gres_when_disabled() -> None:
    env = os.environ.copy()
    env.update(
        {
            "NEMO_RL_DIR": str(REPO_ROOT),
            "MODEL_LABEL": "qwen32-smoke",
            "CONFIG_FILE": "examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml",
            "TARGET_MODEL_ID": "/tmp/qwen32",
            "DRAFT_MODEL": "/tmp/pard",
            "CONTAINER": str(REPO_ROOT / "pyproject.toml"),
            "DRY_RUN": "true",
            "PARTITION": "batch",
            "ACCOUNT": "coreai_dlalgo_llm",
            "NUM_NODES": "4",
            "GPUS_PER_NODE": "4",
            "USE_GRES": "false",
            "WANDB_ENABLED": "false",
        }
    )

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    sbatch_line = next(
        line for line in result.stdout.splitlines() if line.startswith("[DRY-RUN] sbatch")
    )
    assert "--gres" not in sbatch_line
    assert "--segment 4" in sbatch_line


def test_launcher_rejects_cuda_graph_disabled_without_ablation_opt_in() -> None:
    env = os.environ.copy()
    env.update(
        {
            "NEMO_RL_DIR": str(REPO_ROOT),
            "MODEL_LABEL": "qwen32-eager-ablation",
            "CONFIG_FILE": "examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml",
            "TARGET_MODEL_ID": "/tmp/qwen32",
            "DRAFT_MODEL": "/tmp/pard",
            "CONTAINER": str(REPO_ROOT / "pyproject.toml"),
            "DRY_RUN": "true",
            "VLLM_ENFORCE_EAGER": "true",
            "WANDB_ENABLED": "false",
        }
    )

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "CUDA Graph" in result.stderr


def test_import_smoke_uses_container_venv_without_gres() -> None:
    env = os.environ.copy()
    env.update(
        {
            "REMOTE_REPO": str(REPO_ROOT),
            "CONTAINER": str(REPO_ROOT / "pyproject.toml"),
            "ACCOUNT": "coreai_dlalgo_llm",
            "PARTITION": "batch",
            "USE_GRES": "false",
            "DRY_RUN": "true",
        }
    )

    result = subprocess.run(
        ["bash", str(IMPORT_SMOKE)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--segment=1" in result.stdout
    assert "--gres" not in result.stdout
    assert (
        "/opt/nemo_rl_venv/bin/python"
        in result.stdout
    )
    assert "scripts/nemorl_import_smoke.py" in result.stdout
    assert "python\\ -c" not in result.stdout
