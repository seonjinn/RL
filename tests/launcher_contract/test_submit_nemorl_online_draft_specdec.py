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
ASSET_STAGE = REPO_ROOT / "scripts" / "submit_hf_snapshot_stage.sh"


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


def test_launcher_forces_short_node_local_ray_temp_root() -> None:
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
            "TMPDIR": "/lustre/a/path/that/is/too/long/for/ray/unix/sockets",
            "RAY_TEMP_ROOT": "/lustre/another/long/ray/temp/root",
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
    assert "[DRY-RUN] ray-temp-root=/tmp" in result.stdout


def test_launcher_passes_short_ray_temp_root_to_sbatch(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch = bin_dir / "sbatch"
    sbatch.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'STUB_TMPDIR=%s\\n' \"${TMPDIR-}\"\n"
        "printf 'STUB_RAY_TMPDIR=%s\\n' \"${RAY_TMPDIR-}\"\n"
    )
    sbatch.chmod(0o755)
    target = tmp_path / "qwen32"
    target.mkdir()

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "NEMO_RL_DIR": str(REPO_ROOT),
            "MODEL_LABEL": "qwen32-smoke",
            "CONFIG_FILE": "examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml",
            "TARGET_MODEL_ID": str(target),
            "DRAFT_MODEL": str(target),
            "CONTAINER": str(REPO_ROOT / "pyproject.toml"),
            "BASE_LOG_DIR": str(tmp_path / "logs"),
            "RUN_CACHE_ROOT": str(tmp_path / "cache"),
            "NRL_MEGATRON_CHECKPOINT_DIR": str(tmp_path / "checkpoints"),
            "DRY_RUN": "false",
            "TMPDIR": "/lustre/a/path/that/is/too/long/for/ray/unix/sockets",
            "RAY_TEMP_ROOT": "/lustre/another/long/ray/temp/root",
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
    assert "STUB_TMPDIR=/tmp" in result.stdout
    assert "STUB_RAY_TMPDIR=/tmp" in result.stdout


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


def test_launcher_rejects_unsupported_pard_draft_tp_mismatch() -> None:
    env = os.environ.copy()
    env.update(
        {
            "NEMO_RL_DIR": str(REPO_ROOT),
            "MODEL_LABEL": "qwen32-pard-tp-mismatch",
            "CONFIG_FILE": "examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml",
            "TARGET_MODEL_ID": "/tmp/qwen32",
            "DRAFT_MODEL": "/tmp/pard",
            "CONTAINER": str(REPO_ROOT / "pyproject.toml"),
            "DRY_RUN": "true",
            "DRAFT_FORMAT": "pard",
            "SPECDEC_METHOD": "draft_model",
            "POLICY_DRAFT_ENABLED": "false",
            "TARGET_TP": "2",
            "DRAFT_TP": "1",
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
    assert "target TP=2" in result.stderr
    assert "draft TP=1" in result.stderr


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


def test_hf_asset_stage_uses_lustre_cache_without_gres() -> None:
    env = os.environ.copy()
    env.update(
        {
            "REMOTE_REPO": str(REPO_ROOT),
            "CONTAINER": str(REPO_ROOT / "pyproject.toml"),
            "ACCOUNT": "coreai_dlalgo_llm",
            "PARTITION": "batch",
            "USE_GRES": "false",
            "HF_HOME": "/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home",
            "MODEL_IDS": "amd/PARD-Qwen3-0.6B,RedHatAI/Qwen3-32B-speculator.eagle3",
            "DRY_RUN": "true",
        }
    )

    result = subprocess.run(
        ["bash", str(ASSET_STAGE)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--segment=1" in result.stdout
    assert "--gres" not in result.stdout
    assert "scripts/stage_hf_snapshots.py" in result.stdout
    assert "amd/PARD-Qwen3-0.6B" in result.stdout
