import os
import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPE = (
    REPO_ROOT
    / "examples"
    / "configs"
    / "recipes"
    / "llm"
    / "performance"
    / "grpo-qwen3-8b-2n4g.yaml"
)
LAUNCHER = (
    REPO_ROOT
    / "experiments"
    / "vllm_025_qwen8_dflash"
    / "submit_qwen8_dflash_perfcfg_lyris.sh"
)


def _dry_run(variant: str, model_profile: str = "qwen8") -> str:
    env = os.environ.copy()
    env.update(
        {
            "MODE": "dry-run",
            "REPO_DIR": str(REPO_ROOT),
            "RUN_TAG": "contract-test",
            "VARIANT": variant,
            "MODEL_PROFILE": model_profile,
        }
    )
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        check=True,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    ).stdout.replace("\\", "")


def test_qwen8_recipe_preserves_8b_performance_shape() -> None:
    config = yaml.safe_load(RECIPE.read_text())

    assert config["defaults"] == "./grpo-llama3.1-8b-instruct-2n4g.yaml"
    assert config["policy"]["model_name"] == "Qwen/Qwen3-8B"
    assert config["policy"]["tokenizer"]["name"] == "Qwen/Qwen3-8B"
    assert config["policy"]["generation"]["stop_token_ids"] is None
    assert config["checkpointing"]["checkpoint_dir"].endswith(
        "grpo-qwen3-8b-2n4g"
    )
    assert "cluster" not in config


def test_baseline_uses_qwen8_performance_recipe_without_specdec() -> None:
    output = _dry_run("baseline")

    assert "grpo-qwen3-8b-2n4g.yaml" in output
    assert "grpo.max_num_steps=20" in output
    assert "checkpointing.enabled=false" in output
    assert "policy.generation.vllm_cfg.enforce_eager=false" in output
    assert "policy.generation.temperature=1.0" in output
    assert "policy.generation.top_p=1.0" in output
    assert "logger.wandb_enabled=true" in output
    assert "NRL_MEGATRON_CHECKPOINT_DIR=" in output
    assert "nemo_rl-v025-dflash-perfcfg-20260715" in output
    assert ".secrets/wandb_api_key" in output
    assert 'export WANDB_API_KEY="$(< "${WANDB_API_KEY_FILE}")"' in output
    assert "--nodes=2" in output
    assert "--partition=gb200" in output
    assert "--segment=2" in output
    assert "--time=05:00:00" in output
    assert "--gres" not in output
    assert "speculative_config" not in output


def test_launcher_defaults_to_staged_lyris_nightly_container() -> None:
    launcher = LAUNCHER.read_text()

    assert (
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/"
        "nemo_rl_nightly_20260715.sqsh"
    ) in launcher


def test_dflash_uses_matched_public_drafter_and_full_cuda_graphs() -> None:
    output = _dry_run("dflash_k16")

    assert "speculative_config.method=dflash" in output
    assert "speculative_config.num_speculative_tokens=16" in output
    assert "speculative_config.draft_tensor_parallel_size=1" in output
    assert "speculative_config.max_model_len=4096" in output
    assert "speculative_config.attention_backend=FLASH_ATTN" in output
    assert "kernel_config.enable_flashinfer_autotune=false" in output
    assert "models--z-lab--Qwen3-8B-DFlash-b16" in output
    assert "compilation_config.cudagraph_mode=FULL" in output
    assert (
        "compilation_config.cudagraph_capture_sizes=[17,34,68,136,272,544,1088]"
        in output
    )
    assert "policy.draft.enabled=true" not in output


def test_qwen30_profile_uses_unmodified_performance_recipe_and_matched_k7() -> None:
    output = _dry_run("dflash_k7", model_profile="qwen30ba3b")
    launcher = LAUNCHER.read_text()

    assert "grpo-qwen3-30ba3b-4n4g.yaml" in output
    assert "models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777" in launcher
    assert "models--inference-optimization--Qwen3-30B-A3B-speculator.dflash" in output
    assert "speculative_config.num_speculative_tokens=7" in output
    assert "speculative_config.attention_backend=FLASH_ATTN" in output
    assert "kernel_config.enable_flashinfer_autotune=false" in output
    assert "compilation_config.cudagraph_capture_sizes=[8,16,32,64,128,256,512]" in output
    assert "cluster.segment_size=4" in output
    assert "--nodes=4" in output
    assert "--segment=4" in output


def test_invalid_variant_fails_before_submission() -> None:
    env = os.environ.copy()
    env.update(
        {
            "MODE": "dry-run",
            "REPO_DIR": str(REPO_ROOT),
            "RUN_TAG": "contract-test",
            "VARIANT": "dflash_k15",
        }
    )
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        check=False,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "VARIANT must be baseline or dflash_k16" in result.stderr


def test_submission_accepts_single_file_dflash_checkpoint(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "model.safetensors.index.json").write_text("{}")
    draft = tmp_path / "draft"
    draft.mkdir()
    (draft / "model.safetensors").write_text("")
    container = tmp_path / "nemo_rl_nightly.sqsh"
    container.write_text("")
    wandb_api_key_file = tmp_path / "wandb_api_key"
    wandb_api_key_file.write_text("test-key")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text("#!/usr/bin/env bash\necho test-only-ok\n")
    fake_sbatch.chmod(0o755)
    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$*" == *"branch -r --contains"* ]]; then\n'
        "  echo origin/test\n"
        'elif [[ "$*" == *"rev-parse"* ]]; then\n'
        "  echo deadbeef\n"
        "fi\n"
    )
    fake_git.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "MODE": "test-only",
            "REPO_DIR": str(REPO_ROOT),
            "RUN_TAG": "contract-test-single-file",
            "RUN_DIR": str(tmp_path / "run"),
            "VARIANT": "dflash_k16",
            "TARGET_SNAPSHOT": str(target),
            "DRAFT_SNAPSHOT": str(draft),
            "CONTAINER": str(container),
            "WANDB_API_KEY_FILE": str(wandb_api_key_file),
            "PATH": f"{fake_bin}:{env['PATH']}",
        }
    )
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        check=True,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert "test-only-ok" in result.stdout


def test_submission_rejects_missing_wandb_key_before_sbatch(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "model.safetensors.index.json").write_text("{}")
    container = tmp_path / "nemo_rl_nightly.sqsh"
    container.write_text("")

    env = os.environ.copy()
    env.update(
        {
            "MODE": "test-only",
            "REPO_DIR": str(REPO_ROOT),
            "RUN_TAG": "contract-test-missing-wandb-key",
            "RUN_DIR": str(tmp_path / "run"),
            "VARIANT": "baseline",
            "TARGET_SNAPSHOT": str(target),
            "CONTAINER": str(container),
            "WANDB_API_KEY_FILE": str(tmp_path / "missing-key"),
        }
    )
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        check=False,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "W&B API key file is unavailable" in result.stderr


def test_submission_rejects_missing_container_before_sbatch(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "model.safetensors.index.json").write_text("{}")

    env = os.environ.copy()
    env.update(
        {
            "MODE": "test-only",
            "REPO_DIR": str(REPO_ROOT),
            "RUN_TAG": "contract-test-missing-container",
            "RUN_DIR": str(tmp_path / "run"),
            "VARIANT": "baseline",
            "TARGET_SNAPSHOT": str(target),
            "CONTAINER": str(tmp_path / "missing.sqsh"),
        }
    )
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        check=False,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "Container does not exist" in result.stderr
