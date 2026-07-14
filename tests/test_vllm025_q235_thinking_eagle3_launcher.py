import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    REPO_ROOT
    / "experiments"
    / "vllm_025_q235_specdec"
    / "submit_q235_thinking_eagle3_ptyche.sh"
)


def _dry_run(variant: str, **env_overrides: str) -> str:
    env = os.environ.copy()
    env.pop("CUDAGRAPH_METRICS", None)
    env.update(
        {
            "MODE": "dry-run",
            "REPO_DIR": str(REPO_ROOT),
            "VARIANT": variant,
            "RUN_TAG": "contract-test",
        }
    )
    env.update(env_overrides)
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        check=True,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    return result.stdout


def test_baseline_preserves_verified_q235_performance_contract() -> None:
    output = _dry_run("baseline")

    assert "grpo-qwen3-235b-16n4g.yaml" in output
    assert "policy.model_name=" not in output
    assert "policy.tokenizer.name=" not in output
    assert "HF_HUB_OFFLINE=1" in output
    assert "TRANSFORMERS_OFFLINE=1" in output
    assert "grpo.max_num_steps=1" in output
    assert "checkpointing.enabled=false" in output
    assert "policy.generation.vllm_cfg.enforce_eager=false" in output
    assert "policy.generation.temperature=1.0" in output
    assert "policy.generation.top_p=1.0" in output
    assert "NRL_DISABLE_NUMA_MEMBIND=1" in output
    assert "BASE_LOG_DIR=" in output
    assert "logger.wandb_enabled=false" in output
    assert "--nodes=16" in output
    assert "--segment=16" in output
    assert "--time=05:00:00" in output
    assert "--gres" not in output
    assert "speculative_config" not in output


def test_eagle3_uses_static_thinking_drafter_without_online_training() -> None:
    output = _dry_run("eagle3_k5")

    assert "speculative_config.method=eagle3" in output
    assert "speculative_config.num_speculative_tokens=5" in output
    assert "speculative_config.draft_tensor_parallel_size=1" in output
    assert "models--RedHatAI--Qwen3-235B-A22B-Thinking-2507-speculator.eagle3" in output
    assert "policy.draft.enabled=true" not in output


def test_cudagraph_metrics_are_opt_in() -> None:
    default_output = _dry_run("eagle3_k5")
    diagnostic_output = _dry_run("eagle3_k5", CUDAGRAPH_METRICS="true")

    assert "vllm_kwargs.cudagraph_metrics" not in default_output
    assert "++policy.generation.vllm_kwargs.cudagraph_metrics=true" in diagnostic_output


def test_dynamic_sd_schedule_is_opt_in() -> None:
    schedule = "[[1,4,5],[5,8,3],[9,16,1],[17,64,0]]"
    default_output = _dry_run("eagle3_k5")
    dynamic_output = _dry_run("eagle3_k5", DYNAMIC_SD_SCHEDULE=schedule).replace(
        "\\", ""
    )

    assert "num_speculative_tokens_per_batch_size" not in default_output
    assert (
        "speculative_config.num_speculative_tokens_per_batch_size="
        f"{schedule}" in dynamic_output
    )


def test_supported_variants_render_expected_speculative_token_count() -> None:
    expected_capture_sizes = {
        1: "[2,4,8,16,32,64,128]",
        3: "[4,8,16,32,64,128,256]",
        5: "[6,12,24,48,96,192,384]",
        7: "[8,16,32,64,128,256,512]",
        9: "[10,20,40,80,160,320,640]",
    }

    for k, capture_sizes in expected_capture_sizes.items():
        output = _dry_run(f"eagle3_k{k}").replace("\\", "")
        assert f"speculative_config.num_speculative_tokens={k}" in output
        assert (
            "vllm_kwargs.compilation_config.cudagraph_capture_sizes="
            f"{capture_sizes}" in output
        )


def test_invalid_variant_fails_before_submission() -> None:
    env = os.environ.copy()
    env.update(
        {
            "MODE": "dry-run",
            "REPO_DIR": str(REPO_ROOT),
            "VARIANT": "eagle3_k11",
            "RUN_TAG": "contract-test",
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
    assert "VARIANT must be" in result.stderr


def test_submission_runs_sbatch_from_repo_directory(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors.index.json").write_text("{}")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text("#!/usr/bin/env bash\npwd\n")
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
            "VARIANT": "baseline",
            "REPO_DIR": str(REPO_ROOT),
            "RUN_DIR": str(tmp_path / "run"),
            "RUN_TAG": "contract-test-cwd",
            "TARGET_SNAPSHOT": str(tmp_path),
            "PATH": f"{fake_bin}:{env['PATH']}",
        }
    )
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        check=True,
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.stdout.strip() == str(REPO_ROOT)
