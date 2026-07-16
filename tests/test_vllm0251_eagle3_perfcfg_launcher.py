import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    REPO_ROOT
    / "experiments"
    / "vllm_0251_eagle3_perfcfg"
    / "submit_qwen30_eagle3_ptyche.sh"
)


def _dry_run(variant: str, **env_overrides: str) -> str:
    env = os.environ.copy()
    for name in ("CAPTURE_PROFILE", "CUDAGRAPH_METRICS", "DYNAMIC_SD_SCHEDULE"):
        env.pop(name, None)
    env.update(
        {
            "MODE": "dry-run",
            "REPO_DIR": str(REPO_ROOT),
            "RUN_TAG": "contract-test",
            "VARIANT": variant,
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


def test_baseline_preserves_qwen30_performance_recipe() -> None:
    output = _dry_run("baseline")

    assert "grpo-qwen3-30ba3b-4n4g.yaml" in output
    assert "policy.model_name=" not in output
    assert "policy.tokenizer.name=" not in output
    assert "grpo.max_num_steps=2" in output
    assert "checkpointing.enabled=false" in output
    assert "policy.generation.vllm_cfg.enforce_eager=false" in output
    assert "policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=FULL_AND_PIECEWISE" in output
    assert "VLLM_USE_V2_MODEL_RUNNER=1" in output
    assert "--nodes=4" in output
    assert "--segment=4" in output
    assert "--gres" not in output
    assert "--dependency=" in output
    assert "--dependency=singleton" not in output
    assert "speculative_config" not in output


def test_eagle3_uses_static_drafter_and_native_mrv2_capture_sizes() -> None:
    output = _dry_run("eagle3_k5").replace("\\", "")

    assert "speculative_config.method=eagle3" in output
    assert "speculative_config.num_speculative_tokens=5" in output
    assert "speculative_config.draft_tensor_parallel_size=1" in output
    assert "models--RedHatAI--Qwen3-30B-A3B-Thinking-2507-speculator.eagle3" in output
    assert "compilation_config.cudagraph_capture_sizes=" not in output
    assert "policy.draft.enabled=true" not in output


def test_compact_profile_scales_capture_sizes_with_k_plus_one() -> None:
    expected = {
        1: "[2,4,8,16,32,64,128]",
        3: "[4,8,16,32,64,128,256]",
        5: "[6,12,24,48,96,192,384]",
        7: "[8,16,32,64,128,256,512]",
        9: "[10,20,40,80,160,320,640]",
    }

    for k, capture_sizes in expected.items():
        output = _dry_run(
            f"eagle3_k{k}", CAPTURE_PROFILE="compact"
        ).replace("\\", "")
        assert f"speculative_config.num_speculative_tokens={k}" in output
        assert f"compilation_config.cudagraph_capture_sizes={capture_sizes}" in output


def test_invalid_capture_profile_fails_before_submission() -> None:
    env = os.environ.copy()
    env.update(
        {
            "CAPTURE_PROFILE": "all",
            "MODE": "dry-run",
            "REPO_DIR": str(REPO_ROOT),
            "RUN_TAG": "contract-test",
            "VARIANT": "eagle3_k3",
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
    assert "CAPTURE_PROFILE must be native or compact" in result.stderr


def test_dynamic_sd_patch_and_graph_metrics_are_opt_in() -> None:
    schedule = "[[1,4,5],[5,8,3],[9,16,1],[17,64,0]]"
    default_output = _dry_run("eagle3_k5")
    dynamic_output = _dry_run(
        "eagle3_k5",
        DYNAMIC_SD_SCHEDULE=schedule,
        CUDAGRAPH_METRICS="true",
    ).replace("\\", "")

    assert "NRL_VENV_POST_SYNC_SCRIPT=" not in default_output
    assert "vllm_kwargs.cudagraph_metrics" not in default_output
    assert f"num_speculative_tokens_per_batch_size={schedule}" in dynamic_output
    assert "apply_vllm0251_dynamic_sd_cg_fix.py" in dynamic_output
    assert (
        "NRL_VENV_POST_SYNC_TARGET="
        "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"
        in dynamic_output
    )
    assert "vllm_kwargs.cudagraph_metrics" not in dynamic_output
    assert "policy.generation.vllm_cfg.enable_vllm_metrics_logger=true" in dynamic_output


def test_dynamic_sd_rejects_compact_capture_profile() -> None:
    env = os.environ.copy()
    env.update(
        {
            "CAPTURE_PROFILE": "compact",
            "DYNAMIC_SD_SCHEDULE": "[[1,4,3],[5,64,0]]",
            "MODE": "dry-run",
            "REPO_DIR": str(REPO_ROOT),
            "RUN_TAG": "contract-test",
            "VARIANT": "eagle3_k3",
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
    assert "DynamicSD requires CAPTURE_PROFILE=native" in result.stderr


def test_invalid_variant_fails_before_submission() -> None:
    env = os.environ.copy()
    env.update(
        {
            "MODE": "dry-run",
            "REPO_DIR": str(REPO_ROOT),
            "RUN_TAG": "contract-test",
            "VARIANT": "eagle3_k11",
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
