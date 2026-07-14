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


def _dry_run(variant: str) -> str:
    env = os.environ.copy()
    env.update(
        {
            "MODE": "dry-run",
            "REPO_DIR": str(REPO_ROOT),
            "VARIANT": variant,
            "RUN_TAG": "contract-test",
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
    return result.stdout


def test_baseline_preserves_verified_q235_performance_contract() -> None:
    output = _dry_run("baseline")

    assert "grpo-qwen3-235b-16n4g.yaml" in output
    assert "grpo.max_num_steps=1" in output
    assert "checkpointing.enabled=false" in output
    assert "policy.generation.vllm_cfg.enforce_eager=false" in output
    assert "policy.generation.temperature=1.0" in output
    assert "policy.generation.top_p=1.0" in output
    assert "NRL_DISABLE_NUMA_MEMBIND=1" in output
    assert "logger.wandb_enabled=false" in output
    assert "--nodes=16" in output
    assert "--segment=16" in output
    assert "--gres" not in output
    assert "speculative_config" not in output


def test_eagle3_uses_static_thinking_drafter_without_online_training() -> None:
    output = _dry_run("eagle3_k5")

    assert "speculative_config.method=eagle3" in output
    assert "speculative_config.num_speculative_tokens=5" in output
    assert "speculative_config.draft_tensor_parallel_size=1" in output
    assert (
        "models--RedHatAI--Qwen3-235B-A22B-Thinking-2507-speculator.eagle3"
        in output
    )
    assert "policy.draft.enabled=true" not in output


def test_supported_variants_render_expected_speculative_token_count() -> None:
    for k in (3, 5, 7, 9):
        output = _dry_run(f"eagle3_k{k}")
        assert f"speculative_config.num_speculative_tokens={k}" in output


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
