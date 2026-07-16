import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    REPO_ROOT
    / "experiments"
    / "nemotron3_super_dynamic_mtp"
    / "submit_nemotron3_super_dynamic_mtp.sh"
)


def _dry_run(variant: str, cluster: str = "ptyche") -> str:
    env = os.environ.copy()
    env.update(
        {
            "CLUSTER": cluster,
            "MODE": "dry-run",
            "REPO_DIR": str(REPO_ROOT),
            "RUN_TAG": "contract-test",
            "VARIANT": variant,
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


def test_ptyche_smoke_uses_pr3207_gb200_topology() -> None:
    output = _dry_run("mtp_off")

    assert "--partition=batch" in output
    assert "--nodes=32" in output
    assert "--segment=8" in output
    assert "--ntasks-per-node=1" in output
    assert "--time=02:00:00" in output
    assert "--gres" not in output
    assert "nemo_rl_nightly_20260715.sqsh" in output
    assert "nemo-rl-cg/containers" in output


def test_all_variants_keep_cuda_graphs_and_vllm_v1_enabled() -> None:
    for variant in (
        "pr_baseline",
        "mtp_off",
        "native_mtp_k5",
        "dynamic_native_mtp_k5",
    ):
        output = _dry_run(variant)

        assert "VLLM_USE_V2_MODEL_RUNNER=0" in output
        assert "VLLM_ATTENTION_BACKEND=TRITON_ATTN" in output
        assert "policy.generation.vllm_cfg.enforce_eager=false" in output
        assert "checkpointing.enabled=false" in output
        assert "grpo.max_num_steps=2" in output
        assert "NCCL_NVLS_ENABLE=0" in output


def test_variants_select_committed_recipe_files() -> None:
    expected = {
        "pr_baseline": "grpo-nemotron3-super-120BA12B-32n4g.yaml",
        "mtp_off": "grpo-nemotron3-super-120BA12B-32n4g-mtp-off.yaml",
        "native_mtp_k5": ("grpo-nemotron3-super-120BA12B-32n4g-native-mtp-k5.yaml"),
        "dynamic_native_mtp_k5": (
            "grpo-nemotron3-super-120BA12B-32n4g-dynamic-native-mtp-k5.yaml"
        ),
    }

    for variant, recipe in expected.items():
        assert recipe in _dry_run(variant)


def test_wandb_and_result_provenance_are_enabled() -> None:
    output = _dry_run("dynamic_native_mtp_k5")

    assert "logger.wandb_enabled=true" in output
    assert "logger.wandb.project=nemo-rl-nemotron3-super-mtp" in output
    assert "logger.tensorboard_enabled=true" in output
    assert ".secrets/wandb_api_key" in output
    assert "WANDB_RESUME=never" in output


def test_lyris_uses_lyris_partition_and_container_path() -> None:
    output = _dry_run("native_mtp_k5", cluster="lyris")

    assert "--partition=gb200" in output
    assert "/users/sna/containers/nemo_rl_nightly_20260715.sqsh" in output
    assert "nemo-rl-cg/containers" not in output


def test_invalid_variant_fails_before_submission() -> None:
    env = os.environ.copy()
    env.update(
        {
            "CLUSTER": "ptyche",
            "MODE": "dry-run",
            "REPO_DIR": str(REPO_ROOT),
            "RUN_TAG": "contract-test",
            "VARIANT": "native_mtp_k3",
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
