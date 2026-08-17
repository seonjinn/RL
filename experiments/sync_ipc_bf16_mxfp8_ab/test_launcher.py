from __future__ import annotations

import os
import subprocess
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).parent
LAUNCHER = EXPERIMENT_DIR / "submit.sh"


def render(model: str, arm: str, max_steps: int = 1) -> str:
    env = os.environ | {
        "ACTION": "render",
        "MODEL": model,
        "ARM": arm,
        "MAX_STEPS": str(max_steps),
        "RUN_SUFFIX": "test",
    }
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
    return result.stdout


def render_sbatch(use_gres: bool, account: str = "coreai_dlalgo_llm") -> str:
    env = os.environ | {
        "ACTION": "render-sbatch",
        "MODEL": "qwen30",
        "ARM": "bf16",
        "USE_GRES": str(use_gres).lower(),
        "SLURM_ACCOUNT": account,
    }
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
    return result.stdout


def test_qwen_bf16_uses_sync_colocated_ipc_and_cuda_graphs() -> None:
    command = render("qwen30", "bf16")

    assert "policy.generation.colocated.enabled=true" in command
    assert "policy.generation.refit_transport=null" in command
    assert "grpo.async_grpo.enabled=false" in command
    assert "data_plane.enabled=false" in command
    assert "policy.generation.real_quant_export_cpu_offload=false" in command
    assert "policy.generation.vllm_cfg.async_engine=false" in command
    assert "policy.generation.vllm_cfg.enforce_eager=false" in command
    assert "policy.generation.vllm_cfg.precision=bfloat16" in command
    assert "policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm" in command


def test_qwen_mxfp8_keeps_training_bf16_and_quantizes_rollout_experts() -> None:
    command = render("qwen30", "mxfp8", max_steps=20)

    assert "grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml" in command
    assert "policy.precision=bfloat16" in command
    assert "policy.generation.vllm_cfg.precision=fp8" in command
    assert "policy.generation.vllm_cfg.is_mx=true" in command
    assert "grpo.max_num_steps=20" in command


def test_nano_arms_use_the_same_sync_ipc_contract() -> None:
    for arm in ("bf16", "mxfp8"):
        command = render("nano", arm)
        assert "cluster.num_nodes=8" in command
        assert "policy.generation.colocated.enabled=true" in command
        assert "policy.generation.colocated.resources.num_nodes=8" in command
        assert "policy.generation.refit_transport=null" in command


def test_unknown_arm_is_rejected() -> None:
    env = os.environ | {
        "ACTION": "render",
        "MODEL": "qwen30",
        "ARM": "unknown",
    }
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 2
    assert "ARM must be bf16 or mxfp8" in result.stderr


def test_gres_is_optional_for_exclusive_node_clusters() -> None:
    assert "--gres" not in render_sbatch(use_gres=False)
    assert "--gres=gpu:4" in render_sbatch(use_gres=True)


def test_job_name_follows_cluster_account_convention() -> None:
    sbatch = render_sbatch(use_gres=False, account="coreai_dlalgo_llm")

    assert "--job-name=coreai_dlalgo_llm-sync-ipc.qwen30-bf16-1s" in sbatch
