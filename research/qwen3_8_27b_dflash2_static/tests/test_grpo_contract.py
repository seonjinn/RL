from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
TARGET_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
DRAFT_REVISION = "dedf8df68adfb1afeaf7b7480c0a0243108177b4"


def _load_training_contract():
    path = EXPERIMENT_ROOT / "training_contract.py"
    spec = importlib.util.spec_from_file_location("dflash2_training_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_grpo_recipe_is_static_target_only_and_defaults_to_one_step() -> None:
    contract = _load_training_contract()

    resolved = contract.load_training_recipe(EXPERIMENT_ROOT / "grpo.yaml")

    assert resolved == {
        "max_num_steps": 1,
        "target_placeholder": "__DFLASH2_TARGET_SNAPSHOT__",
        "draft_placeholder": "__DFLASH2_DRAFT_SNAPSHOT__",
        "draft_refit": False,
        "language_model_only": True,
        "speculative_method": "dflash",
        "num_speculative_tokens": 7,
        "v2_model_runner": "1",
    }


@pytest.mark.parametrize("steps", [0, 2, 19, 21, True])
def test_training_launcher_rejects_any_step_count_except_one_or_twenty(
    steps: object,
) -> None:
    contract = _load_training_contract()

    with pytest.raises(ValueError, match="1 or 20"):
        contract.validate_training_steps(steps)


def test_training_command_runs_real_grpo_with_pinned_local_snapshots() -> None:
    contract = _load_training_contract()
    target = Path(f"/lustre/user/models/Qwen3.8-27B/{TARGET_REVISION}")
    draft = Path(f"/lustre/user/models/Qwen3.8-27B-DFlash2/{DRAFT_REVISION}")
    output = Path("/lustre/user/results/dflash2-grpo/job-123")

    command = contract.build_training_command(
        repo_root=Path("/home/user/RL"),
        recipe=Path("/home/user/RL/research/qwen3_8_27b_dflash2_static/grpo.yaml"),
        target_snapshot=target,
        draft_snapshot=draft,
        output_dir=output,
        steps=20,
    )

    assert command == [
        sys.executable,
        "/home/user/RL/examples/run_grpo.py",
        "--config",
        "/home/user/RL/research/qwen3_8_27b_dflash2_static/grpo.yaml",
        "grpo.max_num_steps=20",
        f"policy.model_name={target}",
        f"policy.tokenizer.name={target}",
        (f"policy.generation.vllm_kwargs.speculative_config.model={draft}"),
        f"logger.log_dir={output}/logs",
        f"checkpointing.checkpoint_dir={output}/checkpoints",
    ]


@pytest.mark.parametrize(
    ("target", "draft", "message"),
    [
        (
            "/lustre/user/models/Qwen3.8-27B/wrong",
            f"/lustre/user/models/Qwen3.8-27B-DFlash2/{DRAFT_REVISION}",
            "target snapshot",
        ),
        (
            f"/lustre/user/models/Qwen3.8-27B/{TARGET_REVISION}",
            "/lustre/user/models/Qwen3.8-27B-DFlash2/wrong",
            "draft snapshot",
        ),
        (
            f"/home/user/models/Qwen3.8-27B/{TARGET_REVISION}",
            f"/lustre/user/models/Qwen3.8-27B-DFlash2/{DRAFT_REVISION}",
            "under /lustre",
        ),
    ],
)
def test_training_command_rejects_unpinned_or_misplaced_snapshots(
    target: str,
    draft: str,
    message: str,
) -> None:
    contract = _load_training_contract()

    with pytest.raises(ValueError, match=message):
        contract.build_training_command(
            repo_root=Path("/home/user/RL"),
            recipe=Path("/home/user/RL/research/qwen3_8_27b_dflash2_static/grpo.yaml"),
            target_snapshot=Path(target),
            draft_snapshot=Path(draft),
            output_dir=Path("/lustre/user/results/dflash2-grpo/job-123"),
            steps=1,
        )


def test_grpo_slurm_dry_run_is_distinct_from_twenty_request_benchmark() -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "CONTAINER_IMAGE": "/lustre/user/containers/nemo-rl-dflash2-f946.sqsh",
            "REPO_ROOT": "/home/user/RL",
            "TARGET_SNAPSHOT": (f"/lustre/user/models/Qwen3.8-27B/{TARGET_REVISION}"),
            "DRAFT_SNAPSHOT": (
                f"/lustre/user/models/Qwen3.8-27B-DFlash2/{DRAFT_REVISION}"
            ),
            "OUTPUT_ROOT": "/lustre/user/results/dflash2-grpo",
            "SMOKE_STEPS": "1",
        }
    )

    result = subprocess.run(
        ["bash", str(EXPERIMENT_ROOT / "run_grpo.slurm"), "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "examples/run_grpo.py" in result.stdout
    assert "grpo.max_num_steps=1" in result.stdout
    assert "NEMO_RL_PY_EXECUTABLES_SYSTEM=1" in result.stdout
    assert "VLLM_USE_V2_MODEL_RUNNER=1" in result.stdout
    assert "--request-count" not in result.stdout


def test_actor_runtime_contract_requires_actual_nemo_worker_and_system_vllm() -> None:
    path = EXPERIMENT_ROOT / "preflight.py"
    spec = importlib.util.spec_from_file_location("dflash2_actor_preflight", path)
    assert spec is not None and spec.loader is not None
    preflight = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preflight)

    preflight.validate_nemo_actor_runtime(
        actor_executable="/opt/venv/bin/python",
        system_executable="/opt/venv/bin/python",
        actor_module_path="/home/user/RL/nemo_rl/models/generation/vllm/vllm_worker.py",
        vllm_module_path="/usr/local/lib/python3.13/site-packages/vllm/__init__.py",
    )

    with pytest.raises(RuntimeError, match="system interpreter"):
        preflight.validate_nemo_actor_runtime(
            actor_executable="uv run --locked --extra vllm",
            system_executable="/opt/venv/bin/python",
            actor_module_path=(
                "/home/user/RL/nemo_rl/models/generation/vllm/vllm_worker.py"
            ),
            vllm_module_path=(
                "/usr/local/lib/python3.13/site-packages/vllm/__init__.py"
            ),
        )

    with pytest.raises(RuntimeError, match="actual NeMo-RL"):
        preflight.validate_nemo_actor_runtime(
            actor_executable="/opt/venv/bin/python",
            system_executable="/opt/venv/bin/python",
            actor_module_path="",
            vllm_module_path=(
                "/usr/local/lib/python3.13/site-packages/vllm/__init__.py"
            ),
        )


def test_runtime_preflight_slurm_imports_the_nemo_actor_without_training() -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "CONTAINER_IMAGE": "/lustre/user/containers/vllm-f946.sqsh",
            "REPO_ROOT": "/home/user/RL",
            "OUTPUT_ROOT": "/lustre/user/results/dflash2-runtime",
        }
    )

    result = subprocess.run(
        ["bash", str(EXPERIMENT_ROOT / "runtime_preflight.slurm"), "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "--gpus-per-task=1" in result.stdout
    assert "--nemo-actor-json" in result.stdout
    assert "NEMO_RL_PY_EXECUTABLES_SYSTEM=1" in result.stdout
    assert "examples/run_grpo.py" not in result.stdout
