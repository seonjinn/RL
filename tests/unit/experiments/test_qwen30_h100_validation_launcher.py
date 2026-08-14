import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parents[3]
RAY_SUB = REPO_ROOT / "ray.sub"
LAUNCHER = (
    REPO_ROOT
    / "experiments/pr3436-h100-validation/scripts/run_qwen30_hybridep.sh"
)


def test_ray_container_names_are_scoped_to_slurm_job() -> None:
    script = RAY_SUB.read_text()

    assert "RAY_CONTAINER_NAME_SUFFIX=${RAY_CONTAINER_NAME_SUFFIX:-${SLURM_JOB_ID:-manual}-${SLURM_RESTART_COUNT:-0}}" in script
    assert "RAY_HEAD_CONTAINER_NAME=\"ray-head-${RAY_CONTAINER_NAME_SUFFIX}\"" in script
    assert "RAY_WORKER_CONTAINER_NAME=\"ray-worker-${RAY_CONTAINER_NAME_SUFFIX}\"" in script
    assert "--container-name=ray-head" not in script
    assert "--container-name=ray-worker" not in script


def run_launcher(
    tmp_path: Path, **overrides: str
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    capture_file = tmp_path / "uv-arguments.txt"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$@\" > \"${CAPTURE_FILE}\"\n"
    )
    fake_uv.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "CAPTURE_FILE": str(capture_file),
            "EXPERIMENT_OUTPUT_DIR": str(tmp_path / "output"),
            "PATH": f"{tmp_path}:{env['PATH']}",
        }
    )
    env.update(overrides)
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    captured_arguments = (
        capture_file.read_text().splitlines() if capture_file.exists() else []
    )
    return result, captured_arguments


@pytest.mark.parametrize(
    ("environment", "expected_arguments"),
    [
        (
            {
                "DISPATCHER_MODE": "alltoall",
                "LOGPROB_BATCH_SIZE": "2",
                "LOGPROB_CHUNK_SIZE": "null",
            },
            {
                "policy.logprob_batch_size=2",
                "policy.logprob_chunk_size=null",
                "policy.megatron_cfg.moe_token_dispatcher_type=alltoall",
                "~policy.megatron_cfg.moe_flex_dispatcher_backend",
                "~policy.megatron_cfg.moe_hybridep_num_sms",
                "~policy.megatron_cfg.moe_hybridep_prepad_packed_inputs",
            },
        ),
        (
            {
                "DISPATCHER_MODE": "hybridep",
                "LOGPROB_BATCH_SIZE": "1",
                "LOGPROB_CHUNK_SIZE": "null",
            },
            {
                "policy.logprob_batch_size=1",
                "policy.logprob_chunk_size=null",
                "policy.megatron_cfg.moe_token_dispatcher_type=flex",
                "policy.megatron_cfg.moe_flex_dispatcher_backend=hybridep",
            },
        ),
        (
            {
                "DISPATCHER_MODE": "hybridep",
                "LOGPROB_BATCH_SIZE": "2",
                "LOGPROB_CHUNK_SIZE": "1024",
            },
            {
                "policy.logprob_batch_size=2",
                "policy.logprob_chunk_size=1024",
                "policy.megatron_cfg.moe_token_dispatcher_type=flex",
                "policy.megatron_cfg.moe_flex_dispatcher_backend=hybridep",
            },
        ),
    ],
)
def test_launcher_builds_matched_performance_arms(
    tmp_path: Path,
    environment: dict[str, str],
    expected_arguments: set[str],
) -> None:
    result, captured_arguments = run_launcher(tmp_path, **environment)

    assert result.returncode == 0, result.stderr
    assert expected_arguments.issubset(set(captured_arguments))


def test_launcher_rejects_unknown_dispatcher_mode(tmp_path: Path) -> None:
    result, _ = run_launcher(tmp_path, DISPATCHER_MODE="unknown")

    assert result.returncode != 0
    assert "Unsupported DISPATCHER_MODE" in result.stderr
