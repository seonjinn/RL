# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MINI_LAUNCHER = (
    REPO_ROOT
    / "experiments"
    / "vllm_024_upgrade"
    / "submit_tail_gated_specdec_mini_sync_grpo.sh"
)
MINI_VARIANTS = (
    "baseline_v2",
    "always_on_v2_k5",
    "fastrl_threshold_v2_k5",
)


def _run_mini(**environment: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(MINI_LAUNCHER), "dry-run"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "REPO_DIR": "/lustre/test/nemo-rl",
            "LYRIS_ROOT": "/lustre/test",
            "HF_HOME": "/lustre/test/hf_home",
            "CONTAINER": "/lustre/test/nemo-rl.sqsh",
            "EXPERIMENT_ROOT": "/lustre/test/tail-gate-runs",
            "RUN_TAG": "mini-contract",
            "ATTEMPT_ID": "attempt-1",
            **environment,
        },
        check=False,
        capture_output=True,
        text=True,
    )


def _dry_run_mini(**environment: str) -> str:
    result = _run_mini(**environment)

    assert result.returncode == 0, result.stderr
    return result.stdout


def test_mini_wrapper_renders_exact_three_arm_sync_grpo_smoke() -> None:
    output = _dry_run_mini()
    job_lines = [
        line for line in output.splitlines() if line.startswith("[DRY-RUN] job ")
    ]

    assert len(job_lines) == 3
    assert [
        line.split("variant=", maxsplit=1)[1].split()[0] for line in job_lines
    ] == list(MINI_VARIANTS)
    assert "efficient_roofline_v2_k5" not in output
    assert "baseline_v1" not in output
    assert "always_on_v1_k5" not in output

    for expected in (
        "grpo-qwen3-32b-4n4g.yaml",
        "grpo.max_num_steps=2",
        "grpo.num_prompts_per_step=16",
        "grpo.num_generations_per_prompt=4",
        "policy.train_global_batch_size=64",
        "policy.max_total_sequence_length=1024",
        "policy.generation.max_new_tokens=1024",
        "policy.generation.vllm_cfg.max_model_len=1056",
        "checkpointing.enabled=false",
        "uv run examples/run_grpo.py",
        "cudagraph_mode=FULL_AND_PIECEWISE",
        "--nodes=4",
        "--segment=4",
        "--account=coreai_dlalgo_llm",
        "--partition=batch",
        "WANDB_RUN_GROUP=mini-contract",
        "logger.wandb.project=nemorl-vllm024-tail-gated-mini-sync-grpo-pre-tyche",
        "logger.wandb.entity=nvidia",
    ):
        assert expected in output
    assert "--gres" not in output

    command_by_variant = {
        line.split("-qwen32b-", maxsplit=1)[1].split()[0]: line
        for line in output.splitlines()
        if line.startswith("[DRY-RUN] command ")
    }
    assert "scheduler_cls=" not in command_by_variant["baseline_v2"]
    assert "draft_sample_method=" not in command_by_variant["baseline_v2"]
    assert "scheduler_cls=" not in command_by_variant["always_on_v2_k5"]
    assert "draft_sample_method=probabilistic" in command_by_variant["always_on_v2_k5"]
    threshold_command = command_by_variant["fastrl_threshold_v2_k5"]
    assert "draft_sample_method=probabilistic" in threshold_command
    assert "sd_tail_gate_mode=threshold" in threshold_command
    assert "sd_tail_gate_threshold=4" in threshold_command
    assert "sd_tail_gate_consecutive_checks=10" in threshold_command


def test_mini_wrapper_allows_explicit_smoke_setting_overrides() -> None:
    output = _dry_run_mini(
        MAX_STEPS="3", WANDB_PROJECT="caller-project", TAIL_GATE_THRESHOLD="3"
    )
    command_lines = [
        line for line in output.splitlines() if line.startswith("[DRY-RUN] command ")
    ]

    assert len(command_lines) == 3
    assert all("grpo.max_num_steps=3" in line for line in command_lines)
    assert all("logger.wandb.project=caller-project" in line for line in command_lines)
    threshold_command = next(
        line for line in command_lines if "fastrl_threshold_v2_k5" in line
    )
    assert "sd_tail_gate_threshold=3" in threshold_command


def test_mini_wrapper_rejects_threshold_at_local_scheduler_capacity() -> None:
    result = _run_mini(TAIL_GATE_THRESHOLD="8")

    assert result.returncode == 2
    assert (
        "TAIL_GATE_THRESHOLD must be below local scheduler capacity 8" in result.stderr
    )
    assert "[DRY-RUN] job" not in result.stdout


def test_mini_wrapper_isolates_default_retry_attempts(tmp_path: Path) -> None:
    def run_once() -> str:
        result = subprocess.run(
            ["bash", str(MINI_LAUNCHER), "dry-run"],
            cwd=REPO_ROOT,
            env={
                **os.environ,
                "REPO_DIR": "/lustre/test/nemo-rl",
                "LYRIS_ROOT": str(tmp_path),
                "HF_HOME": "/lustre/test/hf_home",
                "CONTAINER": "/lustre/test/nemo-rl.sqsh",
                "RUN_TAG": "mini-contract",
            },
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        return next(
            token.removeprefix("BASE_LOG_DIR=")
            for token in result.stdout.split()
            if token.startswith("BASE_LOG_DIR=")
        )

    assert run_once() != run_once()
