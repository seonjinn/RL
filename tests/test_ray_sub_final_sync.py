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

"""Regression coverage for final Ray-log synchronization in ``ray.sub``."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RAY_SUB = REPO_ROOT / "ray.sub"


def _final_sync_status(
    driver_status: int, sync_complete: int
) -> subprocess.CompletedProcess[str]:
    source = RAY_SUB.read_text(encoding="utf-8")
    definitions_start = source.index("FINAL_SYNC_FAILURE_EXIT_CODE=")
    definitions_end = source.index("\n# Record job-start epoch", definitions_start)
    definitions = source[definitions_start:definitions_end]
    return subprocess.run(
        [
            "bash",
            "-c",
            (
                "set +e\n"
                f"{definitions}\n"
                f"final_sync_exit_status {driver_status} {sync_complete}\n"
                "exit $?"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("driver_status", "sync_complete", "expected_status"),
    [
        (0, 1, 0),
        (0, 0, 86),
        (17, 1, 17),
        (17, 0, 17),
    ],
)
def test_final_sync_exit_status_preserves_driver_failure(
    driver_status: int, sync_complete: int, expected_status: int
) -> None:
    result = _final_sync_status(driver_status, sync_complete)

    assert result.returncode == expected_status, result.stderr


def test_final_sync_requires_node_evidence_and_nonempty_ray_sessions() -> None:
    source = RAY_SUB.read_text(encoding="utf-8")

    assert "found_ray_logs=0" in source
    assert 'found_ray_logs" -eq 0' in source
    assert ".ray_logs_final_sync_evidence" in source
    assert 'for sync_node in "${nodes_array[@]}"; do' in source
    assert ".ray_logs_final_sync_ack.\\$sync_node" in source
    assert (
        'final_sync_exit_status "\\$driver_exit_code" "\\$final_sync_complete"'
        in source
    )


def test_restart_output_is_appended_without_clobbering_prior_attempt_logs() -> None:
    source = RAY_SUB.read_text(encoding="utf-8")

    assert 'LOG_DIR="$BASE_LOG_DIR/$SLURM_JOB_ID-$SLURM_RESTART_COUNT-logs"' in source
    assert "--open-mode=append" in (
        REPO_ROOT / "experiments/vllm_024_upgrade/submit_tail_gated_specdec_step20.sh"
    ).read_text(encoding="utf-8")
