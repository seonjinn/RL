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

import json
import os
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


def _sync_ray_logs_once(source: str, assignment: str) -> str:
    start = source.index("sync-ray-logs-once()", source.index(f"{assignment}=$(cat"))
    end = source.index("\nlog-sync-sidecar()", start)
    return source[start:end].replace("\\$", "$")


@pytest.mark.parametrize("assignment", ("head_cmd", "worker_cmd"))
def test_final_sync_rejects_empty_ray_sessions_before_acknowledging(
    tmp_path: Path, assignment: str
) -> None:
    source = RAY_SUB.read_text(encoding="utf-8")
    session_logs = tmp_path / "ray" / "session_1" / "logs"
    session_logs.mkdir(parents=True)
    function = _sync_ray_logs_once(source, assignment).replace(
        "/tmp/ray", str(tmp_path / "ray")
    )
    log_dir = tmp_path / "collected"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "rsync").write_text(
        '#!/usr/bin/env bash\ncp -a "${2%/}/." "$3"\n', encoding="utf-8"
    )
    (fake_bin / "rsync").chmod(0o755)

    empty = subprocess.run(
        ["bash", "-c", f"{function}\nsync-ray-logs-once"],
        check=False,
        capture_output=True,
        text=True,
        env={
            "LOG_DIR": str(log_dir),
            "SLURMD_NODENAME": "node-a",
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
        },
    )
    assert empty.returncode == 1, empty.stderr

    (session_logs / "worker-1.err").write_text("Ray log\n", encoding="utf-8")
    populated = subprocess.run(
        ["bash", "-c", f"{function}\nsync-ray-logs-once"],
        check=False,
        capture_output=True,
        text=True,
        env={
            "LOG_DIR": str(log_dir),
            "SLURMD_NODENAME": "node-a",
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
        },
    )
    assert populated.returncode == 0, populated.stderr


def test_final_sync_requires_copied_logs_and_canonical_node_evidence() -> None:
    source = RAY_SUB.read_text(encoding="utf-8")

    assert source.count("local copied_ray_logs=0") == 2
    assert source.count('copied_ray_logs" -eq 0') == 2
    assert source.count('find "\\$session_dir/logs" -type f -print -quit') == 2
    assert source.count("copied_ray_logs=1") >= 2
    assert ".ray_logs_final_sync_evidence" in source
    assert ".ray_logs_final_sync_ack.head" in source
    assert ".ray_logs_final_sync_ack.worker-\\$SLURM_PROCID" in source
    assert 'for sync_node in "\\${expected_sync_nodes[@]}"; do' in source
    assert (
        'final_sync_exit_status "\\$driver_exit_code" "\\$final_sync_complete"'
        in source
    )


def test_final_sync_persists_structured_driver_status(tmp_path: Path) -> None:
    source = RAY_SUB.read_text(encoding="utf-8")
    definitions_start = source.index("FINAL_SYNC_FAILURE_EXIT_CODE=")
    definitions_end = source.index("\n# Record job-start epoch", definitions_start)
    definitions = source[definitions_start:definitions_end]

    result = subprocess.run(
        [
            "bash",
            "-c",
            f'{definitions}\nwrite_final_sync_status "$1" 17 1',
            "bash",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(
        (tmp_path / ".ray_logs_final_sync_status.json").read_text(encoding="utf-8")
    ) == {
        "schema_version": 1,
        "driver_exit_code": 17,
        "final_sync_complete": True,
    }


def test_runtime_checkout_guard_rejects_changed_queued_checkout(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    tracked = repo / "tracked.txt"
    tracked.write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "seed"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source = RAY_SUB.read_text(encoding="utf-8")
    definitions_start = source.index("FINAL_SYNC_FAILURE_EXIT_CODE=")
    definitions_end = source.index("\n# Record job-start epoch", definitions_start)
    definitions = source[definitions_start:definitions_end]

    clean = subprocess.run(
        ["bash", "-c", f"{definitions}\nverify_runtime_checkout"],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "NRL_RUNTIME_CHECKOUT": str(repo),
            "NRL_EXPECTED_RUNTIME_COMMIT": commit,
        },
    )
    assert clean.returncode == 0, clean.stderr

    tracked.write_text("changed while queued\n", encoding="utf-8")
    changed = subprocess.run(
        ["bash", "-c", f"{definitions}\nverify_runtime_checkout"],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "NRL_RUNTIME_CHECKOUT": str(repo),
            "NRL_EXPECTED_RUNTIME_COMMIT": commit,
        },
    )
    assert changed.returncode != 0
    assert "runtime checkout is not clean" in changed.stderr


def test_restart_output_is_appended_without_clobbering_prior_attempt_logs() -> None:
    source = RAY_SUB.read_text(encoding="utf-8")

    assert 'LOG_DIR="$BASE_LOG_DIR/$SLURM_JOB_ID-$SLURM_RESTART_COUNT-logs"' in source
    assert "--open-mode=append" in (
        REPO_ROOT / "experiments/vllm_024_upgrade/submit_tail_gated_specdec_step20.sh"
    ).read_text(encoding="utf-8")
