import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RAY_SUB = REPO_ROOT / "ray.sub"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _write_fake_commands(bin_dir: Path) -> None:
    _write_executable(
        bin_dir / "sinfo",
        "#!/bin/bash\nexit 0\n",
    )
    _write_executable(
        bin_dir / "scontrol",
        """#!/bin/bash
if [[ "$1 $2" == "show hostnames" ]]; then
  printf 'node0\\nnode1\\n'
elif [[ "$1 $2" == "show node" ]]; then
  printf 'NodeName=%s CPUTot=8\\n' "$3"
fi
""",
    )
    _write_executable(
        bin_dir / "host",
        "#!/bin/bash\nprintf '%s has address 127.0.0.1\\n' \"$1\"\n",
    )
    _write_executable(
        bin_dir / "srun",
        """#!/bin/bash
role=""
for arg in "$@"; do
  case "$arg" in
    --container-name=ray-head) role=head ;;
    --container-name=ray-worker) role=worker ;;
  esac
done
if [[ -n "$role" ]]; then
  printf '%s' "${!#}" > "$RAY_SUB_CAPTURE_DIR/$role.sh"
fi
""",
    )
    _write_executable(
        bin_dir / "ray",
        """#!/bin/bash
if [[ "$1" == "status" ]]; then
  printf 'worker_units 8.0/8.0\\n'
  exit 0
fi
if [[ "$1" == "start" ]]; then
  env | sort > "$RAY_SUB_DAEMON_ENV_DIR/$(date +%s%N)-$$"
  count_file="$RAY_SUB_DAEMON_ENV_DIR/start-count"
  count=0
  [[ -f "$count_file" ]] && count=$(<"$count_file")
  count=$((count + 1))
  printf '%s' "$count" > "$count_file"
  if (( count <= ${RAY_SUB_RAY_START_FAILURES:-0} )); then
    exit 1
  fi
fi
""",
    )
    _write_executable(
        bin_dir / "sleep",
        "#!/bin/bash\n/bin/sleep 0.02\n",
    )
    _write_executable(bin_dir / "rm", "#!/bin/bash\nexit 0\n")
    _write_executable(bin_dir / "sed", "#!/bin/bash\nexit 0\n")


def _base_environment(
    tmp_path: Path, bin_dir: Path, *, job_id: str = "424242"
) -> dict[str, str]:
    return {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "SLURM_JOB_ID": job_id,
        "SLURM_JOB_NUM_NODES": "2",
        "SLURM_JOB_PARTITION": "test",
        "SLURM_JOB_ACCOUNT": "test",
        "SLURM_JOB_NODELIST": "node[0-1]",
        "SLURM_SUBMIT_DIR": str(tmp_path),
        "BASE_LOG_DIR": str(tmp_path),
        "CONTAINER": "test.sqsh",
        "MOUNTS": "",
        "GPUS_PER_NODE": "4",
        "CPUS_PER_WORKER": "8",
        "RAY_SUB_CAPTURE_DIR": str(tmp_path / "capture"),
        "RAY_SUB_DAEMON_ENV_DIR": str(tmp_path / "daemon-env"),
        "RAY_SUB_OBSERVATIONS": str(tmp_path / "observations"),
    }


def _render_launcher(
    tmp_path: Path,
    *,
    job_id: str = "424242",
    caller_alias: str | None = None,
    setup_command: str = 'printf "setup:%s\\n" "$SLURM_JOB_ID" >> "$RAY_SUB_OBSERVATIONS"',
    driver_command: str = 'printf "driver:%s\\n" "$SLURM_JOB_ID" >> "$RAY_SUB_OBSERVATIONS"',
) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    capture_dir = tmp_path / "capture"
    daemon_env_dir = tmp_path / "daemon-env"
    bin_dir.mkdir()
    capture_dir.mkdir()
    daemon_env_dir.mkdir()
    _write_fake_commands(bin_dir)
    env = _base_environment(tmp_path, bin_dir, job_id=job_id)
    if caller_alias is not None:
        env["NATIVE_SLURM_JOB_ID"] = caller_alias
    env["SETUP_COMMAND"] = setup_command
    env["COMMAND"] = driver_command
    renderer = tmp_path / "render-ray-sub.sh"
    source = RAY_SUB.read_text()
    source = source.replace(
        "declare -A SRUN_PIDS", ": # test harness does not launch sruns"
    )
    source = source.replace(
        "########################################################\n# Optional sandbox sidecar for NeMo-Skills-backed Gym resources.",
        """printf '%s' "$head_cmd" > "$RAY_SUB_CAPTURE_DIR/head.sh"
printf '%s' "$worker_cmd" > "$RAY_SUB_CAPTURE_DIR/worker.sh"
exit 0

########################################################
# Optional sandbox sidecar for NeMo-Skills-backed Gym resources.""",
        1,
    )
    renderer.write_text(source)
    subprocess.run(
        ["bash", str(renderer)],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    env.pop("NATIVE_SLURM_JOB_ID", None)
    return env, tmp_path / f"{job_id}-logs"


def _run_generated_script(
    script: Path, env: dict[str, str], *, expect_success: bool = True
) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        ["bash", str(script)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        timeout=10,
    )
    if expect_success:
        assert result.returncode == 0
    return result


def _observations(env: dict[str, str]) -> list[str]:
    path = Path(env["RAY_SUB_OBSERVATIONS"])
    return path.read_text().splitlines() if path.exists() else []


def _daemon_environments(env: dict[str, str]) -> list[str]:
    return [
        path.read_text()
        for path in Path(env["RAY_SUB_DAEMON_ENV_DIR"]).iterdir()
        if path.name != "start-count"
    ]


def test_generated_head_worker_and_driver_receive_only_the_native_scheduler_id(
    tmp_path,
):
    """Fails if generated user hooks do not receive the captured Slurm job ID."""
    env, log_dir = _render_launcher(tmp_path, caller_alias="999999")

    _run_generated_script(tmp_path / "capture" / "head.sh", env)
    _run_generated_script(tmp_path / "capture" / "worker.sh", env, expect_success=False)

    assert _observations(env) == ["setup:424242", "driver:424242", "setup:424242"]
    assert (log_dir / "ENDED").exists()
    assert _daemon_environments(env)
    for daemon_environment in _daemon_environments(env):
        assert not any(
            line.startswith(
                ("PMI", "PMIX", "MPI", "OMPI", "SLURM_", "NATIVE_SLURM_JOB_ID=")
            )
            for line in daemon_environment.splitlines()
        )


@pytest.mark.parametrize("job_id", ["", "not-a-decimal-job-id"])
def test_invalid_native_scheduler_id_fails_before_rendering_or_launching(
    tmp_path, job_id
):
    """Fails if an absent or malformed scheduler identity reaches user hooks or Ray."""
    env, _ = _render_launcher(tmp_path, job_id=job_id)

    assert not (Path(env["RAY_SUB_CAPTURE_DIR"]) / "head.sh").exists()
    assert not _observations(env)
    assert not _daemon_environments(env)


@pytest.mark.parametrize("role", ["head", "worker"])
def test_setup_failure_signals_ended_and_prevents_ray_and_driver(tmp_path, role):
    """Fails if a failed per-node setup is ignored before the Ray retry loop."""
    env, log_dir = _render_launcher(
        tmp_path,
        setup_command='printf "setup:%s\\n" "$SLURM_JOB_ID" >> "$RAY_SUB_OBSERVATIONS"; exit 23',
    )
    env["RAY_SUB_RAY_START_FAILURES"] = "100"

    result = _run_generated_script(
        tmp_path / "capture" / f"{role}.sh", env, expect_success=False
    )

    assert result.returncode != 0
    assert _observations(env) == ["setup:424242"]
    assert (log_dir / "ENDED").exists()
    assert not _daemon_environments(env)


def test_successful_setup_runs_once_when_the_head_retries_ray_start(tmp_path):
    """Fails if setup remains inside the head's Ray-start retry loop."""
    env, _ = _render_launcher(tmp_path)
    env["RAY_SUB_RAY_START_FAILURES"] = "2"

    _run_generated_script(tmp_path / "capture" / "head.sh", env)

    assert _observations(env) == ["setup:424242", "driver:424242"]
    assert len(_daemon_environments(env)) == 3
