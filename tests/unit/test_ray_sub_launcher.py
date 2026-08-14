import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RAY_SUB = REPO_ROOT / "ray.sub"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _write_fake_commands(bin_dir: Path) -> None:
    _write_executable(bin_dir / "sinfo", "#!/bin/bash\nexit 0\n")
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
case "$role" in
  head) export SLURMD_NODENAME=node0 SLURM_PROCID=0 ;;
  worker) export SLURMD_NODENAME=node1 SLURM_PROCID=1 ;;
  *) exit 0 ;;
esac
export RAY_SUB_TEST_ROLE="$role"
printf '%s\\n' "$role" >> "$RAY_SUB_SRUN_LOG"
if [[ "$role" == "worker" && "${RAY_SUB_SKIP_WORKER:-0}" == "1" ]]; then
  exit 0
fi
if [[ "$role" == "worker" && "${RAY_SUB_HOLD_WORKER_BEFORE_COMMAND:-0}" == "1" ]]; then
  printf 'worker-holding-before-command\\n' >> "$RAY_SUB_SRUN_LOG"
  while [[ ! -f "$RAY_SUB_LOG_DIR/ENDED" ]]; do /bin/sleep 0.02; done
  exit 1
fi
exec /bin/bash -x -c "${!#}"
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
  if [[ " $* " == *" --head "* ]]; then
    count_file="$RAY_SUB_DAEMON_ENV_DIR/head-start-count"
    count=0
    [[ -f "$count_file" ]] && count=$(<"$count_file")
    count=$((count + 1))
    printf '%s' "$count" > "$count_file"
    if (( count <= ${RAY_SUB_RAY_START_FAILURES:-0} )); then
      exit 1
    fi
  fi
  if [[ " $* " == *" --block "* ]]; then
    while [[ ! -f "$RAY_SUB_LOG_DIR/ENDED" ]]; do /bin/sleep 0.02; done
    exit 1
  fi
fi
""",
    )
    _write_executable(bin_dir / "sleep", "#!/bin/bash\n/bin/sleep 0.02\n")
    _write_executable(bin_dir / "rm", "#!/bin/bash\nexit 0\n")
    _write_executable(bin_dir / "sed", "#!/bin/bash\nexit 0\n")


def _base_environment(
    tmp_path: Path, bin_dir: Path, *, job_id: str = "424242"
) -> dict[str, str]:
    log_dir = tmp_path / f"{job_id}-logs"
    return {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "SLURM_JOB_ID": job_id,
        "SLURM_JOB_NUM_NODES": "2",
        "SLURM_JOB_PARTITION": "test",
        "SLURM_JOB_ACCOUNT": "test",
        "SLURM_JOB_NODELIST": "node[0-1]",
        "SLURM_STEP_ID": "7",
        "SLURMD_NODENAME": "caller-forgery",
        "PMI_RANK": "1",
        "PMIX_RANK": "1",
        "MPI_LOCALRANKID": "1",
        "OMPI_COMM_WORLD_RANK": "1",
        "SLURM_SUBMIT_DIR": str(tmp_path),
        "BASE_LOG_DIR": str(tmp_path),
        "CONTAINER": "test.sqsh",
        "MOUNTS": "",
        "GPUS_PER_NODE": "4",
        "CPUS_PER_WORKER": "8",
        "RAY_SUB_DAEMON_ENV_DIR": str(tmp_path / "daemon-env"),
        "RAY_SUB_OBSERVATIONS": str(tmp_path / "observations"),
        "RAY_SUB_SRUN_LOG": str(tmp_path / "srun"),
        "RAY_SUB_LOG_DIR": str(log_dir),
        "ray": "3",
        "head": "1",
        "workers": "0",
        "sandbox": "4",
    }


def _bash_3_compatibility_shim(path: Path) -> None:
    path.write_text(
        """declare() {
  if [[ "$1" == "-A" ]]; then
    shift
  fi
  builtin declare "$@"
}
"""
    )


@dataclass
class LauncherRun:
    returncode: int
    environment: dict[str, str]
    log_dir: Path
    ended_before_cleanup: bool


def _run_launcher(
    tmp_path: Path,
    *,
    job_id: str = "424242",
    caller_alias: str | None = None,
    ray_start_failures: int = 0,
    setup_timeout_seconds: int | None = None,
    skip_worker: bool = False,
    hold_worker_before_command: bool = False,
    stale_markers: bool = False,
    caller_marker_dir: Path | None = None,
    setup_command: str = 'printf "setup:%s\\n" "$SLURM_JOB_ID" >> "$RAY_SUB_OBSERVATIONS"',
    driver_command: str = 'printf "driver:%s\\n" "$SLURM_JOB_ID" >> "$RAY_SUB_OBSERVATIONS"',
) -> LauncherRun:
    bin_dir = tmp_path / "bin"
    daemon_env_dir = tmp_path / "daemon-env"
    bin_dir.mkdir()
    daemon_env_dir.mkdir()
    _write_fake_commands(bin_dir)
    shim = tmp_path / "bash-env"
    _bash_3_compatibility_shim(shim)
    env = _base_environment(tmp_path, bin_dir, job_id=job_id)
    env["BASH_ENV"] = str(shim)
    if caller_alias is not None:
        env["NATIVE_SLURM_JOB_ID"] = caller_alias
    env["SETUP_COMMAND"] = setup_command
    env["COMMAND"] = driver_command
    env["RAY_SUB_RAY_START_FAILURES"] = str(ray_start_failures)
    if setup_timeout_seconds is not None:
        env["SETUP_TIMEOUT_SECONDS"] = str(setup_timeout_seconds)
    if skip_worker:
        env["RAY_SUB_SKIP_WORKER"] = "1"
    if hold_worker_before_command:
        env["RAY_SUB_HOLD_WORKER_BEFORE_COMMAND"] = "1"
    log_dir = Path(env["RAY_SUB_LOG_DIR"])
    if stale_markers:
        log_dir.mkdir()
        (log_dir / "setup-complete-node0").touch()
        (log_dir / "setup-complete-node1").touch()
    if caller_marker_dir is not None:
        caller_marker_dir.mkdir()
        (caller_marker_dir / "node0").touch()
        (caller_marker_dir / "node1").touch()
        env["SETUP_MARKER_DIR"] = str(caller_marker_dir)
    outer_log = (tmp_path / "outer.log").open("w")
    process = subprocess.Popen(
        ["/bin/bash", str(RAY_SUB)],
        env=env,
        stdout=outer_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        returncode = process.wait(timeout=10)
        ended_before_cleanup = (log_dir / "ENDED").exists()
    finally:
        log_dir.mkdir(exist_ok=True)
        (log_dir / "ENDED").touch()
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        time.sleep(0.05)
        outer_log.close()
    return LauncherRun(returncode, env, log_dir, ended_before_cleanup)


def _run_invalid_launcher(
    tmp_path: Path, job_id: str
) -> tuple[subprocess.CompletedProcess[str], dict[str, str]]:
    bin_dir = tmp_path / "bin"
    daemon_env_dir = tmp_path / "daemon-env"
    bin_dir.mkdir()
    daemon_env_dir.mkdir()
    _write_fake_commands(bin_dir)
    shim = tmp_path / "bash-env"
    _bash_3_compatibility_shim(shim)
    env = _base_environment(tmp_path, bin_dir, job_id=job_id)
    env["BASH_ENV"] = str(shim)
    result = subprocess.run(
        ["/bin/bash", str(RAY_SUB)],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result, env


def _observations(env: dict[str, str]) -> list[str]:
    path = Path(env["RAY_SUB_OBSERVATIONS"])
    return path.read_text().splitlines() if path.exists() else []


def _daemon_environments(env: dict[str, str]) -> list[str]:
    return [path.read_text() for path in Path(env["RAY_SUB_DAEMON_ENV_DIR"]).iterdir()]


def _srun_events(env: dict[str, str]) -> list[str]:
    path = Path(env["RAY_SUB_SRUN_LOG"])
    return path.read_text().splitlines() if path.exists() else []


def test_real_srun_launch_preserves_only_native_id_for_user_hooks(tmp_path):
    """Fails if the real srun boundary leaks an alias or daemon scheduler state."""
    run = _run_launcher(tmp_path, caller_alias="999999")

    assert run.returncode == 0
    assert sorted(_observations(run.environment)) == [
        "driver:424242",
        "setup:424242",
        "setup:424242",
    ]
    marker_dirs = list(run.log_dir.glob(".setup-markers-*"))
    assert len(marker_dirs) == 1
    assert sorted(path.name for path in marker_dirs[0].iterdir()) == ["node0", "node1"]
    assert _daemon_environments(run.environment)
    forbidden = (
        "PMI_RANK=",
        "PMIX_RANK=",
        "MPI_LOCALRANKID=",
        "OMPI_COMM_WORLD_RANK=",
        "SLURM_JOB_ID=",
        "SLURM_STEP_ID=",
        "SLURMD_NODENAME=",
        "NATIVE_SLURM_JOB_ID=",
    )
    for daemon_environment in _daemon_environments(run.environment):
        assert not any(
            line.startswith(forbidden) for line in daemon_environment.splitlines()
        )


@pytest.mark.parametrize("job_id", ["", "not-a-decimal-job-id"])
def test_invalid_native_scheduler_id_fails_before_srun_rendering_or_launching(
    tmp_path, job_id
):
    """Fails if an invalid native ID gets past the renderer's entry validation."""
    result, env = _run_invalid_launcher(tmp_path, job_id)

    assert result.returncode != 0
    assert "SLURM_JOB_ID must be a non-empty decimal scheduler job ID" in result.stderr
    assert not _observations(env)
    assert not _daemon_environments(env)


def test_worker_setup_failure_blocks_ray_start_and_driver_allocation_wide(tmp_path):
    """Fails if head startup races ahead of a concurrent worker setup failure."""
    run = _run_launcher(
        tmp_path,
        setup_command='if [[ "$RAY_SUB_TEST_ROLE" == "worker" ]]; then /bin/sleep 0.5; exit 23; fi',
    )

    assert run.returncode != 0
    assert run.ended_before_cleanup
    assert not _daemon_environments(run.environment)
    assert not _observations(run.environment)


def test_head_setup_failure_signals_ended_before_ray_or_driver(tmp_path):
    """A failed head setup terminates a worker that successfully reached the barrier."""
    run = _run_launcher(
        tmp_path,
        setup_command='if [[ "$RAY_SUB_TEST_ROLE" == "head" ]]; then exit 23; fi; '
        'printf "worker-setup:%s\\n" "$SLURM_JOB_ID" >> "$RAY_SUB_OBSERVATIONS"',
    )

    assert run.returncode != 0
    assert run.ended_before_cleanup
    assert _observations(run.environment) == ["worker-setup:424242"]
    assert not _daemon_environments(run.environment)


def test_stale_root_markers_do_not_bypass_a_worker_setup_failure(tmp_path):
    """Fails if markers from an earlier launcher invocation satisfy this barrier."""
    run = _run_launcher(
        tmp_path,
        stale_markers=True,
        caller_marker_dir=tmp_path / "forged-markers",
        setup_command='if [[ "$RAY_SUB_TEST_ROLE" == "worker" ]]; then exit 23; fi',
    )

    assert run.returncode != 0
    assert run.ended_before_cleanup
    assert not _daemon_environments(run.environment)
    assert not _observations(run.environment)


def test_setup_barrier_timeout_signals_ended_before_ray_or_driver(tmp_path):
    """A live worker without a marker fails only through the setup-barrier timeout."""
    run = _run_launcher(
        tmp_path,
        setup_timeout_seconds=0,
        hold_worker_before_command=True,
    )

    assert run.returncode != 0
    assert run.ended_before_cleanup
    outer_log = (tmp_path / "outer.log").read_text()
    assert "Timed out waiting for setup completion" in outer_log
    assert "Background srun 'ray-workers'" not in outer_log
    assert sorted(_srun_events(run.environment)) == [
        "head",
        "worker",
        "worker-holding-before-command",
    ]
    assert _observations(run.environment) == ["setup:424242"]
    assert not _daemon_environments(run.environment)


def test_successful_setup_runs_once_when_the_head_retries_ray_start(tmp_path):
    """Fails if setup remains inside the head's Ray-start retry loop."""
    run = _run_launcher(tmp_path, ray_start_failures=2)

    assert run.returncode == 0
    assert _observations(run.environment).count("setup:424242") == 2
