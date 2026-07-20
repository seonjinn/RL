#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from openhands.events.action import CmdRunAction
from openhands.runtime.utils.bash import BashSession


def _select_runnable_jq() -> Path:
    candidates = (
        Path("/usr/bin/jq"),
        Path("/openhands_setup/miniforge3/bin/jq"),
    )
    selected: Path | None = None
    for candidate in candidates:
        try:
            result = subprocess.run(
                [str(candidate), "--version"],
                check=False,
                text=True,
                capture_output=True,
                timeout=5,
            )
            detail = (result.stdout or result.stderr).strip()
            print(
                f"jq_probe={candidate} returncode={result.returncode} detail={detail}"
            )
            if result.returncode == 0 and selected is None:
                selected = candidate
        except (OSError, subprocess.TimeoutExpired) as error:
            print(f"jq_probe={candidate} error={error}")
    if selected is None:
        raise RuntimeError("no runnable jq found in the SWE container")

    tool_dir = Path("/tmp/nemorl-native-tools")
    tool_dir.mkdir(parents=True, exist_ok=True)
    jq_link = tool_dir / "jq"
    jq_link.unlink(missing_ok=True)
    jq_link.symlink_to(selected)
    os.environ["PATH"] = f"{tool_dir}:{os.environ.get('PATH', '')}"
    print(f"selected_jq={selected}")
    return selected


def _load_instance(dataset_path: Path) -> dict[str, Any]:
    record = json.loads(dataset_path.read_text().splitlines()[0])
    metadata = record["responses_create_params"]["metadata"]
    instance = json.loads(metadata["instance_dict"])
    if not isinstance(instance, dict):
        raise TypeError("instance_dict must decode to an object")
    return instance


def _workspace_path(instance: dict[str, Any]) -> Path:
    repo = str(instance["repo"])
    version = str(instance["version"])
    return Path("/workspace") / f"{repo}__{version}".replace("/", "__")


def _prepare_swe_util(entry_script: Path, instance: dict[str, Any]) -> None:
    swe_util = Path("/swe_util")
    instances_dir = swe_util / "eval_data" / "instances"
    instances_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(entry_script, swe_util / "instance_swe_entry.sh")
    (instances_dir / "swe-bench-instance.json").write_text(json.dumps([instance]))


def _run_direct_source(workspace: Path) -> float:
    shutil.rmtree(workspace, ignore_errors=True)
    started = time.perf_counter()
    result = subprocess.run(
        [
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            "set -x; source /swe_util/instance_swe_entry.sh",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
    )
    elapsed = time.perf_counter() - started
    print(f"direct_source_elapsed_s={elapsed:.3f}")
    print(f"direct_source_returncode={result.returncode}")
    print("direct_source_stdout_begin")
    print(result.stdout.rstrip())
    print("direct_source_stdout_end")
    print("direct_source_stderr_begin")
    print(result.stderr.rstrip())
    print("direct_source_stderr_end")
    if result.returncode != 0:
        raise RuntimeError("direct entry-script execution failed")
    return elapsed


def _run_openhands_source(workspace: Path, timeout_s: int) -> tuple[float, int]:
    shutil.rmtree(workspace, ignore_errors=True)
    socket_root = Path(tempfile.mkdtemp(prefix="nemorl-swe-entry-tmux-"))
    os.environ["TMUX_TMPDIR"] = str(socket_root)
    os.environ.pop("TMUX", None)

    session = BashSession(work_dir="/tmp")
    session.initialize()
    try:
        prompt_probe = CmdRunAction(
            command=(
                "printf 'PROMPT_COMMAND_before=%s\\n' \"$PROMPT_COMMAND\"; "
                "source ~/.bashrc; "
                "printf 'PROMPT_COMMAND_after=%s\\n' \"$PROMPT_COMMAND\""
            )
        )
        prompt_probe.set_hard_timeout(10)
        prompt_observation = session.execute(prompt_probe)
        print("prompt_probe_observation_begin")
        print(prompt_observation)
        print("prompt_probe_observation_end")

        action = CmdRunAction(command="source /swe_util/instance_swe_entry.sh")
        action.set_hard_timeout(timeout_s)
        started = time.perf_counter()
        observation = session.execute(action)
        elapsed = time.perf_counter() - started
        exit_code = int(observation.metadata.exit_code)
        print(f"openhands_source_elapsed_s={elapsed:.3f}")
        print(f"openhands_source_exit_code={exit_code}")
        print("openhands_source_observation_begin")
        print(observation)
        print("openhands_source_observation_end")
        return elapsed, exit_code
    finally:
        session.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare direct and OpenHands BashSession SWE entry setup"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--entry-script", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=20)
    args = parser.parse_args()

    instance = _load_instance(args.dataset)
    instance_id = str(instance["instance_id"])
    os.environ["SWE_INSTANCE_ID"] = instance_id
    _prepare_swe_util(args.entry_script, instance)
    _select_runnable_jq()
    workspace = _workspace_path(instance)

    print(f"instance_id={instance_id}")
    print(f"workspace={workspace}")
    _run_direct_source(workspace)
    _, exit_code = _run_openhands_source(workspace, args.timeout_seconds)
    if exit_code != 0:
        raise RuntimeError(f"OpenHands entry-script execution exited {exit_code}")
    print("result=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
