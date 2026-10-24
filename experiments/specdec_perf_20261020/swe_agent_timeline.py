"""Split each SWE agent trajectory into startup, runtime setup, controller and teardown phases.

    python swe_agent_timeline.py <swebench_results_dir> [out.csv]

<swebench_results_dir> is the per-run directory that the swe_agents server logs as
"SWE agents results root for this run". The W&B openhands_run_time starts before
the agent container and ends after the container exits, so it holds phases that
no W&B metric reports. This script rebuilds them from the per-trajectory files:
  spinup     gen start -> apptainer start (generation_apptainer_spinup_timestamp)
  pre        apptainer start -> OpenHands create_runtime (shell script, python imports)
  connect    OpenHands connect_to_runtime (action execution server start)
  init       OpenHands initialize_runtime (instance setup commands)
  model      LLM calls inside the controller
  exec       tool commands inside the controller
  ctrl_other controller time outside model and exec
  post       controller end -> agent container exit
"""

import csv
import glob
import json
import re
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

PHASES = ["spinup", "pre", "connect", "init", "model", "exec", "ctrl_other", "post"]
FIELDS = ["dir", "start", "turns", "openhands_run", *PHASES]
CONTROLLER = re.compile(r"run controller: ([0-9.]+) seconds")


def epoch(ts: str) -> float:
    dt = datetime.fromisoformat(ts)
    # OpenHands history timestamps are naive UTC
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()


def trajectory(d: Path) -> dict[str, float] | None:
    try:
        m = json.loads((d / "nemo_gym_metrics.json").read_text())
        t0 = epoch(m["generation_start_timestamp"])
        t_app = float((d / "generation_apptainer_spinup_timestamp").read_text())
        out = json.loads(
            open(glob.glob(str(d / "trajectories/*/output.jsonl"))[0]).readline()
        )
        log = "".join(
            p.read_text(errors="ignore")
            for p in (d / "apptainer_logs").glob("*_agent.log")
        )
    except (OSError, IndexError, KeyError, ValueError):
        return None
    history = [e for e in out.get("history") or [] if e.get("timestamp")]
    found = CONTROLLER.search(log)
    if not history or not found:
        return None
    setup = (
        m["create_runtime_time"]
        + m["connect_to_runtime_time"]
        + m["initialize_runtime_time"]
    )
    ctrl_start = epoch(history[0]["timestamp"])
    controller = float(found.group(1))
    model = m["total_model_call_time"]
    exec_ = m["total_command_exec_time"]
    return {
        "start": t0,
        "turns": len((out.get("metrics") or {}).get("response_latencies") or []),
        "openhands_run": m["openhands_run_time"],
        "spinup": t_app - t0,
        "pre": ctrl_start - setup - t_app,
        "connect": m["create_runtime_time"] + m["connect_to_runtime_time"],
        "init": m["initialize_runtime_time"],
        "model": model,
        "exec": exec_,
        "ctrl_other": controller - model - exec_,
        "post": t0 + m["openhands_run_time"] - ctrl_start - controller,
    }


def main() -> None:
    root = Path(sys.argv[1])
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        t = trajectory(d)
        if t is not None:
            rows.append({"dir": d.name, **t})
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    print(f"{len(rows)} trajectories with a full timeline")
    print(f"{'phase':12s} {'mean':>7s} {'p50':>7s} {'p90':>7s} {'max':>7s}")
    for k in ["openhands_run", *PHASES, "turns"]:
        xs = sorted(r[k] for r in rows)
        print(
            f"{k:12s} {st.mean(xs):7.1f} {xs[len(xs) // 2]:7.1f} {xs[int(0.9 * len(xs))]:7.1f} {xs[-1]:7.1f}"
        )
    per_turn = [r["ctrl_other"] / r["turns"] for r in rows if r["turns"]]
    print(f"ctrl_other per turn mean {st.mean(per_turn):.2f} s")


if __name__ == "__main__":
    main()
