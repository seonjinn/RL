"""Submit authorized broker jobs sequentially after successful predecessor exits."""

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    state: dict[str, Any] = {
        "status": "monitoring",
        "predecessor": plan["predecessor"],
        "submissions": [],
        "observations": [],
    }
    if args.state.exists():
        raise RuntimeError("State already exists; reconcile jobs before any restart")

    def save() -> None:
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        temporary = args.state.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(args.state)

    def call(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, text=True, capture_output=True, timeout=180)

    predecessor = str(plan["predecessor"])
    failures = 0
    next_trial = 0
    save()
    try:
        while True:
            result = call(
                [plan["brr"], "query", plan["cluster"], "sacct", "-j", predecessor,
                 "--format=JobID,State,ExitCode", "--parsable2"]
            )
            if result.returncode:
                failures += 1
                state["last_error"] = (result.stdout + result.stderr)[-4000:]
                save()
                if failures >= 3:
                    raise RuntimeError("Three read failures; manual reconciliation required")
                time.sleep(120)
                continue
            rows = [row.split("|") for row in result.stdout.splitlines()]
            rows = [row for row in rows if len(row) >= 3 and row[0] == predecessor]
            if not rows:
                raise RuntimeError(f"No accounting row for known job {predecessor}")
            _, job_state, exit_code = rows[0][:3]
            failures = 0
            state["observations"].append({
                "job_id": predecessor, "state": job_state, "exit_code": exit_code,
                "utc": datetime.now(timezone.utc).isoformat(),
            })
            save()
            print(f"{predecessor}: {job_state}, exit {exit_code}", flush=True)
            if job_state == "COMPLETED" and exit_code == "0:0":
                if next_trial == len(plan["trials"]):
                    state["status"] = "all_jobs_exited_successfully"
                    save()
                    return
                trial = plan["trials"][next_trial]
                state["status"] = "submitting"
                state["submission_in_flight"] = trial["name"]
                save()
                submitted = call(trial["command"])
                output = submitted.stdout.strip()
                if submitted.returncode or not output.isdigit():
                    state["submission_response"] = {
                        "returncode": submitted.returncode,
                        "stdout": output, "stderr": submitted.stderr,
                    }
                    raise RuntimeError("Submission not confirmed; do not automatically retry")
                predecessor = output
                state["submissions"].append({
                    "name": trial["name"], "job_id": predecessor,
                    "arms": trial["arms"],
                    "submitted_utc": datetime.now(timezone.utc).isoformat(),
                })
                state.pop("submission_in_flight", None)
                state["predecessor"] = predecessor
                state["status"] = "monitoring"
                next_trial += 1
                save()
                print(f"Submitted {trial['name']}: {predecessor}", flush=True)
            elif job_state not in {
                "PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED",
                "RESIZING", "REQUEUED", "REQUEUE_FED", "REQUEUE_HOLD", "SIGNALING",
            }:
                raise RuntimeError(f"Predecessor {predecessor}: {job_state} / {exit_code}")
            time.sleep(120)
    except Exception as error:
        state["status"] = "needs_attention"
        state["error"] = str(error)
        save()
        raise


if __name__ == "__main__":
    main()
