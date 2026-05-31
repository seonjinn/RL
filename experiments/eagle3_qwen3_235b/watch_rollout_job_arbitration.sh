#!/usr/bin/env bash
set -euo pipefail

# Periodically refresh rollout queue/arbitration reports. When enabled, cancel
# only duplicate rollout jobs that are still PENDING after another rollout has
# started or canonical promotion has already been claimed.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/experiments/eagle3_qwen3_235b"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3}"
REPORT_DIR="${REPORT_DIR:-$ARTIFACT_ROOT/reports}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_POLLS="${MAX_POLLS:-720}"
AUTO_CANCEL_PENDING_DUPLICATES="${AUTO_CANCEL_PENDING_DUPLICATES:-false}"
RUN_OPERATOR_REFRESH="${RUN_OPERATOR_REFRESH:-true}"
LOCK_FILE="${LOCK_FILE:-$REPORT_DIR/rollout_job_arbitration_watch.lock}"

QUEUE_JSON="${QUEUE_JSON:-$REPORT_DIR/rollout_queue_wait_summary.json}"
QUEUE_MD="${QUEUE_MD:-$REPORT_DIR/rollout_queue_wait_summary.md}"
ARBITRATION_JSON="${ARBITRATION_JSON:-$REPORT_DIR/rollout_job_arbitration.json}"
ARBITRATION_MD="${ARBITRATION_MD:-$REPORT_DIR/rollout_job_arbitration.md}"

mkdir -p "$REPORT_DIR"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[$(date)] another arbitration watcher holds lock: $LOCK_FILE"
  exit 0
fi

cd "$ROOT_DIR"

active_rollout_count() {
  python3 - "$QUEUE_JSON" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
active = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "RESIZING"}
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print(0)
    raise SystemExit
count = 0
for job in data.get("jobs") or []:
    snapshot = job.get("current_squeue") if isinstance(job, dict) else {}
    if isinstance(snapshot, dict) and str(snapshot.get("state") or "").upper() in active:
        count += 1
print(count)
PY
}

arbitration_status() {
  python3 - "$ARBITRATION_JSON" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print("missing 0")
    raise SystemExit
print(str(data.get("overall_status") or "unknown"), len(data.get("cancel_candidates") or []))
PY
}

run_queue_refresh() {
  python3 "$SCRIPT_DIR/summarize_rollout_queue_wait.py" \
    --artifact-root "$ARTIFACT_ROOT" \
    --json-out "$QUEUE_JSON" \
    --markdown-out "$QUEUE_MD"
}

run_arbitration_report() {
  python3 "$SCRIPT_DIR/arbitrate_rollout_jobs.py" \
    --artifact-root "$ARTIFACT_ROOT" \
    --queue-json "$QUEUE_JSON" \
    --json-out "$ARBITRATION_JSON" \
    --markdown-out "$ARBITRATION_MD"
}

run_arbitration_cancel() {
  python3 "$SCRIPT_DIR/arbitrate_rollout_jobs.py" \
    --artifact-root "$ARTIFACT_ROOT" \
    --queue-json "$QUEUE_JSON" \
    --json-out "$ARBITRATION_JSON" \
    --markdown-out "$ARBITRATION_MD" \
    --execute-cancel \
    --allow-scancel
}

run_operator_refresh() {
  python3 "$SCRIPT_DIR/refresh_eagle3_operator_state.py" \
    --artifact-root "$ARTIFACT_ROOT" \
    --json-out "$REPORT_DIR/eagle3_operator_state_refresh.json" \
    --markdown-out "$REPORT_DIR/eagle3_operator_state_refresh.md"
}

echo "[$(date)] rollout arbitration watcher start auto_cancel=$AUTO_CANCEL_PENDING_DUPLICATES poll_seconds=$POLL_SECONDS max_polls=$MAX_POLLS"
for _ in $(seq 1 "$MAX_POLLS"); do
  run_queue_refresh
  run_arbitration_report
  read -r status candidate_count < <(arbitration_status)
  active_count="$(active_rollout_count)"
  echo "[$(date)] arbitration status=$status cancel_candidates=$candidate_count active_rollout_jobs=$active_count"

  if [[ "$candidate_count" != "0" && ( "$AUTO_CANCEL_PENDING_DUPLICATES" == "true" || "$AUTO_CANCEL_PENDING_DUPLICATES" == "True" || "$AUTO_CANCEL_PENDING_DUPLICATES" == "1" ) ]]; then
    echo "[$(date)] cancelling pending duplicate rollout jobs"
    run_arbitration_cancel
    if [[ "$RUN_OPERATOR_REFRESH" == "true" || "$RUN_OPERATOR_REFRESH" == "True" || "$RUN_OPERATOR_REFRESH" == "1" ]]; then
      run_operator_refresh
    fi
    echo "[$(date)] rollout arbitration watcher completed after cancellation"
    exit 0
  fi

  if (( active_count <= 1 )) && [[ "$status" != "action_recommended" ]]; then
    echo "[$(date)] no duplicate active rollout jobs remain; arbitration watcher exits"
    if [[ "$RUN_OPERATOR_REFRESH" == "true" || "$RUN_OPERATOR_REFRESH" == "True" || "$RUN_OPERATOR_REFRESH" == "1" ]]; then
      run_operator_refresh
    fi
    exit 0
  fi

  sleep "$POLL_SECONDS"
done

echo "[$(date)] rollout arbitration watcher timeout"
exit 2
