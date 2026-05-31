#!/usr/bin/env bash
set -euo pipefail

# Wait for the canonical rollout corpus and pipeline submit preflight to become
# ready, then submit the Eagle3 hidden-state/train/export pipeline through the
# strict gated helper. This watcher never bypasses submit_ready=true.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/experiments/eagle3_qwen3_235b"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3}"
REPORT_DIR="${REPORT_DIR:-$ARTIFACT_ROOT/reports}"
POLL_SECONDS="${POLL_SECONDS:-120}"
MAX_POLLS="${MAX_POLLS:-720}"
REFRESH_EVERY_POLLS="${REFRESH_EVERY_POLLS:-5}"
RUN_OPERATOR_REFRESH="${RUN_OPERATOR_REFRESH:-true}"
LOCK_FILE="${LOCK_FILE:-$REPORT_DIR/eagle3_pipeline_ready_submit_watch.lock}"

PREFLIGHT_JSON="${PREFLIGHT_JSON:-$REPORT_DIR/eagle3_pipeline_submit_preflight.json}"
GATED_JSON="${GATED_JSON:-$REPORT_DIR/eagle3_pipeline_gated_submit.json}"
GATED_MD="${GATED_MD:-$REPORT_DIR/eagle3_pipeline_gated_submit.md}"
WATCH_PID_FILE="${WATCH_PID_FILE:-$REPORT_DIR/eagle3_pipeline_ready_submit_watch.pid}"

mkdir -p "$REPORT_DIR"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[$(date)] another pipeline-ready submit watcher holds lock: $LOCK_FILE"
  exit 0
fi
printf '%s\n' "$$" > "$WATCH_PID_FILE"
cleanup_watch_pid() {
  if [[ -f "$WATCH_PID_FILE" ]] && [[ "$(cat "$WATCH_PID_FILE" 2>/dev/null || true)" == "$$" ]]; then
    rm -f "$WATCH_PID_FILE"
  fi
}
trap cleanup_watch_pid EXIT

cd "$ROOT_DIR"

pipeline_already_submitted() {
  python3 - "$GATED_JSON" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print("false")
    raise SystemExit
jobs = data.get("jobs") if isinstance(data.get("jobs"), dict) else {}
required = {"dump_job", "train_job", "export_job"}
ok = data.get("overall_status") == "pass" and data.get("executed") is True and required.issubset(jobs)
print("true" if ok else "false")
PY
}

pipeline_preflight_ready() {
  python3 - "$PREFLIGHT_JSON" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print("false missing")
    raise SystemExit
status = str(data.get("overall_status") or "unknown")
ready = data.get("submit_ready") is True and status == "pass"
input_data = str(data.get("input_data") or "")
print(("true" if ready else "false"), status, str(data.get("submit_ready")), input_data)
PY
}

run_gated_readiness_check() {
  python3 "$SCRIPT_DIR/submit_eagle3_pipeline_if_ready.py" \
    --artifact-root "$ARTIFACT_ROOT" \
    --preflight-json "$PREFLIGHT_JSON" \
    --json-out "$GATED_JSON" \
    --markdown-out "$GATED_MD" \
    --exit-zero-if-not-ready
}

submit_pipeline() {
  python3 "$SCRIPT_DIR/submit_eagle3_pipeline_if_ready.py" \
    --artifact-root "$ARTIFACT_ROOT" \
    --preflight-json "$PREFLIGHT_JSON" \
    --json-out "$GATED_JSON" \
    --markdown-out "$GATED_MD" \
    --execute \
    --allow-heavy-gpu
}

run_operator_refresh() {
  python3 "$SCRIPT_DIR/refresh_eagle3_operator_state.py" \
    --artifact-root "$ARTIFACT_ROOT" \
    --json-out "$REPORT_DIR/eagle3_operator_state_refresh.json" \
    --markdown-out "$REPORT_DIR/eagle3_operator_state_refresh.md"
}

echo "[$(date)] pipeline-ready submit watcher start poll_seconds=$POLL_SECONDS max_polls=$MAX_POLLS"
for poll in $(seq 1 "$MAX_POLLS"); do
  if [[ "$(pipeline_already_submitted)" == "true" ]]; then
    echo "[$(date)] pipeline already submitted according to $GATED_JSON"
    exit 0
  fi

  read -r ready status submit_ready input_data < <(pipeline_preflight_ready)
  echo "[$(date)] preflight ready=$ready status=$status submit_ready=$submit_ready input_data=${input_data:-unknown}"
  if [[ "$ready" == "true" ]]; then
    echo "[$(date)] submitting Eagle3 pipeline through gated helper"
    submit_pipeline
    if [[ "$RUN_OPERATOR_REFRESH" == "true" || "$RUN_OPERATOR_REFRESH" == "True" || "$RUN_OPERATOR_REFRESH" == "1" ]]; then
      run_operator_refresh
    fi
    echo "[$(date)] pipeline-ready submit watcher completed"
    exit 0
  fi

  run_gated_readiness_check || true
  if (( REFRESH_EVERY_POLLS > 0 && poll % REFRESH_EVERY_POLLS == 0 )); then
    echo "[$(date)] refreshing operator state while waiting for pipeline readiness"
    run_operator_refresh || echo "[$(date)] operator refresh failed while waiting"
  fi
  sleep "$POLL_SECONDS"
done

echo "[$(date)] pipeline-ready submit watcher timeout"
exit 2
