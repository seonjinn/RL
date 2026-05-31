#!/usr/bin/env bash
set -euo pipefail

# Refresh rollout_capture_state_advance reports while a rollout Slurm job is
# still pending/running. This helper does not materialize data and exits as soon
# as the job leaves squeue, leaving terminal handling to
# watch_rollout_capture_materialize.sh.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3}"
SWE_REPO_ROOT="${SWE_REPO_ROOT:-${REPO_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL}}"
JOB_ID="${JOB_ID:?set JOB_ID}"
ROLLOUT_LOG_DIR="${ROLLOUT_LOG_DIR:?set ROLLOUT_LOG_DIR}"
OUTPUT_CONVERSATIONS="${OUTPUT_CONVERSATIONS:?set OUTPUT_CONVERSATIONS}"
REPORT_PREFIX="${REPORT_PREFIX:-}"
POLL_SECONDS="${POLL_SECONDS:-120}"
MAX_POLLS="${MAX_POLLS:-240}"
WAIT_FOR_LOCK="${WAIT_FOR_LOCK:-false}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-coreai_dlalgo_nemorl}"
SBATCH_PARTITION="${SBATCH_PARTITION:-batch}"

REPORT_DIR="$ARTIFACT_ROOT/reports"
CANONICAL_OUTPUT_CONVERSATIONS="$ARTIFACT_ROOT/data/qwen3_235b_swe_rollout_conversations.jsonl"

if [[ -z "${STATE_JSON:-}" ]]; then
  if [[ "$OUTPUT_CONVERSATIONS" == "$CANONICAL_OUTPUT_CONVERSATIONS" ]]; then
    STATE_JSON="$REPORT_DIR/rollout_capture_state_advance.json"
  else
    STATE_JSON="$REPORT_DIR/${REPORT_PREFIX:-rollout_capture_${JOB_ID}}_state_advance.json"
  fi
fi
STATE_MD="${STATE_MD:-${STATE_JSON%.json}.md}"
LOCK_FILE="${LOCK_FILE:-$REPORT_DIR/rollout_pending_state_${JOB_ID}.lock}"
WATCH_PID_FILE="${WATCH_PID_FILE:-$REPORT_DIR/${REPORT_PREFIX:-rollout_capture_${JOB_ID}}_pending_state_watch.pid}"

REPORT_PREFIX_ARGS=()
if [[ -n "$REPORT_PREFIX" && "$STATE_JSON" != "$REPORT_DIR/rollout_capture_state_advance.json" ]]; then
  REPORT_PREFIX_ARGS=(--report-prefix "$REPORT_PREFIX")
fi

mkdir -p "$REPORT_DIR"

exec 9>"$LOCK_FILE"
if [[ "$WAIT_FOR_LOCK" == "true" || "$WAIT_FOR_LOCK" == "True" || "$WAIT_FOR_LOCK" == "1" ]]; then
  echo "[$(date)] waiting for pending-state watcher lock: $LOCK_FILE"
  flock 9
else
  if ! flock -n 9; then
    echo "[$(date)] another pending-state watcher holds lock: $LOCK_FILE"
    exit 0
  fi
fi
printf '%s\n' "$$" > "$WATCH_PID_FILE"
cleanup_watch_pid() {
  if [[ -f "$WATCH_PID_FILE" ]] && [[ "$(cat "$WATCH_PID_FILE" 2>/dev/null || true)" == "$$" ]]; then
    rm -f "$WATCH_PID_FILE"
  fi
}
trap cleanup_watch_pid EXIT

cd "$ROOT_DIR"

refresh_state() {
  python3 experiments/eagle3_qwen3_235b/advance_rollout_capture_state.py \
    --artifact-root "$ARTIFACT_ROOT" \
    --repo-root "$SWE_REPO_ROOT" \
    --job-id "$JOB_ID" \
    --rollout-log-dir "$ROLLOUT_LOG_DIR" \
    --output-data "$OUTPUT_CONVERSATIONS" \
    "${REPORT_PREFIX_ARGS[@]}" \
    --sbatch-account "$SBATCH_ACCOUNT" \
    --sbatch-partition "$SBATCH_PARTITION" \
    --json-out "$STATE_JSON" \
    --markdown-out "$STATE_MD"
}

echo "[$(date)] pending-state watcher start job=$JOB_ID state_json=$STATE_JSON poll_seconds=$POLL_SECONDS max_polls=$MAX_POLLS"
for _ in $(seq 1 "$MAX_POLLS"); do
  state="$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null || true)"
  if [[ -z "$state" ]]; then
    echo "[$(date)] job=$JOB_ID no longer in squeue; pending-state watcher exits"
    exit 0
  fi
  start_time="$(squeue -j "$JOB_ID" -h -o "%S" 2>/dev/null || true)"
  reason="$(squeue -j "$JOB_ID" -h -o "%R" 2>/dev/null || true)"
  echo "[$(date)] job=$JOB_ID active state=$state start=${start_time:-unknown} reason=${reason:-unknown}; refreshing $STATE_JSON"
  refresh_state || echo "[$(date)] pending-state refresh failed for job=$JOB_ID"
  sleep "$POLL_SECONDS"
done

echo "[$(date)] pending-state watcher timeout before terminal state: job=$JOB_ID"
exit 2
