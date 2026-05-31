#!/usr/bin/env bash
set -euo pipefail

# Poll a submitted Slurm action and run the guarded follow-up only after the
# job disappears from squeue. This script never submits new Slurm work.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3}"
ACTION_ID="${ACTION_ID:-submit_rollout_capture}"
JOB_ID="${JOB_ID:-}"
POLL_SECONDS="${POLL_SECONDS:-120}"
MAX_POLLS="${MAX_POLLS:-240}"

PLAN_JSON="${PLAN_JSON:-$ARTIFACT_ROOT/reports/eagle3_next_actions.json}"
OPERATOR_SHEET_JSON="${OPERATOR_SHEET_JSON:-$ARTIFACT_ROOT/reports/eagle3_operator_sheet.json}"
EXECUTION_RECORD="${EXECUTION_RECORD:-$ARTIFACT_ROOT/reports/operator_execution/01_${ACTION_ID}.json}"
FOLLOWUP_JSON="${FOLLOWUP_JSON:-$ARTIFACT_ROOT/reports/operator_followups/01_${ACTION_ID}.json}"
FOLLOWUP_MARKDOWN="${FOLLOWUP_MARKDOWN:-$ARTIFACT_ROOT/reports/operator_followups/01_${ACTION_ID}.md}"
REFRESH_JSON="${REFRESH_JSON:-$ARTIFACT_ROOT/reports/eagle3_operator_state_refresh.json}"
REFRESH_MARKDOWN="${REFRESH_MARKDOWN:-$ARTIFACT_ROOT/reports/eagle3_operator_state_refresh.md}"
LOCK_FILE="${LOCK_FILE:-$ARTIFACT_ROOT/reports/operator_followups/01_${ACTION_ID}_watch.lock}"

if [[ -z "$JOB_ID" ]]; then
  echo "ERROR: JOB_ID is required" >&2
  exit 2
fi

mkdir -p "$(dirname "$FOLLOWUP_JSON")"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[$(date)] another watcher holds lock: $LOCK_FILE"
  exit 0
fi

echo "[$(date)] watcher start action=$ACTION_ID job=$JOB_ID poll_seconds=$POLL_SECONDS max_polls=$MAX_POLLS"

cd "$ROOT_DIR"

for _ in $(seq 1 "$MAX_POLLS"); do
  state="$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null || true)"
  if [[ -n "$state" ]]; then
    echo "[$(date)] job=$JOB_ID active state=$state"
    sleep "$POLL_SECONDS"
    continue
  fi

  echo "[$(date)] job=$JOB_ID no longer in squeue; running guarded follow-up"
  python3 experiments/eagle3_qwen3_235b/run_eagle3_slurm_followups.py \
    --artifact-root "$ARTIFACT_ROOT" \
    --plan-json "$PLAN_JSON" \
    --operator-sheet-json "$OPERATOR_SHEET_JSON" \
    --action-id "$ACTION_ID" \
    --execution-record "$EXECUTION_RECORD" \
    --json-out "$FOLLOWUP_JSON" \
    --markdown-out "$FOLLOWUP_MARKDOWN" \
    --execute-after

  python3 experiments/eagle3_qwen3_235b/refresh_eagle3_operator_state.py \
    --artifact-root "$ARTIFACT_ROOT" \
    --json-out "$REFRESH_JSON" \
    --markdown-out "$REFRESH_MARKDOWN"

  echo "[$(date)] watcher completed follow-up refresh"
  exit 0
done

echo "[$(date)] watcher timeout before terminal state: job=$JOB_ID"
exit 2
