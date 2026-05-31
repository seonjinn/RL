#!/usr/bin/env bash
set -euo pipefail

# Watch one vLLM source-build job. If it times out before writing a PASS report,
# submit a longer retry and attach the normal source-build -> ABI -> rollout
# watcher to the retry. Non-timeout failures are left for log inspection.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/experiments/eagle3_qwen3_235b"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3}"
REPORT_DIR="${REPORT_DIR:-$ARTIFACT_ROOT/reports}"
SOURCE_JOB_FILE="${SOURCE_JOB_FILE:-$ROOT_DIR/latest_vllm_native_source_build_job.txt}"
SOURCE_JOB_ID="${SOURCE_JOB_ID:-}"
SOURCE_BUILD_JSON="${SOURCE_BUILD_JSON:-$REPORT_DIR/vllm_native_source_build.json}"
SOURCE_BUILD_MD="${SOURCE_BUILD_MD:-$REPORT_DIR/vllm_native_source_build.md}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_POLLS="${MAX_POLLS:-300}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-coreai_dlalgo_nemorl}"
SBATCH_PARTITION="${SBATCH_PARTITION:-batch}"
RETRY_SBATCH_TIME="${RETRY_SBATCH_TIME:-06:00:00}"
RETRY_WATCH_MAX_POLLS="${RETRY_WATCH_MAX_POLLS:-600}"
LOG_FILE="${LOG_FILE:-$REPORT_DIR/watch_vllm_source_build_retry_on_timeout.log}"
PID_FILE="${PID_FILE:-$REPORT_DIR/watch_vllm_source_build_retry_on_timeout.pid}"
RETRY_SUBMITTED_FILE="${RETRY_SUBMITTED_FILE:-$REPORT_DIR/vllm_source_build_timeout_retry.env}"
ANALYSIS_JSON="${ANALYSIS_JSON:-$REPORT_DIR/vllm_source_build_job_analysis.json}"
ANALYSIS_MD="${ANALYSIS_MD:-$REPORT_DIR/vllm_source_build_job_analysis.md}"
REFRESH_ANALYSIS_EVERY_POLLS="${REFRESH_ANALYSIS_EVERY_POLLS:-5}"

mkdir -p "$REPORT_DIR"
printf "%s\n" "$$" > "$PID_FILE"

kv_value() {
  awk -F= -v key="$2" '$1 == key {print $2; exit}' "$1" 2>/dev/null || true
}

if [[ -z "$SOURCE_JOB_ID" ]]; then
  SOURCE_JOB_ID="$(kv_value "$SOURCE_JOB_FILE" vllm_native_source_build_job)"
fi
if [[ -z "$SOURCE_JOB_ID" ]]; then
  echo "[$(date)] Could not determine SOURCE_JOB_ID from $SOURCE_JOB_FILE" | tee -a "$LOG_FILE" >&2
  exit 1
fi

json_status() {
  python3 - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("missing")
    raise SystemExit(0)
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception as exc:
    print(f"invalid:{exc}")
    raise SystemExit(0)
print(payload.get("overall_status") or payload.get("status") or "unknown")
PY
}

job_state() {
  local job_id="$1"
  local state
  state="$(squeue -j "$job_id" -h -o "%T" 2>/dev/null | head -1 || true)"
  if [[ -n "$state" ]]; then
    echo "$state"
    return 0
  fi
  sacct -j "$job_id" --format=State -P -n 2>/dev/null | head -1 | cut -d'|' -f1 || true
}

log() {
  echo "[$(date)] $*" | tee -a "$LOG_FILE"
}

write_analysis() {
  python3 "$SCRIPT_DIR/analyze_vllm_source_build_job.py" \
    --artifact-root "$ARTIFACT_ROOT" \
    --job-id "$SOURCE_JOB_ID" \
    --json-out "$ANALYSIS_JSON" \
    --markdown-out "$ANALYSIS_MD" >/dev/null || true
}

submit_retry() {
  log "submitting vLLM source-build retry with SBATCH_TIME=$RETRY_SBATCH_TIME"
  ARTIFACT_ROOT="$ARTIFACT_ROOT" \
  SBATCH_ACCOUNT="$SBATCH_ACCOUNT" \
  SBATCH_PARTITION="$SBATCH_PARTITION" \
  SBATCH_TIME="$RETRY_SBATCH_TIME" \
  SUBMIT=true \
    bash "$SCRIPT_DIR/submit_vllm_native_source_build.sh" | tee -a "$LOG_FILE"

  local retry_job
  retry_job="$(kv_value "$SOURCE_JOB_FILE" vllm_native_source_build_job)"
  if [[ -z "$retry_job" || "$retry_job" == "$SOURCE_JOB_ID" ]]; then
    log "retry job id was not recorded correctly in $SOURCE_JOB_FILE"
    exit 1
  fi
  {
    echo "previous_source_job=$SOURCE_JOB_ID"
    echo "retry_source_job=$retry_job"
    echo "retry_sbatch_time=$RETRY_SBATCH_TIME"
    echo "submitted_at=$(date)"
  } > "$RETRY_SUBMITTED_FILE"

  local retry_log="$REPORT_DIR/watch_vllm_source_build_${retry_job}_then_rollout.log"
  log "starting normal source-build watcher for retry job $retry_job"
  (
    cd "$ROOT_DIR"
    SOURCE_JOB_ID="$retry_job" \
    ARTIFACT_ROOT="$ARTIFACT_ROOT" \
    SBATCH_ACCOUNT="$SBATCH_ACCOUNT" \
    SBATCH_PARTITION="$SBATCH_PARTITION" \
    MAX_POLLS="$RETRY_WATCH_MAX_POLLS" \
      bash "$SCRIPT_DIR/watch_vllm_source_build_then_rollout.sh"
  ) >>"$retry_log" 2>&1 &
  echo "$!" > "$REPORT_DIR/watch_vllm_source_build_${retry_job}_then_rollout.pid"
  log "retry watcher pid=$(cat "$REPORT_DIR/watch_vllm_source_build_${retry_job}_then_rollout.pid") log=$retry_log"
}

for i in $(seq 1 "$MAX_POLLS"); do
  status="$(json_status "$SOURCE_BUILD_JSON")"
  state="$(job_state "$SOURCE_JOB_ID")"
  if [[ "$i" == "1" || $((i % REFRESH_ANALYSIS_EVERY_POLLS)) -eq 0 ]]; then
    write_analysis
  fi
  log "source-build job=$SOURCE_JOB_ID state=${state:-unknown} report=$status poll=$i/$MAX_POLLS"
  if [[ "$status" == "pass" ]]; then
    log "source build already passed; normal watcher should handle ABI probe and rollout"
    exit 0
  fi
  case "$state" in
    ""|PENDING|CONFIGURING|RUNNING|COMPLETING|SUSPENDED)
      sleep "$POLL_SECONDS"
      ;;
    TIMEOUT|TIMEOUT*|CANCELLED|CANCELLED*)
      write_analysis
      if [[ -e "$RETRY_SUBMITTED_FILE" ]]; then
        log "retry marker already exists at $RETRY_SUBMITTED_FILE; not submitting another retry"
        exit 0
      fi
      submit_retry
      exit 0
      ;;
    *)
      write_analysis
      log "source build ended as $state without PASS; inspect $SOURCE_BUILD_MD and Slurm logs before retry"
      exit 0
      ;;
  esac
done

log "watchdog reached polling budget without terminal source-build state"
