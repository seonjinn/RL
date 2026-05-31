#!/usr/bin/env bash
set -euo pipefail

# Watch the canonical vLLM source-build job. If it reaches a non-timeout
# terminal state without PASS, optionally submit the vLLM 0.13.0 source-build
# candidate and attach the 0.13.0 source-build -> ABI -> rollout watcher.
#
# TIMEOUT/CANCELLED are intentionally left to watch_vllm_source_build_retry_on_timeout.sh.

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
AUTO_SUBMIT_013_ON_FAIL="${AUTO_SUBMIT_013_ON_FAIL:-false}"
LOG_FILE="${LOG_FILE:-$REPORT_DIR/watch_vllm_source_build_fallback_0_13_0.log}"
PID_FILE="${PID_FILE:-$REPORT_DIR/watch_vllm_source_build_fallback_0_13_0.pid}"
FALLBACK_PLAN_FILE="${FALLBACK_PLAN_FILE:-$REPORT_DIR/vllm_source_build_fallback_0_13_0.env}"
FALLBACK_REPORT_MD="${FALLBACK_REPORT_MD:-$REPORT_DIR/vllm_source_build_fallback_0_13_0.md}"
FALLBACK_JOB_FILE="${FALLBACK_JOB_FILE:-$ROOT_DIR/latest_vllm_native_source_build_0_13_0_job.txt}"
FALLBACK_WATCH_MAX_POLLS="${FALLBACK_WATCH_MAX_POLLS:-600}"

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

write_plan() {
  local state="$1"
  {
    echo "source_job=$SOURCE_JOB_ID"
    echo "source_state=$state"
    echo "source_json=$SOURCE_BUILD_JSON"
    echo "source_markdown=$SOURCE_BUILD_MD"
    echo "fallback_version=0.13.0"
    echo "fallback_job_file=$FALLBACK_JOB_FILE"
    echo "auto_submit_013_on_fail=$AUTO_SUBMIT_013_ON_FAIL"
    echo "recorded_at=$(date)"
  } > "$FALLBACK_PLAN_FILE"
  cat > "$FALLBACK_REPORT_MD" <<EOF
# vLLM 0.13.0 Fallback Plan

Source job: \`$SOURCE_JOB_ID\`
Source state: \`$state\`
Source report: \`$SOURCE_BUILD_JSON\`

The canonical vLLM source-build did not produce PASS. The higher-version
candidate is vLLM 0.13.0, using:

\`\`\`bash
SUBMIT=true \\
SBATCH_ACCOUNT=$SBATCH_ACCOUNT \\
SBATCH_PARTITION=$SBATCH_PARTITION \\
bash experiments/eagle3_qwen3_235b/submit_vllm_native_source_build_0_13_0.sh
\`\`\`

Automatic submit enabled: \`$AUTO_SUBMIT_013_ON_FAIL\`
EOF
}

submit_013() {
  log "submitting vLLM 0.13.0 fallback source build"
  ARTIFACT_ROOT="$ARTIFACT_ROOT" \
  SBATCH_ACCOUNT="$SBATCH_ACCOUNT" \
  SBATCH_PARTITION="$SBATCH_PARTITION" \
  SUBMIT=true \
    bash "$SCRIPT_DIR/submit_vllm_native_source_build_0_13_0.sh" | tee -a "$LOG_FILE"

  local fallback_job
  fallback_job="$(kv_value "$FALLBACK_JOB_FILE" vllm_native_source_build_job)"
  if [[ -z "$fallback_job" || "$fallback_job" == "VLLM_SOURCE_BUILD_JOB_ID" ]]; then
    log "fallback job id was not recorded correctly in $FALLBACK_JOB_FILE"
    exit 1
  fi

  local fallback_log="$REPORT_DIR/watch_vllm_source_build_${fallback_job}_0_13_0_then_rollout.log"
  log "starting vLLM 0.13.0 watcher for fallback job $fallback_job"
  (
    cd "$ROOT_DIR"
    SOURCE_JOB_ID="$fallback_job" \
    ARTIFACT_ROOT="$ARTIFACT_ROOT" \
    SBATCH_ACCOUNT="$SBATCH_ACCOUNT" \
    SBATCH_PARTITION="$SBATCH_PARTITION" \
    MAX_POLLS="$FALLBACK_WATCH_MAX_POLLS" \
      bash "$SCRIPT_DIR/watch_vllm_source_build_0_13_0_then_rollout.sh"
  ) >>"$fallback_log" 2>&1 &
  echo "$!" > "$REPORT_DIR/watch_vllm_source_build_${fallback_job}_0_13_0_then_rollout.pid"
  log "fallback watcher pid=$(cat "$REPORT_DIR/watch_vllm_source_build_${fallback_job}_0_13_0_then_rollout.pid") log=$fallback_log"
}

for i in $(seq 1 "$MAX_POLLS"); do
  status="$(json_status "$SOURCE_BUILD_JSON")"
  state="$(job_state "$SOURCE_JOB_ID")"
  log "source-build job=$SOURCE_JOB_ID state=${state:-unknown} report=$status poll=$i/$MAX_POLLS"
  if [[ "$status" == "pass" ]]; then
    log "source build passed; no vLLM 0.13.0 fallback needed"
    exit 0
  fi
  case "$state" in
    ""|PENDING|CONFIGURING|RUNNING|COMPLETING|SUSPENDED)
      sleep "$POLL_SECONDS"
      ;;
    TIMEOUT|TIMEOUT*|CANCELLED|CANCELLED*)
      log "source build ended as $state; timeout watchdog owns retry path"
      exit 0
      ;;
    *)
      write_plan "$state"
      if [[ -e "$FALLBACK_JOB_FILE" ]] && grep -q '^vllm_native_source_build_job=[0-9]' "$FALLBACK_JOB_FILE"; then
        log "0.13.0 fallback job already recorded in $FALLBACK_JOB_FILE; not submitting another"
        exit 0
      fi
      if [[ "$AUTO_SUBMIT_013_ON_FAIL" == "true" || "$AUTO_SUBMIT_013_ON_FAIL" == "True" || "$AUTO_SUBMIT_013_ON_FAIL" == "1" ]]; then
        submit_013
      else
        log "AUTO_SUBMIT_013_ON_FAIL=false; wrote fallback plan to $FALLBACK_REPORT_MD"
      fi
      exit 0
      ;;
  esac
done

log "fallback watcher reached polling budget without terminal source-build state"
