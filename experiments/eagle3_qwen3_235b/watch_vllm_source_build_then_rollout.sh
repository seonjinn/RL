#!/usr/bin/env bash
set -euo pipefail

# Poll the vLLM source-build job. If it produces a PASS report, run the native
# ABI probe on the source-built site and, only if that probe passes, submit the
# Qwen3-235B SWE-Gym one-step rollout capture smoke.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/experiments/eagle3_qwen3_235b"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3}"
REPORT_DIR="${REPORT_DIR:-$ARTIFACT_ROOT/reports}"
SOURCE_JOB_FILE="${SOURCE_JOB_FILE:-$ROOT_DIR/latest_vllm_native_source_build_job.txt}"
SOURCE_JOB_ID="${SOURCE_JOB_ID:-}"
SOURCE_BUILD_JSON="${SOURCE_BUILD_JSON:-$REPORT_DIR/vllm_native_source_build.json}"
SOURCE_BUILD_MD="${SOURCE_BUILD_MD:-$REPORT_DIR/vllm_native_source_build.md}"
SOURCE_VLLM_SITE="${SOURCE_VLLM_SITE:-$ARTIFACT_ROOT/python_site/vllm_0_10_2_cu129_torch28nv_source_py312}"
ABI_PROBE_JSON="${ABI_PROBE_JSON:-$REPORT_DIR/vllm_native_abi_probe.json}"
ABI_PROBE_MD="${ABI_PROBE_MD:-$REPORT_DIR/vllm_native_abi_probe.md}"
ABI_PROBE_JOB_FILE="${ABI_PROBE_JOB_FILE:-$ROOT_DIR/latest_vllm_native_abi_probe_job.txt}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-coreai_dlalgo_nemorl}"
SBATCH_PARTITION="${SBATCH_PARTITION:-batch}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_POLLS="${MAX_POLLS:-600}"
SUBMIT_ABI_PROBE="${SUBMIT_ABI_PROBE:-true}"
SUBMIT_ROLLOUT="${SUBMIT_ROLLOUT:-true}"

mkdir -p "$REPORT_DIR"

is_true() {
  case "${1:-}" in
    true|True|TRUE|1|yes|Yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

kv_value() {
  awk -F= -v key="$2" '$1 == key {print $2; exit}' "$1" 2>/dev/null || true
}

if [[ -z "$SOURCE_JOB_ID" ]]; then
  SOURCE_JOB_ID="$(kv_value "$SOURCE_JOB_FILE" vllm_native_source_build_job)"
fi
if [[ -z "$SOURCE_JOB_ID" ]]; then
  echo "Could not determine SOURCE_JOB_ID from $SOURCE_JOB_FILE" >&2
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

wait_terminal() {
  local job_id="$1"
  local label="$2"
  local i state
  for i in $(seq 1 "$MAX_POLLS"); do
    state="$(job_state "$job_id")"
    echo "[$(date)] $label job=$job_id state=${state:-unknown} poll=$i/$MAX_POLLS"
    case "$state" in
      ""|PENDING|CONFIGURING|RUNNING|COMPLETING|SUSPENDED)
        sleep "$POLL_SECONDS"
        ;;
      *)
        return 0
        ;;
    esac
  done
  echo "$label job did not reach terminal state within polling budget: $job_id" >&2
  return 1
}

wait_terminal "$SOURCE_JOB_ID" "source-build"

source_status="$(json_status "$SOURCE_BUILD_JSON")"
echo "[$(date)] source build report status=$source_status json=$SOURCE_BUILD_JSON"
if [[ "$source_status" != "pass" ]]; then
  echo "Source build did not pass. Inspect $SOURCE_BUILD_MD and Slurm logs before retrying." >&2
  exit 1
fi

if is_true "$SUBMIT_ABI_PROBE"; then
  echo "[$(date)] submitting ABI probe for source-built site: $SOURCE_VLLM_SITE"
  ARTIFACT_ROOT="$ARTIFACT_ROOT" \
  SBATCH_ACCOUNT="$SBATCH_ACCOUNT" \
  SBATCH_PARTITION="$SBATCH_PARTITION" \
  VLLM_SITE_CANDIDATES="$SOURCE_VLLM_SITE" \
  JSON_OUT="$ABI_PROBE_JSON" \
  MARKDOWN_OUT="$ABI_PROBE_MD" \
  JOB_FILE="$ABI_PROBE_JOB_FILE" \
  SUBMIT=true \
    bash "$SCRIPT_DIR/submit_vllm_native_abi_probe.sh"
  probe_job="$(kv_value "$ABI_PROBE_JOB_FILE" vllm_native_abi_probe_job)"
  if [[ -z "$probe_job" ]]; then
    echo "ABI probe job id was not recorded" >&2
    exit 1
  fi
  wait_terminal "$probe_job" "abi-probe"
fi

abi_status="$(json_status "$ABI_PROBE_JSON")"
echo "[$(date)] ABI probe status=$abi_status json=$ABI_PROBE_JSON"
if [[ "$abi_status" != "pass" ]]; then
  echo "ABI probe did not pass; rollout smoke will not be submitted." >&2
  exit 1
fi

if is_true "$SUBMIT_ROLLOUT"; then
  echo "[$(date)] submitting source-built vLLM rollout smoke"
  ARTIFACT_ROOT="$ARTIFACT_ROOT" \
  SOURCE_VLLM_SITE="$SOURCE_VLLM_SITE" \
  SOURCE_BUILD_JSON="$SOURCE_BUILD_JSON" \
  DRY_RUN=false \
    bash "$SCRIPT_DIR/submit_source_vllm_rollout_smoke.sh"
else
  echo "[$(date)] SUBMIT_ROLLOUT=false; stopping after successful ABI probe."
fi
