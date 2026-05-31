#!/usr/bin/env bash
set -euo pipefail

# Poll the Megatron compatibility probe and, only after it reports PASS, print
# or submit the next Qwen3-235B rollout smoke. Safe defaults never submit heavy
# GPU work.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/experiments/eagle3_qwen3_235b"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  bash experiments/eagle3_qwen3_235b/followup_megatron_probe_to_rollout.sh

Safe defaults:
  SUBMIT_ROLLOUT=false
  ALLOW_HEAVY_GPU=false

To submit after the probe PASSes:
  SUBMIT_ROLLOUT=true ALLOW_HEAVY_GPU=true \
  bash experiments/eagle3_qwen3_235b/followup_megatron_probe_to_rollout.sh

Useful env:
  PROBE_JOB_ID=2867766
  JOB_FILE=latest_megatron_compat_probe_job.txt
  ARTIFACT_ROOT=/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3
  SWE_REPO_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL
  JSON_OUT=$ARTIFACT_ROOT/reports/megatron_compat_probe.json
  FAIL_ON_NOT_READY=true
EOF
  exit 0
fi

is_true() {
  case "${1:-}" in
    true|True|TRUE|1|yes|Yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

print_cmd() {
  printf "%q " "$@"
  printf "\n"
}

key_value() {
  local file="$1"
  local key="$2"
  [[ -f "$file" ]] || return 1
  awk -F= -v k="$key" '$1 == k {print substr($0, length(k) + 2); exit}' "$file"
}

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3}"
SWE_REPO_ROOT="${SWE_REPO_ROOT:-${REPO_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL}}"
REPORT_DIR="${REPORT_DIR:-$ARTIFACT_ROOT/reports}"
JOB_FILE="${JOB_FILE:-$ROOT_DIR/latest_megatron_compat_probe_job.txt}"
REPORT_JOB_FILE="${REPORT_JOB_FILE:-$REPORT_DIR/megatron_compat_probe_job.env}"
JSON_OUT="${JSON_OUT:-$REPORT_DIR/megatron_compat_probe.json}"
SUBMIT_ROLLOUT="${SUBMIT_ROLLOUT:-false}"
ALLOW_HEAVY_GPU="${ALLOW_HEAVY_GPU:-false}"
FAIL_ON_NOT_READY="${FAIL_ON_NOT_READY:-false}"

PROBE_JOB_ID="${PROBE_JOB_ID:-}"
if [[ -z "$PROBE_JOB_ID" ]]; then
  PROBE_JOB_ID="$(key_value "$JOB_FILE" megatron_compat_probe_job 2>/dev/null || true)"
fi
if [[ -z "$PROBE_JOB_ID" || "$PROBE_JOB_ID" == "MEGATRON_COMPAT_PROBE_JOB_ID" ]]; then
  PROBE_JOB_ID="$(key_value "$REPORT_JOB_FILE" megatron_compat_probe_job 2>/dev/null || true)"
fi

if [[ -z "$PROBE_JOB_ID" || "$PROBE_JOB_ID" == "MEGATRON_COMPAT_PROBE_JOB_ID" ]]; then
  echo "No concrete Megatron compatibility probe job id is recorded." >&2
  echo "Set PROBE_JOB_ID or update $JOB_FILE." >&2
  if is_true "$FAIL_ON_NOT_READY"; then
    exit 1
  fi
  exit 0
fi

echo "# Megatron compatibility probe follow-up"
echo "PROBE_JOB_ID=$PROBE_JOB_ID"
echo "JSON_OUT=$JSON_OUT"
echo "SUBMIT_ROLLOUT=$SUBMIT_ROLLOUT"
echo "ALLOW_HEAVY_GPU=$ALLOW_HEAVY_GPU"

if command -v squeue >/dev/null 2>&1; then
  squeue -j "$PROBE_JOB_ID" -o "%i|%T|%M|%D|%R|%S" --noheader || true
else
  echo "squeue not available on this host; skipping live queue check."
fi

if command -v sacct >/dev/null 2>&1; then
  sacct -j "$PROBE_JOB_ID" --format=JobIDRaw,State,ExitCode,Elapsed,Start,End -P -n 2>/dev/null | tail -40 || true
else
  echo "sacct not available on this host; skipping accounting check."
fi

probe_status="$(
  python3 - "$JSON_OUT" <<'PY'
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

api = payload.get("api") if isinstance(payload.get("api"), dict) else {}
grouped = api.get("tpaware_grouped_linear_detection")
temp = api.get("grouped_linear_temporary_weight_attr")
save_compat = api.get("community_import_save_compat")
errors = payload.get("errors") or []
checks = [
    payload.get("overall_status") == "pass",
    isinstance(grouped, dict),
    grouped.get("TEColumnParallelGroupedLinear") == "replicated" if isinstance(grouped, dict) else False,
    grouped.get("TERowParallelGroupedLinear") == "replicated" if isinstance(grouped, dict) else False,
    isinstance(temp, dict),
    temp.get("has_weight") is True if isinstance(temp, dict) else False,
    temp.get("weight_is_weight0") is True if isinstance(temp, dict) else False,
    isinstance(save_compat, dict),
    save_compat.get("helper_available") is True if isinstance(save_compat, dict) else False,
    save_compat.get("checkpoint_fallback_available") is True if isinstance(save_compat, dict) else False,
    (
        save_compat.get("model_load_save_available") is True
        or save_compat.get("checkpointing_save_available") is True
        if isinstance(save_compat, dict)
        else False
    ),
    not errors,
]
if all(checks):
    print("pass")
else:
    print("not_pass")
    print(f"overall_status={payload.get('overall_status')}", file=sys.stderr)
    print(f"tpaware_grouped_linear_detection={grouped}", file=sys.stderr)
    print(f"grouped_linear_temporary_weight_attr={temp}", file=sys.stderr)
    print(f"community_import_save_compat={save_compat}", file=sys.stderr)
    print(f"errors={len(errors)} {errors[:3]}", file=sys.stderr)
PY
)"

if [[ "$probe_status" != "pass" ]]; then
  echo "Megatron compatibility probe is not PASS yet: $probe_status"
  if is_true "$FAIL_ON_NOT_READY"; then
    exit 1
  fi
  exit 0
fi

echo "Megatron compatibility probe PASS. Preparing rollout smoke command."

WANDB_NAME="${WANDB_NAME:-qwen3-235b-swe-rollout-capture-balanced24n4g-headport-finalize-etpkwarg-providemodels-cudadev-automapgroupedweight-excl-t06-t01}"
ROLLOUT_LOG_DIR="${ROLLOUT_LOG_DIR:-$ARTIFACT_ROOT/rl_rollout_capture_logs/qwen3_235b_swe_capture_balanced24n4g_headport_finalize_etpkwarg_providemodels_cudadev_automapgroupedweight_excl_t06_t01}"
OUTPUT_CONVERSATIONS="${OUTPUT_CONVERSATIONS:-$ARTIFACT_ROOT/data/qwen3_235b_swe_rollout_conversations_balanced24n4g_headport_finalize_etpkwarg_providemodels_cudadev_automapgroupedweight_excl_t06_t01.jsonl}"
ROLLOUT_REPORT_PREFIX_TAG="${ROLLOUT_REPORT_PREFIX_TAG:-vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etpkwarg_providemodels_cudadev_automapgroupedweight_excl_t06_t01}"

rollout_cmd=(
  env
  "ARTIFACT_ROOT=$ARTIFACT_ROOT"
  "SWE_REPO_ROOT=$SWE_REPO_ROOT"
  "SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-coreai_dlalgo_nemorl}"
  "SBATCH_PARTITION=${SBATCH_PARTITION:-batch}"
  "DRY_RUN=false"
  "START_WATCHER=${START_WATCHER:-true}"
  "REQUIRE_SOURCE_BUILD_PASS=${REQUIRE_SOURCE_BUILD_PASS:-true}"
  "NUM_GPU=${NUM_GPU:-4}"
  "NUM_NODES=${NUM_NODES:-24}"
  "NUM_GEN_NODES=${NUM_GEN_NODES:-8}"
  "TP=${TP:-4}"
  "ETP=${ETP:-1}"
  "EP=${EP:-16}"
  "CP=${CP:-1}"
  "PP=${PP:-4}"
  "VLLM_TP=${VLLM_TP:-8}"
  "PP_FIRST_STAGE=${PP_FIRST_STAGE:-23}"
  "PP_LAST_STAGE=${PP_LAST_STAGE:-23}"
  "SBATCH_EXCLUDE=${SBATCH_EXCLUDE:-nvl72089-T06,nvl72007-T01}"
  "WANDB_NAME=$WANDB_NAME"
  "ROLLOUT_LOG_DIR=$ROLLOUT_LOG_DIR"
  "OUTPUT_CONVERSATIONS=$OUTPUT_CONVERSATIONS"
  "ROLLOUT_REPORT_PREFIX_TAG=$ROLLOUT_REPORT_PREFIX_TAG"
  bash
  "$SCRIPT_DIR/submit_source_vllm_rollout_smoke.sh"
)

if ! is_true "$SUBMIT_ROLLOUT"; then
  echo "# rollout command (not submitted; set SUBMIT_ROLLOUT=true ALLOW_HEAVY_GPU=true to run)"
  print_cmd "${rollout_cmd[@]}"
  exit 0
fi

if ! is_true "$ALLOW_HEAVY_GPU"; then
  echo "Refusing to submit rollout without ALLOW_HEAVY_GPU=true." >&2
  exit 1
fi

print_cmd "${rollout_cmd[@]}"
"${rollout_cmd[@]}"
