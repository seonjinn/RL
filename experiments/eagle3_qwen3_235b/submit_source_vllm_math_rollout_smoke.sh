#!/usr/bin/env bash
set -euo pipefail

# Submit or print the next Qwen3-235B math rollout smoke after the source-built
# vLLM target passes native import. The watcher materializes math train_data
# into ModelOpt conversation JSONL and can run the no-submit Eagle3 preflight.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/experiments/eagle3_qwen3_235b"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3}"
MATH_REPO_ROOT="${MATH_REPO_ROOT:-${REPO_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL}}"
REPORT_DIR="${REPORT_DIR:-$ARTIFACT_ROOT/reports}"
SOURCE_VLLM_SITE="${SOURCE_VLLM_SITE:-$ARTIFACT_ROOT/python_site/vllm_0_10_2_cu129_torch28nv_source_py312}"
SOURCE_BUILD_JSON="${SOURCE_BUILD_JSON:-$REPORT_DIR/vllm_native_source_build.json}"
VLLM_PIP_SPEC="${VLLM_PIP_SPEC:-https://files.pythonhosted.org/packages/7d/0a/278d7bbf454f7de5322a5007427eed3e8b34ed6c2802491b56bbdfd7bbb4/vllm-0.10.2.tar.gz}"
WANDB_NAME="${WANDB_NAME:-qwen3-235b-math-rollout-vllm0102src-smoke1step}"
ROLLOUT_LOG_DIR="${ROLLOUT_LOG_DIR:-$ARTIFACT_ROOT/rl_rollout_capture_logs/qwen3_235b_math_capture_vllm0102src_smoke1step}"
OUTPUT_CONVERSATIONS="${OUTPUT_CONVERSATIONS:-$ARTIFACT_ROOT/data/qwen3_235b_math_rollout_conversations_vllm0102src_smoke.jsonl}"
DRY_RUN="${DRY_RUN:-true}"
START_WATCHER="${START_WATCHER:-true}"
REQUIRE_SOURCE_BUILD_PASS="${REQUIRE_SOURCE_BUILD_PASS:-true}"
PROMOTE_TO_CANONICAL="${PROMOTE_TO_CANONICAL:-true}"
RUN_PIPELINE_PREFLIGHT="${RUN_PIPELINE_PREFLIGHT:-true}"
AUTO_SUBMIT_PIPELINE="${AUTO_SUBMIT_PIPELINE:-false}"
RUN_FULL_ROLLOUT_GATE="${RUN_FULL_ROLLOUT_GATE:-false}"
ROLLOUT_REPORT_PREFIX_TAG="${ROLLOUT_REPORT_PREFIX_TAG:-math_vllm0102src}"

mkdir -p "$REPORT_DIR" "$ROLLOUT_LOG_DIR" "$(dirname "$OUTPUT_CONVERSATIONS")"

is_true() {
  case "${1:-}" in
    true|True|TRUE|1|yes|Yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

source_build_passed() {
  python3 - "$SOURCE_BUILD_JSON" "$SOURCE_VLLM_SITE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_site = sys.argv[2]
if not path.exists():
    raise SystemExit(1)
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("overall_status") != "pass":
    raise SystemExit(1)
site = payload.get("output_site")
if site and site != expected_site:
    raise SystemExit(1)
PY
}

if is_true "$REQUIRE_SOURCE_BUILD_PASS"; then
  if ! source_build_passed; then
    echo "Source-built vLLM site is not proven PASS yet: $SOURCE_BUILD_JSON" >&2
    echo "Set REQUIRE_SOURCE_BUILD_PASS=false only for dry-run/debug." >&2
    exit 1
  fi
fi

submit_env=(
  ARTIFACT_ROOT="$ARTIFACT_ROOT"
  MATH_REPO_ROOT="$MATH_REPO_ROOT"
  REPO_ROOT="$MATH_REPO_ROOT"
  SHARED_VLLM_SITE="$SOURCE_VLLM_SITE"
  VLLM_PIP_SPEC="$VLLM_PIP_SPEC"
  WANDB_NAME="$WANDB_NAME"
  EXP_SUFFIX_OVERRIDE="$WANDB_NAME"
  CHECKPOINT_SUBDIR="$WANDB_NAME"
  ROLLOUT_LOG_DIR="$ROLLOUT_LOG_DIR"
  OUTPUT_CONVERSATIONS="$OUTPUT_CONVERSATIONS"
  MAX_NUM_STEPS="${MAX_NUM_STEPS:-1}"
  PPS="${PPS:-4}"
  GPP="${GPP:-8}"
  GBS="${GBS:-32}"
  SEQLEN="${SEQLEN:-8192}"
  DRY_RUN="$DRY_RUN"
  VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-True}"
  VLLM_COMPILATION_LEVEL="${VLLM_COMPILATION_LEVEL:-0}"
  VLLM_USE_INDUCTOR="${VLLM_USE_INDUCTOR:-False}"
  EAGLE3_TARGET_CONTEXT=math
)

echo "# source-built vLLM math rollout smoke"
echo "SOURCE_VLLM_SITE=$SOURCE_VLLM_SITE"
echo "SOURCE_BUILD_JSON=$SOURCE_BUILD_JSON"
echo "ROLLOUT_LOG_DIR=$ROLLOUT_LOG_DIR"
echo "OUTPUT_CONVERSATIONS=$OUTPUT_CONVERSATIONS"
echo "PROMOTE_TO_CANONICAL=$PROMOTE_TO_CANONICAL"
echo "RUN_PIPELINE_PREFLIGHT=$RUN_PIPELINE_PREFLIGHT"
echo "AUTO_SUBMIT_PIPELINE=$AUTO_SUBMIT_PIPELINE"
echo "RUN_FULL_ROLLOUT_GATE=$RUN_FULL_ROLLOUT_GATE"
echo "DRY_RUN=$DRY_RUN"

if is_true "$DRY_RUN"; then
  printf "%q " env "${submit_env[@]}" bash "$SCRIPT_DIR/run_math_rollout_capture_smoke.sh"
  printf "\n"
  exit 0
fi

env "${submit_env[@]}" bash "$SCRIPT_DIR/run_math_rollout_capture_smoke.sh"

job_file="$MATH_REPO_ROOT/latest_235b_math_job_id.txt"
if [[ ! -s "$job_file" ]]; then
  echo "Expected job id file was not written: $job_file" >&2
  exit 1
fi
JOB_ID="$(cat "$job_file")"
REPORT_PREFIX="rollout_capture_${ROLLOUT_REPORT_PREFIX_TAG}_${JOB_ID}"

echo "Submitted source-built vLLM math rollout smoke job: $JOB_ID"
echo "report_prefix=$REPORT_PREFIX"

if is_true "$START_WATCHER"; then
  watcher_log="$REPORT_DIR/watch_rollout_capture_${JOB_ID}_${ROLLOUT_REPORT_PREFIX_TAG}_smoke.log"
  pid_file="$REPORT_DIR/${REPORT_PREFIX}_watch.pid"
  nohup env \
    ARTIFACT_ROOT="$ARTIFACT_ROOT" \
    SWE_REPO_ROOT="$MATH_REPO_ROOT" \
    REPO_ROOT="$MATH_REPO_ROOT" \
    JOB_ID="$JOB_ID" \
    ROLLOUT_LOG_DIR="$ROLLOUT_LOG_DIR" \
    OUTPUT_CONVERSATIONS="$OUTPUT_CONVERSATIONS" \
    REPORT_PREFIX="$REPORT_PREFIX" \
    TARGET_CONTEXT=math \
    EAGLE3_TARGET_CONTEXT=math \
    PROMOTE_TO_CANONICAL="$PROMOTE_TO_CANONICAL" \
    RUN_PIPELINE_PREFLIGHT="$RUN_PIPELINE_PREFLIGHT" \
    AUTO_SUBMIT_PIPELINE="$AUTO_SUBMIT_PIPELINE" \
    RUN_FULL_ROLLOUT_GATE="$RUN_FULL_ROLLOUT_GATE" \
    bash "$SCRIPT_DIR/watch_rollout_capture_materialize.sh" >"$watcher_log" 2>&1 &
  echo "$!" > "$pid_file"
  echo "Watcher PID $(cat "$pid_file") -> $watcher_log"
fi
