#!/usr/bin/env bash
set -euo pipefail

# Submit or print an apples-to-apples baseline/static-Eagle3 smoke pair.
#
# Default is safe dry-run:
#   bash experiments/eagle3_qwen3_235b/submit_static_specdec_smoke_pair.sh
#
# Actual submission:
#   SUBMIT=true bash experiments/eagle3_qwen3_235b/submit_static_specdec_smoke_pair.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/experiments/eagle3_qwen3_235b"
JOB_FILE="${JOB_FILE:-$ROOT_DIR/latest_static_specdec_smoke_jobs.txt}"

SUBMIT="${SUBMIT:-false}"
RUN_BASELINE="${RUN_BASELINE:-true}"
RUN_SPECDEC="${RUN_SPECDEC:-true}"
SPECDEC_DEPENDS_ON_BASELINE="${SPECDEC_DEPENDS_ON_BASELINE:-true}"

MAX_NUM_STEPS="${MAX_NUM_STEPS:-1}"
BASELINE_WANDB_NAME="${BASELINE_WANDB_NAME:-qwen3-235b-swe-baseline-smoke}"
SPECDEC_WANDB_NAME="${SPECDEC_WANDB_NAME:-qwen3-235b-swe-eagle3-public-smoke}"
EAGLE3_DRAFT_MODEL="${EAGLE3_DRAFT_MODEL:-nvidia/Qwen3-235B-A22B-Eagle3}"
EAGLE3_NUM_SPEC_TOKENS="${EAGLE3_NUM_SPEC_TOKENS:-3}"
EAGLE3_DRAFT_TP="${EAGLE3_DRAFT_TP:-1}"
SMOKE_DEPENDENCY="${SMOKE_DEPENDENCY:-}"
BASELINE_SBATCH_DEPENDENCY="${BASELINE_SBATCH_DEPENDENCY:-${SMOKE_DEPENDENCY:-singleton}}"
SPECDEC_SBATCH_DEPENDENCY="${SPECDEC_SBATCH_DEPENDENCY:-${SMOKE_DEPENDENCY:-singleton}}"

run_or_print() {
  local label="$1"
  shift
  if [[ "$SUBMIT" == "true" || "$SUBMIT" == "True" ]]; then
    local tmp
    tmp="$(mktemp)"
    "$@" 2>&1 | tee "$tmp"
    local job_id
    job_id="$(awk '/Job ID:/ {print $3}' "$tmp" | tail -1)"
    rm -f "$tmp"
    if [[ -z "$job_id" ]]; then
      echo "Could not parse job id for $label" >&2
      exit 1
    fi
    echo "$job_id"
  else
    printf '# %s\n' "$label" >&2
    printf '%q ' "$@" >&2
    printf '\n' >&2
  fi
}

: > "$JOB_FILE"

baseline_job=""
if [[ "$RUN_BASELINE" == "true" || "$RUN_BASELINE" == "True" ]]; then
  baseline_cmd=(
    env
    MAX_NUM_STEPS="$MAX_NUM_STEPS"
    WANDB_NAME="$BASELINE_WANDB_NAME"
    EXP_SUFFIX_OVERRIDE="$BASELINE_WANDB_NAME"
    SBATCH_DEPENDENCY="$BASELINE_SBATCH_DEPENDENCY"
    DRY_RUN="$([[ "$SUBMIT" == "true" || "$SUBMIT" == "True" ]] && echo false || echo true)"
    bash "$SCRIPT_DIR/run_baseline_smoke.sh"
  )
  if [[ "$SUBMIT" == "true" || "$SUBMIT" == "True" ]]; then
    baseline_job="$(run_or_print "baseline smoke" "${baseline_cmd[@]}" | tail -1)"
    echo "baseline_job=$baseline_job" | tee -a "$JOB_FILE"
  else
    run_or_print "baseline smoke" "${baseline_cmd[@]}"
    baseline_job="BASELINE_JOB_ID"
  fi
fi

if [[ "$RUN_SPECDEC" == "true" || "$RUN_SPECDEC" == "True" ]]; then
  dependency=""
  if [[ "$SPECDEC_DEPENDS_ON_BASELINE" == "true" || "$SPECDEC_DEPENDS_ON_BASELINE" == "True" ]]; then
    [[ -n "$baseline_job" ]] && dependency="afterok:$baseline_job"
  elif [[ -n "$SPECDEC_SBATCH_DEPENDENCY" ]]; then
    dependency="$SPECDEC_SBATCH_DEPENDENCY"
  fi
  specdec_cmd=(
    env
    MAX_NUM_STEPS="$MAX_NUM_STEPS"
    WANDB_NAME="$SPECDEC_WANDB_NAME"
    EXP_SUFFIX_OVERRIDE="$SPECDEC_WANDB_NAME"
    EAGLE3_DRAFT_MODEL="$EAGLE3_DRAFT_MODEL"
    EAGLE3_NUM_SPEC_TOKENS="$EAGLE3_NUM_SPEC_TOKENS"
    EAGLE3_DRAFT_TP="$EAGLE3_DRAFT_TP"
    SBATCH_DEPENDENCY="${dependency:-singleton}"
    DRY_RUN="$([[ "$SUBMIT" == "true" || "$SUBMIT" == "True" ]] && echo false || echo true)"
    bash "$SCRIPT_DIR/run_static_specdec_smoke.sh"
  )
  if [[ "$SUBMIT" == "true" || "$SUBMIT" == "True" ]]; then
    specdec_job="$(run_or_print "static Eagle3 smoke" "${specdec_cmd[@]}" | tail -1)"
    echo "specdec_job=$specdec_job" | tee -a "$JOB_FILE"
  else
    run_or_print "static Eagle3 smoke" "${specdec_cmd[@]}"
  fi
fi

cat <<EOF
# smoke pair
SUBMIT=$SUBMIT
MAX_NUM_STEPS=$MAX_NUM_STEPS
BASELINE_WANDB_NAME=$BASELINE_WANDB_NAME
SPECDEC_WANDB_NAME=$SPECDEC_WANDB_NAME
EAGLE3_DRAFT_MODEL=$EAGLE3_DRAFT_MODEL
EAGLE3_NUM_SPEC_TOKENS=$EAGLE3_NUM_SPEC_TOKENS
EAGLE3_DRAFT_TP=$EAGLE3_DRAFT_TP
SPECDEC_DEPENDS_ON_BASELINE=$SPECDEC_DEPENDS_ON_BASELINE
BASELINE_SBATCH_DEPENDENCY=$BASELINE_SBATCH_DEPENDENCY
SPECDEC_SBATCH_DEPENDENCY=$SPECDEC_SBATCH_DEPENDENCY

# After both jobs finish, compare logs with:
python3 experiments/eagle3_qwen3_235b/analyze_static_specdec_smoke_pair.py \\
  --repo-root "${SWE_REPO_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL}" \\
  --fail-on-missing-spec-metrics
EOF

if [[ "$SUBMIT" != "true" && "$SUBMIT" != "True" ]]; then
  echo "# dry run only. Set SUBMIT=true to submit smoke jobs." >&2
fi
