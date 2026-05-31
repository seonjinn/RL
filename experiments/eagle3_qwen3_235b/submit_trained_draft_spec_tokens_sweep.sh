#!/usr/bin/env bash
set -euo pipefail

# Submit or print a baseline plus trained-draft Eagle3 num_speculative_tokens sweep.
#
# Dry-run:
#   VLLM_DRAFT_DIR=/path/to/qwen3_235b_eagle3_vllm \
#     bash experiments/eagle3_qwen3_235b/submit_trained_draft_spec_tokens_sweep.sh
#
# Submit:
#   SUBMIT=true VLLM_DRAFT_DIR=/path/to/qwen3_235b_eagle3_vllm \
#     bash experiments/eagle3_qwen3_235b/submit_trained_draft_spec_tokens_sweep.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/experiments/eagle3_qwen3_235b"

VLLM_DRAFT_DIR="${VLLM_DRAFT_DIR:?set VLLM_DRAFT_DIR to the converted vLLM Eagle3 draft directory}"
SUBMIT="${SUBMIT:-false}"
JOB_FILE="${JOB_FILE:-$ROOT_DIR/latest_trained_draft_spec_tokens_sweep_jobs.txt}"

MAX_NUM_STEPS="${MAX_NUM_STEPS:-2}"
SPEC_TOKENS_LIST="${SPEC_TOKENS_LIST:-2 3 4}"
EAGLE3_DRAFT_TP="${EAGLE3_DRAFT_TP:-1}"
BASELINE_WANDB_NAME="${BASELINE_WANDB_NAME:-qwen3-235b-swe-baseline-spec-tokens-sweep}"
SPECDEC_WANDB_PREFIX="${SPECDEC_WANDB_PREFIX:-qwen3-235b-swe-eagle3-trained-spec}"
SPECDEC_DEPENDS_ON_BASELINE="${SPECDEC_DEPENDS_ON_BASELINE:-true}"
ALLOW_MISSING_DRAFT_FOR_DEPENDENCY="${ALLOW_MISSING_DRAFT_FOR_DEPENDENCY:-false}"
SMOKE_DEPENDENCY="${SMOKE_DEPENDENCY:-}"
BASELINE_SBATCH_DEPENDENCY="${BASELINE_SBATCH_DEPENDENCY:-${SMOKE_DEPENDENCY:-singleton}}"
SPECDEC_SBATCH_DEPENDENCY="${SPECDEC_SBATCH_DEPENDENCY:-${SMOKE_DEPENDENCY:-singleton}}"

DEFAULT_SPECDEC_RL_DIR="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL"
LEGACY_SWE_REPO_ROOT="/lustre/fsw/portfolios/coreai/users/sna/repos/nemo-rl-qwen-swe"
if [[ -d "$LEGACY_SWE_REPO_ROOT" ]]; then
  DEFAULT_REPO_ROOT="$LEGACY_SWE_REPO_ROOT"
elif [[ -d "$DEFAULT_SPECDEC_RL_DIR" ]]; then
  DEFAULT_REPO_ROOT="$DEFAULT_SPECDEC_RL_DIR"
else
  DEFAULT_REPO_ROOT="$ROOT_DIR"
fi

RECORDED_ARTIFACT_ROOT="${ARTIFACT_ROOT:-}"
if [[ -z "$RECORDED_ARTIFACT_ROOT" && "$(basename "$VLLM_DRAFT_DIR")" == "vllm_draft" ]]; then
  RECORDED_ARTIFACT_ROOT="$(cd "$(dirname "$VLLM_DRAFT_DIR")" 2>/dev/null && pwd -P || dirname "$VLLM_DRAFT_DIR")"
fi
RECORDED_REPO_ROOT="${REPO_ROOT:-$DEFAULT_REPO_ROOT}"
RECORDED_SWE_REPO_ROOT="${SWE_REPO_ROOT:-$RECORDED_REPO_ROOT}"
RECORDED_CONFIG_FILE="${CONFIG_FILE:-$ROOT_DIR/grpo_qwen3_235b_swe.yaml}"
RECORDED_ENV_FILE="${ENV_FILE:-$ROOT_DIR/env.sh}"
if [[ ! -f "$RECORDED_ENV_FILE" && -f "$RECORDED_REPO_ROOT/env.sh" ]]; then
  RECORDED_ENV_FILE="$RECORDED_REPO_ROOT/env.sh"
fi
RECORDED_CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
if [[ -z "$RECORDED_CHAT_TEMPLATE" && -n "$RECORDED_ARTIFACT_ROOT" ]]; then
  candidate_template="$RECORDED_ARTIFACT_ROOT/templates/qwen3_generation_template.jinja2"
  if [[ -f "$candidate_template" ]]; then
    RECORDED_CHAT_TEMPLATE="$candidate_template"
  fi
fi

draft_has_safetensors() {
  compgen -G "$VLLM_DRAFT_DIR/*.safetensors" >/dev/null
}

if [[ "$SUBMIT" == "true" || "$SUBMIT" == "True" ]]; then
  if [[ ! -f "$VLLM_DRAFT_DIR/config.json" ]] || ! draft_has_safetensors; then
    if [[ "$ALLOW_MISSING_DRAFT_FOR_DEPENDENCY" == "true" || "$ALLOW_MISSING_DRAFT_FOR_DEPENDENCY" == "True" ]]; then
      echo "WARN: VLLM_DRAFT_DIR is not complete yet; relying on Slurm dependency before smoke runs: $VLLM_DRAFT_DIR" >&2
    else
      echo "VLLM_DRAFT_DIR must contain config.json and at least one .safetensors weight before submit: $VLLM_DRAFT_DIR" >&2
      exit 1
    fi
  fi
fi

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
{
  echo "# trained draft spec-token sweep"
  echo "vllm_draft_dir=$VLLM_DRAFT_DIR"
  echo "artifact_root=$RECORDED_ARTIFACT_ROOT"
  echo "repo_root=$RECORDED_REPO_ROOT"
  echo "swe_repo_root=$RECORDED_SWE_REPO_ROOT"
  echo "config_file=$RECORDED_CONFIG_FILE"
  echo "env_file=$RECORDED_ENV_FILE"
  echo "chat_template=$RECORDED_CHAT_TEMPLATE"
  echo "max_num_steps=$MAX_NUM_STEPS"
  echo "spec_tokens_list=$SPEC_TOKENS_LIST"
  echo "eagle3_draft_tp=$EAGLE3_DRAFT_TP"
} >> "$JOB_FILE"

baseline_cmd=(
  env
  ARTIFACT_ROOT="$RECORDED_ARTIFACT_ROOT"
  REPO_ROOT="$RECORDED_REPO_ROOT"
  SWE_REPO_ROOT="$RECORDED_SWE_REPO_ROOT"
  CONFIG_FILE="$RECORDED_CONFIG_FILE"
  ENV_FILE="$RECORDED_ENV_FILE"
  CHAT_TEMPLATE="$RECORDED_CHAT_TEMPLATE"
  MAX_NUM_STEPS="$MAX_NUM_STEPS"
  WANDB_NAME="$BASELINE_WANDB_NAME"
  EXP_SUFFIX_OVERRIDE="$BASELINE_WANDB_NAME"
  SBATCH_DEPENDENCY="$BASELINE_SBATCH_DEPENDENCY"
  DRY_RUN="$([[ "$SUBMIT" == "true" || "$SUBMIT" == "True" ]] && echo false || echo true)"
  bash "$SCRIPT_DIR/run_baseline_smoke.sh"
)

baseline_job=""
if [[ "$SUBMIT" == "true" || "$SUBMIT" == "True" ]]; then
  baseline_job="$(run_or_print "baseline sweep smoke" "${baseline_cmd[@]}" | tail -1)"
  echo "baseline_job=$baseline_job" | tee -a "$JOB_FILE"
else
  run_or_print "baseline sweep smoke" "${baseline_cmd[@]}"
  baseline_job="BASELINE_JOB_ID"
  echo "baseline_job=$baseline_job" >> "$JOB_FILE"
fi

# shellcheck disable=SC2206
spec_tokens=($SPEC_TOKENS_LIST)
for tokens in "${spec_tokens[@]}"; do
  if ! [[ "$tokens" =~ ^[0-9]+$ ]]; then
    echo "SPEC_TOKENS_LIST contains a non-integer value: $tokens" >&2
    exit 1
  fi
  dependency=""
  if [[ "$SPECDEC_DEPENDS_ON_BASELINE" == "true" || "$SPECDEC_DEPENDS_ON_BASELINE" == "True" ]]; then
    [[ -n "$baseline_job" ]] && dependency="afterok:$baseline_job"
  elif [[ -n "$SPECDEC_SBATCH_DEPENDENCY" ]]; then
    dependency="$SPECDEC_SBATCH_DEPENDENCY"
  fi
  specdec_name="${SPECDEC_WANDB_PREFIX}-${tokens}tok"
  specdec_cmd=(
    env
    ARTIFACT_ROOT="$RECORDED_ARTIFACT_ROOT"
    REPO_ROOT="$RECORDED_REPO_ROOT"
    SWE_REPO_ROOT="$RECORDED_SWE_REPO_ROOT"
    CONFIG_FILE="$RECORDED_CONFIG_FILE"
    ENV_FILE="$RECORDED_ENV_FILE"
    CHAT_TEMPLATE="$RECORDED_CHAT_TEMPLATE"
    MAX_NUM_STEPS="$MAX_NUM_STEPS"
    WANDB_NAME="$specdec_name"
    EXP_SUFFIX_OVERRIDE="$specdec_name"
    EAGLE3_DRAFT_MODEL="$VLLM_DRAFT_DIR"
    EAGLE3_NUM_SPEC_TOKENS="$tokens"
    EAGLE3_DRAFT_TP="$EAGLE3_DRAFT_TP"
    SBATCH_DEPENDENCY="${dependency:-singleton}"
    DRY_RUN="$([[ "$SUBMIT" == "true" || "$SUBMIT" == "True" ]] && echo false || echo true)"
    bash "$SCRIPT_DIR/run_static_specdec_smoke.sh"
  )
  if [[ "$SUBMIT" == "true" || "$SUBMIT" == "True" ]]; then
    specdec_job="$(run_or_print "trained Eagle3 ${tokens}-token smoke" "${specdec_cmd[@]}" | tail -1)"
    echo "specdec_tokens_${tokens}_job=$specdec_job" | tee -a "$JOB_FILE"
  else
    run_or_print "trained Eagle3 ${tokens}-token smoke" "${specdec_cmd[@]}"
    echo "specdec_tokens_${tokens}_job=SPECDEC_${tokens}_JOB_ID" >> "$JOB_FILE"
  fi
done

cat <<EOF

# Trained draft spec-token sweep analysis:
python3 experiments/eagle3_qwen3_235b/analyze_spec_tokens_sweep.py \\
  --job-file $JOB_FILE \\
  --repo-root "${SWE_REPO_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL}" \\
  --markdown-out /path/to/trained_draft_spec_tokens_sweep.md \\
  --json-out /path/to/trained_draft_spec_tokens_sweep.json \\
  --fail-on-missing-spec-metrics
EOF

if [[ "$SUBMIT" != "true" && "$SUBMIT" != "True" ]]; then
  echo "# dry run only. Set SUBMIT=true to submit sweep jobs." >&2
fi
