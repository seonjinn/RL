#!/usr/bin/env bash
set -euo pipefail

# Plan or submit a short Qwen3-235B math GRPO run that captures
# train_data_step*.jsonl for Eagle3 draft training. Defaults to DRY_RUN=true.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3}"
ROLLOUT_LOG_DIR="${ROLLOUT_LOG_DIR:-$ARTIFACT_ROOT/rl_rollout_capture_logs/qwen3_235b_math_capture_smoke}"
OUTPUT_CONVERSATIONS="${OUTPUT_CONVERSATIONS:-$ARTIFACT_ROOT/data/qwen3_235b_math_rollout_conversations.jsonl}"
MATH_REPO_ROOT="${MATH_REPO_ROOT:-${REPO_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL}}"
CONFIG_FILE="${CONFIG_FILE:-$MATH_REPO_ROOT/examples/configs/recipes/llm/performance/grpo-qwen3-235b-32n8g-async-1off.yaml}"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/env.sh}"
RESOURCE_PROFILE_ENV="${RESOURCE_PROFILE_ENV:-$ARTIFACT_ROOT/reports/eagle3_resource_profile.env}"

if [[ -f "$RESOURCE_PROFILE_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$RESOURCE_PROFILE_ENV"
fi

ROLLOUT_GPUS_PER_NODE="${ROLLOUT_GPUS_PER_NODE:-${TRAIN_GPUS_PER_NODE:-${DUMP_GPUS_PER_NODE:-}}}"
if [[ -n "$ROLLOUT_GPUS_PER_NODE" && -z "${NUM_GPU:-}" && -z "${GPUS_PER_NODE:-}" ]]; then
  export NUM_GPU="$ROLLOUT_GPUS_PER_NODE"
fi

if [[ -n "${NUM_GPU:-}" && "$NUM_GPU" =~ ^[0-9]+$ && "$NUM_GPU" -gt 0 ]]; then
  DEFAULT_ROLLOUT_TOTAL_GPUS="${ROLLOUT_TOTAL_GPUS:-128}"
  DEFAULT_ROLLOUT_GENERATION_GPUS="${ROLLOUT_TOTAL_GENERATION_GPUS:-64}"
  if [[ -z "${NUM_NODES:-}" ]]; then
    export NUM_NODES=$(((DEFAULT_ROLLOUT_TOTAL_GPUS + NUM_GPU - 1) / NUM_GPU))
  fi
  if [[ -z "${NUM_GEN_NODES:-}" ]]; then
    export NUM_GEN_NODES=$(((DEFAULT_ROLLOUT_GENERATION_GPUS + NUM_GPU - 1) / NUM_GPU))
  fi
fi

MAX_NUM_STEPS="${MAX_NUM_STEPS:-1}"
PPS="${PPS:-4}"
GPP="${GPP:-8}"
GBS="${GBS:-32}"
SEQLEN="${SEQLEN:-8192}"
WANDB_NAME="${WANDB_NAME:-qwen3-235b-math-rollout-capture-smoke}"
EXP_SUFFIX_OVERRIDE="${EXP_SUFFIX_OVERRIDE:-$WANDB_NAME}"
CHECKPOINT_SUBDIR="${CHECKPOINT_SUBDIR:-$WANDB_NAME}"
DRY_RUN="${DRY_RUN:-true}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-235B-A22B-Thinking-2507}"

mkdir -p "$ROLLOUT_LOG_DIR" "$(dirname "$OUTPUT_CONVERSATIONS")"

export ARTIFACT_ROOT
export ROLLOUT_LOG_DIR
export OUTPUT_CONVERSATIONS
export REPO_ROOT="$MATH_REPO_ROOT"
export MATH_REPO_ROOT
export CONFIG_FILE
export ENV_FILE
export MAX_NUM_STEPS
export PPS
export GPP
export GBS
export SEQLEN
export MODEL_PATH
export TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
export WANDB_NAME
export EXP_SUFFIX_OVERRIDE
export CHECKPOINT_SUBDIR
export DRY_RUN
export SAVE_PERIOD=1000000
export VAL_PERIOD=1000000
export KEEP_TOP_K=1
export NUM_VAL_SAMPLES_TO_PRINT=0
export SBATCH_DEPENDENCY="${SBATCH_DEPENDENCY:-singleton}"
export EAGLE3_TARGET_CONTEXT=math
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-True}"
export VLLM_COMPILATION_LEVEL="${VLLM_COMPILATION_LEVEL:-0}"
export VLLM_USE_INDUCTOR="${VLLM_USE_INDUCTOR:-False}"

cat <<EOF
Math rollout capture smoke
  DRY_RUN=$DRY_RUN
  MAX_NUM_STEPS=$MAX_NUM_STEPS
  PPS=$PPS
  GPP=$GPP
  GBS=$GBS
  SEQLEN=$SEQLEN
  ROLLOUT_LOG_DIR=$ROLLOUT_LOG_DIR
  EXPECTED_JSONL=$ROLLOUT_LOG_DIR/exp_*/train_data_step1.jsonl
  OUTPUT_CONVERSATIONS=$OUTPUT_CONVERSATIONS
  MATH_REPO_ROOT=$MATH_REPO_ROOT
  CONFIG_FILE=$CONFIG_FILE
  RESOURCE_PROFILE_ENV=$RESOURCE_PROFILE_ENV
  MODEL_PATH=$MODEL_PATH
  NUM_GPU=${NUM_GPU:-<launcher-default>}
  NUM_NODES=${NUM_NODES:-<launcher-default>}
  NUM_GEN_NODES=${NUM_GEN_NODES:-<launcher-default>}

After a submitted run completes, normalize with:
  ARTIFACT_ROOT="$ARTIFACT_ROOT" ROLLOUT_LOG_DIR="$ROLLOUT_LOG_DIR" OUTPUT_DATA="$OUTPUT_CONVERSATIONS" bash experiments/eagle3_qwen3_235b/materialize_rollout_capture_corpus.sh

Track job/corpus state with:
  EAGLE3_TARGET_CONTEXT=math python3 experiments/eagle3_qwen3_235b/analyze_rollout_capture_job.py --artifact-root "$ARTIFACT_ROOT" --repo-root "$MATH_REPO_ROOT" --rollout-log-dir "$ROLLOUT_LOG_DIR" --output-data "$OUTPUT_CONVERSATIONS"
EOF

if [[ "$DRY_RUN" == "true" || "$DRY_RUN" == "True" ]]; then
  echo "[DRY-RUN] launcher:"
  printf '%q ' bash "$ROOT_DIR/run_grpo_qwen3_235b_math.sh"
  printf '\n'
  exit 0
fi

exec bash "$ROOT_DIR/run_grpo_qwen3_235b_math.sh"
