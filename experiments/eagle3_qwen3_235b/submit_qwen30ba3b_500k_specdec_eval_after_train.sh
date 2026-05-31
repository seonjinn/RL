#!/usr/bin/env bash
set -euo pipefail

# Submit Qwen3-30B-A3B NeMo-RL SpecDec K-sweep jobs after the 500K
# Speculators Eagle3 train job has produced a checkpoint.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/experiments/eagle3_qwen3_235b"

REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/Nemo-RL_Qwen3_Roadmap}"
NEMO_RL_DIR="${NEMO_RL_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl}"
TRAIN_JOB_ID="${TRAIN_JOB_ID:-3056818}"
ACCOUNT="${ACCOUNT:-coreai_dlalgo_nemorl}"
PARTITION="${PARTITION:-batch}"
WRAPPER_TIME="${WRAPPER_TIME:-00:30:00}"
WRAPPER_GRES_FLAG="${WRAPPER_GRES_FLAG:---gres=gpu:4}"
SUBMIT_WRAPPER="${SUBMIT_WRAPPER:-true}"
INSIDE_WRAPPER="${INSIDE_WRAPPER:-false}"

DRAFT_ROOT="${DRAFT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/speculators/eagle3_qwen3_30ba3b_mixed_math_nonopenmath_500k_parallel/checkpoints_train_500k_layers48_mlen8193}"
SPEC_TOKENS_LIST="${SPEC_TOKENS_LIST:-1 2 3}"
SPECDEC_SCHEDULER_PATCH_GATE_THRESHOLD="${SPECDEC_SCHEDULER_PATCH_GATE_THRESHOLD:-0}"
SPECDEC_SCHEDULER_PATCH_GATE_TOKEN_THRESHOLD="${SPECDEC_SCHEDULER_PATCH_GATE_TOKEN_THRESHOLD:-4096}"
ENABLE_RUNTIME_SPECDEC_GATE_PATCH="${ENABLE_RUNTIME_SPECDEC_GATE_PATCH:-true}"
VLLM_SPECDEC_BATCH_GATE_LOG_INTERVAL="${VLLM_SPECDEC_BATCH_GATE_LOG_INTERVAL:-256}"
MAX_STEPS="${MAX_STEPS:-20}"
NUM_NODES="${NUM_NODES:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
NUM_GENERATIONS="${NUM_GENERATIONS:-32}"
TRAIN_GLOBAL_BATCH_SIZE="${TRAIN_GLOBAL_BATCH_SIZE:-$((NUM_PROMPTS * NUM_GENERATIONS))}"
NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-true}"
NRL_VLLM_DISABLE_LOG_STATS="${NRL_VLLM_DISABLE_LOG_STATS:-false}"
NRL_VLLM_OMIT_GENERATION_LOGPROBS="${NRL_VLLM_OMIT_GENERATION_LOGPROBS:-false}"
OUT_FILE="${OUT_FILE:-$REMOTE_REPO_ROOT/latest_qwen30ba3b_500k_specdec_eval_jobs.txt}"
DRY_RUN="${DRY_RUN:-false}"

submit_wrapper() {
  local dependency=()
  if [[ -n "$TRAIN_JOB_ID" ]]; then
    dependency=(--dependency="afterok:${TRAIN_JOB_ID}")
  fi
  local remote_script="$REMOTE_REPO_ROOT/experiments/eagle3_qwen3_235b/$(basename "$0")"
  local wrapped
  wrapped=$(
    printf 'cd %q && INSIDE_WRAPPER=true REMOTE_REPO_ROOT=%q NEMO_RL_DIR=%q TRAIN_JOB_ID=%q DRAFT_ROOT=%q SPEC_TOKENS_LIST=%q SPECDEC_SCHEDULER_PATCH_GATE_THRESHOLD=%q SPECDEC_SCHEDULER_PATCH_GATE_TOKEN_THRESHOLD=%q ENABLE_RUNTIME_SPECDEC_GATE_PATCH=%q VLLM_SPECDEC_BATCH_GATE_LOG_INTERVAL=%q MAX_STEPS=%q NUM_NODES=%q GPUS_PER_NODE=%q NUM_PROMPTS=%q NUM_GENERATIONS=%q TRAIN_GLOBAL_BATCH_SIZE=%q NRL_FORCE_REBUILD_VENVS=%q NRL_VLLM_DISABLE_LOG_STATS=%q NRL_VLLM_OMIT_GENERATION_LOGPROBS=%q OUT_FILE=%q bash %q' \
      "$REMOTE_REPO_ROOT" "$REMOTE_REPO_ROOT" "$NEMO_RL_DIR" "$TRAIN_JOB_ID" "$DRAFT_ROOT" "$SPEC_TOKENS_LIST" \
      "$SPECDEC_SCHEDULER_PATCH_GATE_THRESHOLD" "$SPECDEC_SCHEDULER_PATCH_GATE_TOKEN_THRESHOLD" "$ENABLE_RUNTIME_SPECDEC_GATE_PATCH" \
      "$VLLM_SPECDEC_BATCH_GATE_LOG_INTERVAL" \
      "$MAX_STEPS" "$NUM_NODES" "$GPUS_PER_NODE" "$NUM_PROMPTS" "$NUM_GENERATIONS" "$TRAIN_GLOBAL_BATCH_SIZE" \
      "$NRL_FORCE_REBUILD_VENVS" "$NRL_VLLM_DISABLE_LOG_STATS" "$NRL_VLLM_OMIT_GENERATION_LOGPROBS" "$OUT_FILE" "$remote_script"
  )
  local cmd=(
    sbatch
    --nodes=1
    --ntasks=1
    --account="$ACCOUNT"
    --partition="$PARTITION"
    --time="$WRAPPER_TIME"
    "$WRAPPER_GRES_FLAG"
    --job-name="qwen30ba3b-submit-500k-specdec-eval"
    --output="logs/%x_%j.out"
    --error="logs/%x_%j.err"
    "${dependency[@]}"
    --wrap "$wrapped"
  )
  printf '%q ' "${cmd[@]}"
  printf '\n'
  if [[ "$DRY_RUN" != "true" && "$DRY_RUN" != "True" ]]; then
    "${cmd[@]}"
  fi
}

submit_evals() {
  if [[ ! -d "$NEMO_RL_DIR" ]]; then
    echo "ERROR: NEMO_RL_DIR not found: $NEMO_RL_DIR" >&2
    exit 2
  fi
  if [[ ! -f "$SCRIPT_DIR/Qwen30BA3B_GB200_Main_SpecDec.sh" ]]; then
    echo "ERROR: SpecDec submit script not found under $SCRIPT_DIR" >&2
    exit 2
  fi
  local draft_model
  draft_model="$(
    find "$DRAFT_ROOT" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' \
      -exec sh -c 'test -s "$1/config.json"' sh {} \; -print 2>/dev/null \
      | sort -V \
      | tail -n 1
  )"
  if [[ -z "$draft_model" ]]; then
    echo "ERROR: no completed 500K checkpoint with config.json under $DRAFT_ROOT" >&2
    exit 3
  fi

  mkdir -p "$(dirname "$OUT_FILE")"
  {
    echo "# Qwen3-30B-A3B 500K Speculators Eagle3 NeMo-RL eval sweep"
    echo "created_at=$(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "train_job=$TRAIN_JOB_ID"
    echo "draft_root=$DRAFT_ROOT"
    echo "draft_model=$draft_model"
    echo "spec_tokens_list=$SPEC_TOKENS_LIST"
    echo "gate_request_threshold=$SPECDEC_SCHEDULER_PATCH_GATE_THRESHOLD"
    echo "gate_token_threshold=$SPECDEC_SCHEDULER_PATCH_GATE_TOKEN_THRESHOLD"
    echo "gate_log_interval=$VLLM_SPECDEC_BATCH_GATE_LOG_INTERVAL"
    echo "max_steps=$MAX_STEPS"
    echo "num_prompts=$NUM_PROMPTS"
    echo "num_generations=$NUM_GENERATIONS"
    echo "train_global_batch_size=$TRAIN_GLOBAL_BATCH_SIZE"
  } > "$OUT_FILE"

  # shellcheck disable=SC2206
  local tokens_list=($SPEC_TOKENS_LIST)
  local token_count
  for token_count in "${tokens_list[@]}"; do
    if ! [[ "$token_count" =~ ^[0-9]+$ ]]; then
      echo "ERROR: SPEC_TOKENS_LIST contains non-integer value: $token_count" >&2
      exit 4
    fi
    local request_gate="${SPECDEC_SCHEDULER_PATCH_GATE_THRESHOLD:-0}"
    local token_gate="${SPECDEC_SCHEDULER_PATCH_GATE_TOKEN_THRESHOLD:-0}"
    local tag="main-specdec-mixed500k-k${token_count}-req${request_gate}-tok${token_gate}"
    local stamp
    stamp="$(date '+%Y%m%d%H%M%S')"
    local env_dir="${NEMO_RL_DIR}/.driver_venvs/qwen30ba3b_main_specdec_mixed500k_k${token_count}_${stamp}"
    local wandb_name="Qwen30B_A3B_Main_N${NUM_NODES}xG${GPUS_PER_NODE}_specdec_mixed500k_k${token_count}_p${NUM_PROMPTS}_g${NUM_GENERATIONS}_${MAX_STEPS}step_reqgate${request_gate}_tokgate${token_gate}"
    local tmp
    tmp="$(mktemp)"
    env \
      NEMO_RL_DIR="$NEMO_RL_DIR" \
      DRAFT_ROOT="$DRAFT_ROOT" \
      DRAFT_MODEL="$draft_model" \
      NUM_SPECULATIVE_TOKENS="$token_count" \
      SPECDEC_SCHEDULER_PATCH_GATE_THRESHOLD="$SPECDEC_SCHEDULER_PATCH_GATE_THRESHOLD" \
      SPECDEC_SCHEDULER_PATCH_GATE_TOKEN_THRESHOLD="$SPECDEC_SCHEDULER_PATCH_GATE_TOKEN_THRESHOLD" \
      ENABLE_RUNTIME_SPECDEC_GATE_PATCH="$ENABLE_RUNTIME_SPECDEC_GATE_PATCH" \
      VLLM_SPECDEC_BATCH_GATE_LOG_INTERVAL="$VLLM_SPECDEC_BATCH_GATE_LOG_INTERVAL" \
      MAX_STEPS="$MAX_STEPS" \
      NUM_NODES="$NUM_NODES" \
      GPUS_PER_NODE="$GPUS_PER_NODE" \
      NUM_PROMPTS="$NUM_PROMPTS" \
      NUM_GENERATIONS="$NUM_GENERATIONS" \
      TRAIN_GLOBAL_BATCH_SIZE="$TRAIN_GLOBAL_BATCH_SIZE" \
      NRL_FORCE_REBUILD_VENVS="$NRL_FORCE_REBUILD_VENVS" \
      NRL_VLLM_DISABLE_LOG_STATS="$NRL_VLLM_DISABLE_LOG_STATS" \
      NRL_VLLM_OMIT_GENERATION_LOGPROBS="$NRL_VLLM_OMIT_GENERATION_LOGPROBS" \
      DRIVER_UV_PROJECT_ENVIRONMENT="$env_dir" \
      JOB_TAG="$tag" \
      WANDB_NAME="$wandb_name" \
      bash "$SCRIPT_DIR/Qwen30BA3B_GB200_Main_SpecDec.sh" 2>&1 | tee "$tmp"
    local job_id
    job_id="$(awk '/Submitted batch job/{print $4}' "$tmp" | tail -1)"
    rm -f "$tmp"
    if [[ -z "$job_id" ]]; then
      echo "ERROR: could not parse Slurm job id for K=$token_count" >&2
      exit 5
    fi
    echo "specdec_k${token_count}_job=$job_id" | tee -a "$OUT_FILE"
  done
}

if [[ "$INSIDE_WRAPPER" == "true" || "$INSIDE_WRAPPER" == "True" || "$SUBMIT_WRAPPER" == "false" || "$SUBMIT_WRAPPER" == "False" ]]; then
  submit_evals
else
  submit_wrapper
fi
