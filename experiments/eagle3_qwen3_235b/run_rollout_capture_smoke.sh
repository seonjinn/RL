#!/usr/bin/env bash
set -euo pipefail

# Plan or submit a short Qwen3-235B SWE RL run that captures train_data_step*.jsonl
# for Eagle3 draft training. Defaults to DRY_RUN=true and does not submit.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3}"
ROLLOUT_LOG_DIR="${ROLLOUT_LOG_DIR:-$ARTIFACT_ROOT/rl_rollout_capture_logs/qwen3_235b_swe_capture_smoke}"
OUTPUT_CONVERSATIONS="${OUTPUT_CONVERSATIONS:-$ARTIFACT_ROOT/data/qwen3_235b_swe_rollout_conversations.jsonl}"
SWE_REPO_ROOT="${SWE_REPO_ROOT:-${REPO_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL}}"
CONFIG_FILE="${CONFIG_FILE:-$ROOT_DIR/grpo_qwen3_235b_swe.yaml}"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/env.sh}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$ARTIFACT_ROOT/templates/qwen3_generation_template.jinja2}"
RESOURCE_PROFILE_ENV="${RESOURCE_PROFILE_ENV:-$ARTIFACT_ROOT/reports/eagle3_resource_profile.env}"
SWEGYM_EXAMPLE_DATA="${SWEGYM_EXAMPLE_DATA:-/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/sdevare/repos/ultra/tk-nemo-gym/responses_api_agents/swe_agents/data/example.jsonl}"
SWEGYM_FIXED_DATA="${SWEGYM_FIXED_DATA:-$ARTIFACT_ROOT/data/swegym_example_for_sweagent_with_instance_dict.jsonl}"

if [[ -z "${TRAIN_DATA_PATH:-}" && -f "$SWEGYM_FIXED_DATA" ]]; then
  TRAIN_DATA_PATH="$SWEGYM_FIXED_DATA"
elif [[ -z "${TRAIN_DATA_PATH:-}" && -f "$SWEGYM_EXAMPLE_DATA" ]]; then
  TRAIN_DATA_PATH="$SWEGYM_EXAMPLE_DATA"
fi
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-}"
VAL_DATA_PATH="${VAL_DATA_PATH:-$TRAIN_DATA_PATH}"

if [[ -f "$RESOURCE_PROFILE_ENV" ]]; then
  # Reuse the cluster shape discovered by probe_eagle3_slurm_capacity.py.
  # On oci-hsg this is currently 4 GPUs per node, while the original RL
  # launcher was written for 8-GPU GB200 nodes.
  # shellcheck disable=SC1090
  source "$RESOURCE_PROFILE_ENV"
fi

ROLLOUT_GPUS_PER_NODE="${ROLLOUT_GPUS_PER_NODE:-${TRAIN_GPUS_PER_NODE:-${DUMP_GPUS_PER_NODE:-}}}"
if [[ -n "$ROLLOUT_GPUS_PER_NODE" && -z "${NUM_GPU:-}" && -z "${GPUS_PER_NODE:-}" ]]; then
  export NUM_GPU="$ROLLOUT_GPUS_PER_NODE"
fi

if [[ -n "${NUM_GPU:-}" && "$NUM_GPU" =~ ^[0-9]+$ && "$NUM_GPU" -gt 0 ]]; then
  DEFAULT_ROLLOUT_ACTOR_GPUS="${ROLLOUT_TOTAL_ACTOR_GPUS:-128}"
  DEFAULT_ROLLOUT_GENERATION_GPUS="${ROLLOUT_TOTAL_GENERATION_GPUS:-64}"
  if [[ -z "${NUM_NODES:-}" ]]; then
    export NUM_NODES=$(((DEFAULT_ROLLOUT_ACTOR_GPUS + NUM_GPU - 1) / NUM_GPU))
  fi
  if [[ -z "${NUM_GEN_NODES:-}" ]]; then
    export NUM_GEN_NODES=$(((DEFAULT_ROLLOUT_GENERATION_GPUS + NUM_GPU - 1) / NUM_GPU))
  fi
fi

MAX_NUM_STEPS="${MAX_NUM_STEPS:-1}"
WANDB_NAME="${WANDB_NAME:-qwen3-235b-swe-rollout-capture-smoke}"
EXP_SUFFIX_OVERRIDE="${EXP_SUFFIX_OVERRIDE:-$WANDB_NAME}"
CHECKPOINT_SUBDIR="${CHECKPOINT_SUBDIR:-$WANDB_NAME}"
DRY_RUN="${DRY_RUN:-true}"

AGENT_MAX_TURNS_OVERRIDE="${AGENT_MAX_TURNS_OVERRIDE:-1}"
AGENT_TIMEOUT_OVERRIDE="${AGENT_TIMEOUT_OVERRIDE:-900}"
SWE_TEST_TIMEOUT_OVERRIDE="${SWE_TEST_TIMEOUT_OVERRIDE:-60}"
NUM_VAL_SAMPLES_TO_PRINT="${NUM_VAL_SAMPLES_TO_PRINT:-0}"

mkdir -p "$ROLLOUT_LOG_DIR" "$(dirname "$OUTPUT_CONVERSATIONS")"

export MAX_NUM_STEPS
export WANDB_NAME
export EXP_SUFFIX_OVERRIDE
export CHECKPOINT_SUBDIR
export DRY_RUN
export REPO_ROOT="$SWE_REPO_ROOT"
export CONFIG_FILE
export ENV_FILE
export CHAT_TEMPLATE
export TRAIN_DATA_PATH
export VAL_DATA_PATH
export AGENT_MAX_TURNS="$AGENT_MAX_TURNS_OVERRIDE"
export AGENT_TIMEOUT="$AGENT_TIMEOUT_OVERRIDE"
export SAVE_PERIOD=1000000
export VAL_PERIOD=1000000
export KEEP_TOP_K=1
export SBATCH_DEPENDENCY="${SBATCH_DEPENDENCY:-singleton}"
export EXTRA_HYDRA_OVERRIDES="${EXTRA_HYDRA_OVERRIDES:-} \
logger.log_dir=${ROLLOUT_LOG_DIR} \
logger.wandb_enabled=False \
logger.tensorboard_enabled=False \
logger.mlflow_enabled=False \
logger.swanlab_enabled=False \
logger.num_val_samples_to_print=${NUM_VAL_SAMPLES_TO_PRINT} \
env.should_log_nemo_gym_responses=False \
grpo.val_period=1000000 \
grpo.val_at_start=False \
grpo.val_at_end=False \
checkpointing.save_period=1000000 \
checkpointing.keep_top_k=1 \
env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.agent_max_turns=${AGENT_MAX_TURNS_OVERRIDE} \
env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.swebench_agent_timeout=${AGENT_TIMEOUT_OVERRIDE} \
env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.agent_max_turns=${AGENT_MAX_TURNS_OVERRIDE} \
env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.swebench_agent_timeout=${AGENT_TIMEOUT_OVERRIDE} \
++env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.swebench_tests_timeout=${SWE_TEST_TIMEOUT_OVERRIDE}"

cat <<EOF
Rollout capture smoke
  DRY_RUN=$DRY_RUN
  MAX_NUM_STEPS=$MAX_NUM_STEPS
  ROLLOUT_LOG_DIR=$ROLLOUT_LOG_DIR
  EXPECTED_JSONL=$ROLLOUT_LOG_DIR/train_data_step1.jsonl
  OUTPUT_CONVERSATIONS=$OUTPUT_CONVERSATIONS
  SWE_REPO_ROOT=$SWE_REPO_ROOT
  CONFIG_FILE=$CONFIG_FILE
  CHAT_TEMPLATE=$CHAT_TEMPLATE
  RESOURCE_PROFILE_ENV=$RESOURCE_PROFILE_ENV
  TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-<run_grpo-default>}
  VAL_DATA_PATH=${VAL_DATA_PATH:-<train-data-path>}
  SWEGYM_EXAMPLE_DATA=$SWEGYM_EXAMPLE_DATA
  SWEGYM_FIXED_DATA=$SWEGYM_FIXED_DATA
  NUM_GPU=${NUM_GPU:-<launcher-default>}
  NUM_NODES=${NUM_NODES:-<launcher-default>}
  NUM_GEN_NODES=${NUM_GEN_NODES:-<launcher-default>}

After a submitted run completes, normalize with:
  MODE=rollout INPUT_PATHS="$ROLLOUT_LOG_DIR/train_data_step*.jsonl" OUTPUT_DATA="$OUTPUT_CONVERSATIONS" INCLUDE_METADATA=true bash experiments/eagle3_qwen3_235b/prepare_training_conversations.sh

Track job/corpus state with:
  python3 experiments/eagle3_qwen3_235b/analyze_rollout_capture_job.py --artifact-root "$ARTIFACT_ROOT" --repo-root "$SWE_REPO_ROOT" --rollout-log-dir "$ROLLOUT_LOG_DIR" --output-data "$OUTPUT_CONVERSATIONS"
EOF

if [[ "$DRY_RUN" == "true" || "$DRY_RUN" == "True" ]]; then
  echo "[DRY-RUN] launcher:"
  printf '%q ' bash "$ROOT_DIR/run_grpo_qwen3_235b_swe.sh"
  printf '\n'
  echo "[DRY-RUN] EXTRA_HYDRA_OVERRIDES:"
  printf '%s\n' "$EXTRA_HYDRA_OVERRIDES"
  exit 0
fi

exec bash "$ROOT_DIR/run_grpo_qwen3_235b_swe.sh"
