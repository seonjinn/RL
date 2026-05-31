#!/bin/bash
#SBATCH --job-name=q235b-eagle3-online
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=batch
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Required at submission time:
#   sbatch --account=<account> --nodes=<nodes> \
#     --export=ALL,INPUT_DATA=/path/conversations.jsonl,OUTPUT_DIR=/path/modelopt_online,CHAT_TEMPLATE=/path/qwen3_generation_template.jinja2 \
#     experiments/eagle3_qwen3_235b/slurm_online_train.sbatch
#
# Online training keeps the 235B verifier in the ModelOpt training loop. Prefer
# this for GB200-class capacity or sufficiently large H100 allocations. The
# offline dump/train path remains the default first path for Qwen3-235B SWE.

REPO_ROOT="${REPO_ROOT:-$PWD}"
SCRIPT="$REPO_ROOT/experiments/eagle3_qwen3_235b/modelopt_qwen3_235b_online_train.sh"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-235B-A22B-Thinking-2507}"
INPUT_DATA="${INPUT_DATA:?set INPUT_DATA to the ModelOpt conversation JSONL}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/qwen3_235b_thinking_eagle3_modelopt_online}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
TRAINING_SEQ_LEN="${TRAINING_SEQ_LEN:-16384}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1.0e-4}"
SAVE_STEPS="${SAVE_STEPS:-512}"
ANSWER_ONLY_LOSS="${ANSWER_ONLY_LOSS:-true}"
NUM_NODES="${NUM_NODES:-${SLURM_JOB_NUM_NODES:-1}}"
HEAD_NODE_IP="${HEAD_NODE_IP:-}"
DRY_RUN="${DRY_RUN:-false}"
CONTAINER="${CONTAINER:-}"
MOUNTS="${MOUNTS:-}"

if [[ "$ANSWER_ONLY_LOSS" == "true" || "$ANSWER_ONLY_LOSS" == "True" ]]; then
  if [[ -z "$CHAT_TEMPLATE" ]]; then
    echo "CHAT_TEMPLATE is required for online ANSWER_ONLY_LOSS=true." >&2
    exit 1
  fi
fi

if [[ "$NUM_NODES" != "1" && -z "$HEAD_NODE_IP" ]]; then
  HEAD_NODE="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
  HEAD_NODE_IP="$(getent ahostsv4 "$HEAD_NODE" | awk '{print $1; exit}')"
  if [[ -z "$HEAD_NODE_IP" ]]; then
    echo "HEAD_NODE_IP could not be derived; set HEAD_NODE_IP for multi-node online training." >&2
    exit 1
  fi
fi

SRUN_CONTAINER_ARGS=()
if [[ -n "$CONTAINER" ]]; then
  SRUN_CONTAINER_ARGS+=(--container-image="$CONTAINER")
  [[ -n "$MOUNTS" ]] && SRUN_CONTAINER_ARGS+=(--container-mounts="$MOUNTS")
  SRUN_CONTAINER_ARGS+=(--container-workdir="$REPO_ROOT")
fi

mkdir -p logs

srun "${SRUN_CONTAINER_ARGS[@]}" --ntasks="$NUM_NODES" --ntasks-per-node=1 bash -lc '
  set -euo pipefail
  export BASE_MODEL="'"$BASE_MODEL"'"
  export INPUT_DATA="'"$INPUT_DATA"'"
  export OUTPUT_DIR="'"$OUTPUT_DIR"'"
  export CHAT_TEMPLATE="'"$CHAT_TEMPLATE"'"
  export TRAINING_SEQ_LEN="'"$TRAINING_SEQ_LEN"'"
  export PER_DEVICE_TRAIN_BATCH_SIZE="'"$PER_DEVICE_TRAIN_BATCH_SIZE"'"
  export NUM_TRAIN_EPOCHS="'"$NUM_TRAIN_EPOCHS"'"
  export LEARNING_RATE="'"$LEARNING_RATE"'"
  export SAVE_STEPS="'"$SAVE_STEPS"'"
  export ANSWER_ONLY_LOSS="'"$ANSWER_ONLY_LOSS"'"
  export NUM_NODES="'"$NUM_NODES"'"
  export HEAD_NODE_IP="'"$HEAD_NODE_IP"'"
  export DRY_RUN="'"$DRY_RUN"'"
  cd "'"$REPO_ROOT"'"
  bash "'"$SCRIPT"'"
'
