#!/bin/bash
#SBATCH --job-name=q235b-eagle3-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=batch
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Required at submission time:
#   sbatch --account=<account> \
#     --export=ALL,HIDDEN_STATES_DIR=/path/hiddens,OUTPUT_DIR=/path/modelopt_ckpt,USE_FAKE_BASE_FOR_OFFLINE=true \
#     experiments/eagle3_qwen3_235b/slurm_offline_train.sbatch

REPO_ROOT="${REPO_ROOT:-$PWD}"
SCRIPT="$REPO_ROOT/experiments/eagle3_qwen3_235b/modelopt_qwen3_235b_offline_train.sh"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-235B-A22B-Thinking-2507}"
TRAINING_SEQ_LEN="${TRAINING_SEQ_LEN:-16384}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1.0e-4}"
SAVE_STEPS="${SAVE_STEPS:-512}"
ANSWER_ONLY_LOSS="${ANSWER_ONLY_LOSS:-true}"
USE_FAKE_BASE_FOR_OFFLINE="${USE_FAKE_BASE_FOR_OFFLINE:-true}"
MAX_STEPS="${MAX_STEPS:-}"
DATA_SAMPLE_SIZE="${DATA_SAMPLE_SIZE:-}"
EAGLE_TTT_STEPS="${EAGLE_TTT_STEPS:-3}"
EAGLE_LOSS_DECAY_FACTOR="${EAGLE_LOSS_DECAY_FACTOR:-0.9}"
EAGLE_USE_TORCH_COMPILE="${EAGLE_USE_TORCH_COMPILE:-false}"
ARCH_ENV_FILE="${ARCH_ENV_FILE:-}"
MODELOPT_DIR="${MODELOPT_DIR:-$REPO_ROOT/Model-Optimizer}"
DRY_RUN="${DRY_RUN:-false}"
CONTAINER="${CONTAINER:-${EAGLE_CONTAINER:-${EAGLE_IMAGE:-}}}"
MOUNTS="${MOUNTS:-${EAGLE_MOUNTS:-${EAGLE_BIND:-}}}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-$REPO_ROOT}"
TRAIN_PYTHONPATH="$MODELOPT_DIR:$REPO_ROOT:${PYTHONPATH:-}"

SRUN_CONTAINER_ARGS=()
if [[ -n "$CONTAINER" ]]; then
  echo "Using container image: $CONTAINER"
  SRUN_CONTAINER_ARGS+=(--container-image="$CONTAINER")
  [[ -n "$MOUNTS" ]] && SRUN_CONTAINER_ARGS+=(--container-mounts="$MOUNTS")
  SRUN_CONTAINER_ARGS+=(--container-workdir="$CONTAINER_WORKDIR")
else
  echo "No container image configured; running on host Python." >&2
fi

mkdir -p logs

srun "${SRUN_CONTAINER_ARGS[@]}" --ntasks=1 bash -lc '
  set -euo pipefail
  export HIDDEN_STATES_DIR="'"${HIDDEN_STATES_DIR:-}"'"
  export OUTPUT_DIR="'"${OUTPUT_DIR:-}"'"
  export BASE_MODEL="'"$BASE_MODEL"'"
  export TRAINING_SEQ_LEN="'"$TRAINING_SEQ_LEN"'"
  export PER_DEVICE_TRAIN_BATCH_SIZE="'"$PER_DEVICE_TRAIN_BATCH_SIZE"'"
  export NUM_TRAIN_EPOCHS="'"$NUM_TRAIN_EPOCHS"'"
  export LEARNING_RATE="'"$LEARNING_RATE"'"
  export SAVE_STEPS="'"$SAVE_STEPS"'"
  export ANSWER_ONLY_LOSS="'"$ANSWER_ONLY_LOSS"'"
  export USE_FAKE_BASE_FOR_OFFLINE="'"$USE_FAKE_BASE_FOR_OFFLINE"'"
  export MAX_STEPS="'"$MAX_STEPS"'"
  export DATA_SAMPLE_SIZE="'"$DATA_SAMPLE_SIZE"'"
  export EAGLE_TTT_STEPS="'"$EAGLE_TTT_STEPS"'"
  export EAGLE_LOSS_DECAY_FACTOR="'"$EAGLE_LOSS_DECAY_FACTOR"'"
  export EAGLE_USE_TORCH_COMPILE="'"$EAGLE_USE_TORCH_COMPILE"'"
  export ARCH_ENV_FILE="'"$ARCH_ENV_FILE"'"
  export MODELOPT_DIR="'"$MODELOPT_DIR"'"
  export DRY_RUN="'"$DRY_RUN"'"
  export PYTHONPATH="'"$TRAIN_PYTHONPATH"'"
  cd "'"$REPO_ROOT"'"
  bash "'"$SCRIPT"'"
'
