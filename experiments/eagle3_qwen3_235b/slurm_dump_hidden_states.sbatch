#!/bin/bash
#SBATCH --job-name=q235b-eagle3-dump
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
#     --export=ALL,INPUT_DATA=/path/conversations.jsonl,HIDDEN_STATES_DIR=/path/hiddens \
#     experiments/eagle3_qwen3_235b/slurm_dump_hidden_states.sbatch
#
# Multi-node DP:
#   sbatch --nodes=8 --ntasks-per-node=1 --gres=gpu:8 ...
#
# Each Slurm task runs one TRT-LLM process. By default, each process uses TP=8.

REPO_ROOT="${REPO_ROOT:-$PWD}"
SCRIPT="$REPO_ROOT/experiments/eagle3_qwen3_235b/modelopt_qwen3_235b_dump_hidden_states.sh"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-235B-A22B-Thinking-2507}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-16384}"
BACKEND="${BACKEND:-trtllm}"
TP="${TP:-8}"
AUX_LAYERS="${AUX_LAYERS:-1,46,90}"
ANSWER_ONLY_LOSS="${ANSWER_ONLY_LOSS:-true}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
DEBUG_MAX_NUM_CONVERSATIONS="${DEBUG_MAX_NUM_CONVERSATIONS:-}"
MAX_INFLIGHT="${MAX_INFLIGHT:-64}"
ARCH_ENV_FILE="${ARCH_ENV_FILE:-}"
MODELOPT_DIR="${MODELOPT_DIR:-$REPO_ROOT/Model-Optimizer}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"
MOE_EP="${MOE_EP:-}"
MOE_TP="${MOE_TP:-}"
MOE_CP="${MOE_CP:-}"
DP_WORLD_SIZE="${DP_WORLD_SIZE:-${SLURM_NTASKS:-1}}"
DRY_RUN="${DRY_RUN:-false}"
LAUNCHER_PREFIX="${LAUNCHER_PREFIX:-}"
CONTAINER="${CONTAINER:-${EAGLE_CONTAINER:-${EAGLE_IMAGE:-}}}"
MOUNTS="${MOUNTS:-${EAGLE_MOUNTS:-${EAGLE_BIND:-}}}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-$REPO_ROOT}"
SRUN_EXTRA_ARGS="${SRUN_EXTRA_ARGS:-}"

if [[ -z "${INPUT_DATA:-}" ]]; then
  echo "INPUT_DATA is required." >&2
  exit 1
fi
if [[ -z "${HIDDEN_STATES_DIR:-}" ]]; then
  echo "HIDDEN_STATES_DIR is required." >&2
  exit 1
fi
if [[ "$ANSWER_ONLY_LOSS" == "true" || "$ANSWER_ONLY_LOSS" == "True" ]]; then
  if [[ -z "$CHAT_TEMPLATE" ]]; then
    echo "CHAT_TEMPLATE is required for ANSWER_ONLY_LOSS=true." >&2
    exit 1
  fi
fi

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

if [[ -n "$SRUN_EXTRA_ARGS" ]]; then
  # Intentionally split simple srun flags such as: --mpi=pmix --overlap
  # shellcheck disable=SC2206
  srun_extra=($SRUN_EXTRA_ARGS)
else
  srun_extra=()
fi

srun "${srun_extra[@]}" "${SRUN_CONTAINER_ARGS[@]}" --ntasks="${SLURM_NTASKS:-1}" --ntasks-per-node="${SLURM_NTASKS_PER_NODE:-1}" bash -lc '
  set -euo pipefail
  export INPUT_DATA="'"${INPUT_DATA:-}"'"
  export HIDDEN_STATES_DIR="'"${HIDDEN_STATES_DIR:-}"'"
  export BASE_MODEL="'"$BASE_MODEL"'"
  export MAX_SEQ_LEN="'"$MAX_SEQ_LEN"'"
  export BACKEND="'"$BACKEND"'"
  export TP="'"$TP"'"
  export AUX_LAYERS="'"$AUX_LAYERS"'"
  export ANSWER_ONLY_LOSS="'"$ANSWER_ONLY_LOSS"'"
  export CHAT_TEMPLATE="'"$CHAT_TEMPLATE"'"
  export DEBUG_MAX_NUM_CONVERSATIONS="'"$DEBUG_MAX_NUM_CONVERSATIONS"'"
  export MAX_INFLIGHT="'"$MAX_INFLIGHT"'"
  export ARCH_ENV_FILE="'"$ARCH_ENV_FILE"'"
  export MODELOPT_DIR="'"$MODELOPT_DIR"'"
  export TRUST_REMOTE_CODE="'"$TRUST_REMOTE_CODE"'"
  export MOE_EP="'"$MOE_EP"'"
  export MOE_TP="'"$MOE_TP"'"
  export MOE_CP="'"$MOE_CP"'"
  export DP_WORLD_SIZE="'"$DP_WORLD_SIZE"'"
  export DP_RANK="${SLURM_PROCID:-0}"
  export DRY_RUN="'"$DRY_RUN"'"
  export LAUNCHER_PREFIX="'"$LAUNCHER_PREFIX"'"
  cd "'"$REPO_ROOT"'"
  bash "'"$SCRIPT"'"
'
