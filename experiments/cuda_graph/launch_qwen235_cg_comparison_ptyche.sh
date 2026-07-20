#!/usr/bin/env bash

# Validate or submit one Qwen3-235B-A22B CUDA Graph comparison condition.
# Example:
#   CONDITION=adapter-attn STEPS=20 \
#     ./experiments/cuda_graph/launch_qwen235_cg_comparison_ptyche.sh

set -euo pipefail

CONDITION=${CONDITION:?Set CONDITION to adapter-nocg or adapter-attn.}
STEPS=${STEPS:-20}
RUN_TAG=${RUN_TAG:-${CONDITION}-steps${STEPS}}
ADAPTER_WORKTREE=${ADAPTER_WORKTREE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-pr5672-adapter-ptyche-20260719}
CONTAINER=${CONTAINER:-/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/containers/nemo_rl_nightly_20260715.sqsh}
HF_HOME=${HF_HOME:-/lustre/fsw/coreai_dlalgo_llm/users/sna/hf}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/checkpoints}
ACCOUNT=${ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-batch}

case "${CONDITION}" in
  adapter-nocg)
    RECIPE=grpo-qwen3-235b-16n4g-nocg-adapter.yaml
    ;;
  adapter-attn)
    RECIPE=grpo-qwen3-235b-16n4g-cg-attn-adapter.yaml
    ;;
  *)
    echo "Unknown CONDITION: ${CONDITION}" >&2
    exit 2
    ;;
esac

if [[ ! -s "${HF_HOME}/token" ]]; then
  echo "Missing Hugging Face token at ${HF_HOME}/token" >&2
  exit 2
fi

WORKTREE=${ADAPTER_WORKTREE}
if [[ ! -f "${WORKTREE}/ray.sub" ]]; then
  echo "Missing worktree or ray.sub: ${WORKTREE}" >&2
  exit 2
fi

CONFIG="${WORKTREE}/examples/configs/recipes/llm/performance/${RECIPE}"
if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing recipe: ${CONFIG}" >&2
  exit 2
fi

LOG_BASE="${WORKTREE}/experiments/cuda_graph/logs"
CHECKPOINT_DIR="${CHECKPOINT_ROOT}/qwen3-235b-a22b-adapter-20260720"
CHECKPOINT_READY_FILE="${CHECKPOINT_DIR}/Qwen/Qwen3-235B-A22B/iter_0000000/run_config.yaml"

if [[ "${CONDITION}" == "adapter-attn" && ! -f "${CHECKPOINT_READY_FILE}" ]]; then
  echo "Megatron checkpoint is not ready: ${CHECKPOINT_READY_FILE}" >&2
  echo "Run adapter-nocg first and wait for its conversion to finish." >&2
  exit 3
fi

mkdir -p "${LOG_BASE}"

echo "condition=${CONDITION} steps=${STEPS}"
echo "worktree=${WORKTREE} recipe=${RECIPE}"
git -C "${WORKTREE}" rev-parse HEAD
git -C "${WORKTREE}/3rdparty/Megatron-LM-workspace/Megatron-LM" rev-parse HEAD

read -r -d '' COMMAND <<EOF || true
cd ${WORKTREE}
export NRL_IGNORE_VERSION_MISMATCH=1
export NRL_MEGATRON_CHECKPOINT_DIR=${CHECKPOINT_DIR}
export PYTHONPATH=${WORKTREE}:${WORKTREE}/3rdparty/Megatron-LM-workspace/Megatron-LM:${WORKTREE}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge:\${PYTHONPATH:-}
uv run --locked --extra mcore --directory ${WORKTREE} python ${WORKTREE}/examples/run_grpo.py \\
  --config ${CONFIG} \\
  grpo.max_num_steps=${STEPS} \\
  grpo.val_period=10 \\
  logger.wandb_enabled=false \\
  logger.tensorboard_enabled=false \\
  logger.log_dir=logs/qwen3-235b-a22b-cg/${RUN_TAG} \\
  logger.wandb.name=${RUN_TAG}
EOF

COMMAND="${COMMAND}" \
CONTAINER="${CONTAINER}" \
HF_HOME="${HF_HOME}" \
HF_HUB_CACHE="${HF_HOME}/hub" \
HF_DATASETS_CACHE="${HF_HOME}/datasets" \
MOUNTS="/lustre:/lustre" \
GPUS_PER_NODE=4 \
BASE_LOG_DIR="${LOG_BASE}" \
sbatch --test-only \
  --nodes=16 \
  --segment=16 \
  --exclusive \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --time=04:00:00 \
  --job-name="${ACCOUNT}-q235.${CONDITION}" \
  "${WORKTREE}/ray.sub"

echo "Submission validated. Set SUBMIT=1 to submit this condition."
if [[ "${SUBMIT:-0}" == "1" ]]; then
  COMMAND="${COMMAND}" \
  CONTAINER="${CONTAINER}" \
  HF_HOME="${HF_HOME}" \
  HF_HUB_CACHE="${HF_HOME}/hub" \
  HF_DATASETS_CACHE="${HF_HOME}/datasets" \
  MOUNTS="/lustre:/lustre" \
  GPUS_PER_NODE=4 \
  BASE_LOG_DIR="${LOG_BASE}" \
  sbatch \
    --nodes=16 \
    --segment=16 \
    --exclusive \
    --account="${ACCOUNT}" \
    --partition="${PARTITION}" \
    --time=04:00:00 \
    --job-name="${ACCOUNT}-q235.${CONDITION}" \
    "${WORKTREE}/ray.sub"
fi
