#!/usr/bin/env bash

# Submit one independently schedulable Qwen3-30B-A3B CUDA Graph condition.
# Example:
#   CONDITION=adapter-attn STEPS=20 \
#     ./experiments/cuda_graph/launch_qwen30_moe_cg_comparison_ptyche.sh

set -euo pipefail

CONDITION=${CONDITION:?Set CONDITION to <current|pr5672|pr4359|adapter>-<nocg|attn|moe-router|attn-moe-router>.}
STEPS=${STEPS:-20}
RUN_TAG=${RUN_TAG:-${CONDITION}-steps${STEPS}}
CURRENT_WORKTREE=${CURRENT_WORKTREE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-cgseqpack-pr5783-ptyche-runtime-20260716}
PR5672_WORKTREE=${PR5672_WORKTREE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-cgseqpack-pr5672-vs-pr5783-ptyche-20260716}
PR4359_WORKTREE=${PR4359_WORKTREE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-cgseqpack-pr4359-vs-pr5783-ptyche-20260716}
ADAPTER_WORKTREE=${ADAPTER_WORKTREE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-cgseqpack-pr5672-adapter-ptyche-20260719}
CONTAINER=${CONTAINER:-/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/containers/nemo_rl_nightly_20260715.sqsh}
HF_HOME=${HF_HOME:-/lustre/fsw/coreai_dlalgo_llm/users/sna/hf}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/checkpoints}
ACCOUNT=${ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-batch}

case "${CONDITION}" in
  current-* )
    IMPLEMENTATION=current
    WORKTREE=${CURRENT_WORKTREE}
    ;;
  pr5672-* )
    IMPLEMENTATION=pr5672
    WORKTREE=${PR5672_WORKTREE}
    ;;
  pr4359-* )
    IMPLEMENTATION=pr4359
    WORKTREE=${PR4359_WORKTREE}
    ;;
  adapter-* )
    IMPLEMENTATION=adapter
    WORKTREE=${ADAPTER_WORKTREE}
    ;;
  *)
    echo "Unknown CONDITION: ${CONDITION}" >&2
    exit 2
    ;;
esac

case "${CONDITION#*-}" in
  nocg)
    if [[ "${IMPLEMENTATION}" == "adapter" ]]; then
      RECIPE=grpo-qwen3-30ba3b-4n4g-nocg-adapter.yaml
    else
      RECIPE=grpo-qwen3-30ba3b-4n4g-nocg-w3.yaml
    fi
    ;;
  attn)
    if [[ "${IMPLEMENTATION}" == "adapter" ]]; then
      RECIPE=grpo-qwen3-30ba3b-4n4g-cg-attn.yaml
    else
      RECIPE=grpo-qwen3-30ba3b-4n4g-cg-attn-w3.yaml
    fi
    ;;
  moe-router)
    RECIPE=grpo-qwen3-30ba3b-4n4g-cg-moe-router-w3.yaml
    ;;
  attn-moe-router)
    RECIPE=grpo-qwen3-30ba3b-4n4g-cg-attn-moe-router-w3.yaml
    ;;
  *)
    echo "Unknown CUDA Graph scope in CONDITION: ${CONDITION}" >&2
    exit 2
    ;;
esac

ROUTER_DTYPE_OVERRIDE=""
if [[ "${IMPLEMENTATION}" == "adapter" ]] && [[ "${CONDITION#*-}" == *moe-router ]]; then
  # These FP32 router runs are diagnostics only; do not use them for accuracy claims.
  ROUTER_DTYPE_OVERRIDE="policy.megatron_cfg.moe_router_dtype=fp32"
fi

if [[ ! -s "${HF_HOME}/token" ]]; then
  echo "Missing Hugging Face token at ${HF_HOME}/token" >&2
  exit 2
fi

if [[ ! -f "${WORKTREE}/ray.sub" ]]; then
  echo "Missing worktree or ray.sub: ${WORKTREE}" >&2
  exit 2
fi

LOG_BASE="${WORKTREE}/experiments/cuda_graph/logs"
CONFIG="${WORKTREE}/examples/configs/recipes/llm/performance/${RECIPE}"
CHECKPOINT_DIR="${CHECKPOINT_ROOT}/qwen3-30b-a3b-${IMPLEMENTATION}-20260716"
CHECKPOINT_READY_FILE="${CHECKPOINT_DIR}/Qwen/Qwen3-30B-A3B/iter_0000000/run_config.yaml"

# The no-CG baseline performs the one-time HF-to-Megatron conversion for its
# implementation. Other conditions must not observe the partially-created
# directory while that conversion is still writing its run config.
if [[ "${CONDITION#*-}" != "nocg" && ! -f "${CHECKPOINT_READY_FILE}" ]]; then
  echo "Megatron checkpoint is not ready: ${CHECKPOINT_READY_FILE}" >&2
  echo "Run ${IMPLEMENTATION}-nocg first and wait for its conversion to finish." >&2
  exit 3
fi

mkdir -p "${LOG_BASE}" "${CHECKPOINT_DIR}"

echo "condition=${CONDITION} implementation=${IMPLEMENTATION} steps=${STEPS}"
echo "worktree=${WORKTREE} recipe=${RECIPE}"
if [[ -n "${ROUTER_DTYPE_OVERRIDE}" ]]; then
  echo "router_dtype=fp32 (diagnostic only; exclude from production accuracy)"
fi
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
  logger.log_dir=logs/qwen30b-a3b-moe-cg/${RUN_TAG} \\
  logger.wandb.name=${RUN_TAG} ${ROUTER_DTYPE_OVERRIDE}
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
  --nodes=4 \
  --exclusive \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --time=04:00:00 \
  --job-name="${ACCOUNT}-q30.${CONDITION}" \
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
    --nodes=4 \
    --exclusive \
    --account="${ACCOUNT}" \
    --partition="${PARTITION}" \
    --time=04:00:00 \
    --job-name="${ACCOUNT}-q30.${CONDITION}" \
    "${WORKTREE}/ray.sub"
fi
