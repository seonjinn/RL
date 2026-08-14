#!/bin/bash

set -euo pipefail

: "${EXPERIMENT_OUTPUT_DIR:?Set EXPERIMENT_OUTPUT_DIR to a shared filesystem path}"

MAX_NUM_STEPS=${MAX_NUM_STEPS:-20}
RUN_NAME=${RUN_NAME:-qwen3-30ba3b-4n8g-hybridep}
WANDB_ENABLED=${WANDB_ENABLED:-true}

export HYBRID_EP_MULTINODE=0
export NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS:-true}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
export USE_MNNVL=0
export USE_NIXL=0

UV_NO_SYNC=1 uv run env -u UV_NO_SYNC -u UV_FROZEN python examples/run_grpo.py \
  --config examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n8g.yaml \
  grpo.max_num_steps="${MAX_NUM_STEPS}" \
  checkpointing.enabled=false \
  logger.log_dir="${EXPERIMENT_OUTPUT_DIR}/training" \
  logger.wandb_enabled="${WANDB_ENABLED}" \
  logger.wandb.name="${RUN_NAME}"
