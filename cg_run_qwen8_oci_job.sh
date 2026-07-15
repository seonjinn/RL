#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$1"
CONFIG_FILE="$2"
JOB_VENV_DIR="$3"
MAX_STEPS="$4"
shift 4

SUBMODULE_MEGATRON="${PROJECT_ROOT}/3rdparty/Megatron-LM-workspace/Megatron-LM"
export PYTHONPATH="${SUBMODULE_MEGATRON}:${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_HOME=/usr/local/cuda
export NRL_IGNORE_VERSION_MISMATCH=1
export NEMO_RL_VENV_DIR="${JOB_VENV_DIR}"
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-true}"

if [[ -n "${UV_CACHE_DIR_OVERRIDE:-}" ]]; then
    export UV_CACHE_DIR="${UV_CACHE_DIR_OVERRIDE}"
fi

cd "${PROJECT_ROOT}"

uv run ./examples/run_grpo.py \
    --config "${CONFIG_FILE}" \
    cluster.num_nodes=1 \
    cluster.gpus_per_node=4 \
    grpo.max_num_steps="${MAX_STEPS}" \
    grpo.async_grpo.enabled=false \
    grpo.val_period=1000 \
    checkpointing.enabled=false \
    logger.wandb_enabled=true \
logger.wandb.project=nemo-rl-cudagraph \
    logger.tensorboard_enabled=false \
    "$@"
