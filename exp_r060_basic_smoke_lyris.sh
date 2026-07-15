#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

RAY_SUB="${SCRIPT_DIR}/ray.sub"
CONTAINER="/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly.sqsh"
ACCOUNT="coreai_dlalgo_llm"
PARTITION="gb200"
NUM_NODES=1
GPUS_PER_NODE=4
HF_HOME="/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home"
HF_DATASETS_CACHE="${HF_HOME}/cache"
MOUNTS="/lustre:/lustre"
LOG_BASE="${SCRIPT_DIR}/experiments/r060_basic_smoke_lyris_$(date +%Y%m%d)"

mkdir -p "${LOG_BASE}"

submit_one() {
    local name="$1"
    local config_file="$2"
    local extra="$3"

    local command
    command="NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo.py \
--config ${config_file} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
grpo.max_num_steps=20 \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
logger.wandb_enabled=true \
logger.wandb.project=nemo-rl-cudagraph \
logger.tensorboard_enabled=false \
${extra}"

    echo "[SUBMIT] ${name} -> ${config_file} ${extra}"
    COMMAND="${command}" \
    CONTAINER="${CONTAINER}" \
    HF_HOME="${HF_HOME}" \
    HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
    MOUNTS="${MOUNTS}" \
    BASE_LOG_DIR="${LOG_BASE}" \
    GPUS_PER_NODE="${GPUS_PER_NODE}" \
    sbatch \
        --nodes="${NUM_NODES}" \
        --account="${ACCOUNT}" \
        --job-name="${name}" \
        --partition="${PARTITION}" \
        --time=04:00:00 \
        --segment "${NUM_NODES}" \
        "${RAY_SUB}"
}

submit_one \
    "r060-stock-ll8-nocg" \
    "examples/configs/recipes/llm/grpo-llama3.1-8b-instruct-4n4g-fsdp2tp1-long.v3.yaml" \
    ""

submit_one \
    "r060-stock-qw8-nocg" \
    "examples/configs/recipes/llm/grpo-qwen3-8b-base-1n8g-fp8-kvcache-megatron.yaml" \
    ""
