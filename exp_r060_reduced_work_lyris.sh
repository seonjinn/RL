#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

RAY_SUB="${SCRIPT_DIR}/ray.sub"
CONFIG_ROOT="examples/configs/recipes/llm/performance"
CONTAINER="/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly.sqsh"
ACCOUNT="coreai_dlalgo_llm"
PARTITION="gb200"
HF_HOME="/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home"
HF_DATASETS_CACHE="${HF_HOME}/cache"
MOUNTS="/lustre:/lustre"
LOG_BASE="${SCRIPT_DIR}/experiments/r060_reduced_work_lyris_$(date +%Y%m%d_%H%M%S)"

mkdir -p "${LOG_BASE}"

YAMLS=(
    grpo-llama3.1-8b-instruct-1n4g-nocg-short2048.yaml
    grpo-llama3.1-8b-instruct-1n4g-cg-attn-w3-short2048.yaml
    grpo-llama3.1-8b-instruct-1n4g-cg-mlp-w3-short2048.yaml
    grpo-llama3.1-8b-instruct-1n4g-cg-attn-mlp-w3-short2048.yaml
    grpo-qwen3-8b-1n4g-nocg-short2048.yaml
    grpo-qwen3-8b-1n4g-cg-attn-w3-short2048.yaml
    grpo-qwen3-8b-1n4g-cg-mlp-w3-short2048.yaml
    grpo-qwen3-8b-1n4g-cg-attn-mlp-w3-short2048.yaml
)

submit_one() {
    local yaml_name="$1"
    local config_file="${CONFIG_ROOT}/${yaml_name}"
    local base="${yaml_name%.yaml}"
    local nodes="1"
    local gpus_per_node="4"

    local command
    command="NRL_FORCE_REBUILD_VENVS=true CG_COUNT_LOG=1 uv run ./examples/run_grpo.py \
--config ${config_file} \
cluster.num_nodes=${nodes} \
cluster.gpus_per_node=${gpus_per_node} \
grpo.max_num_steps=20 \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
logger.wandb_enabled=true \
logger.wandb.project=nemo-rl-cudagraph \
logger.tensorboard_enabled=false"

    echo "[SUBMIT] ${base}"
    COMMAND="${command}" \
    CONTAINER="${CONTAINER}" \
    HF_HOME="${HF_HOME}" \
    HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
    MOUNTS="${MOUNTS}" \
    BASE_LOG_DIR="${LOG_BASE}" \
    GPUS_PER_NODE="${gpus_per_node}" \
    sbatch \
        --nodes="${nodes}" \
        --account="${ACCOUNT}" \
        --job-name="${base}" \
        --partition="${PARTITION}" \
        --time=04:00:00 \
        --segment "${nodes}" \
        "${RAY_SUB}"
}

for yaml_name in "${YAMLS[@]}"; do
    submit_one "${yaml_name}"
done
