#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

RAY_SUB="${SCRIPT_DIR}/ray.sub"
CONFIG_ROOT="examples/configs/recipes/llm/performance"
CONTAINER="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/nemo_rl_nightly_20260502.sqsh"
ACCOUNT="coreai_dlalgo_nemorl"
PARTITION="batch"
HF_HOME="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home"
HF_DATASETS_CACHE="${HF_HOME}/cache"
MOUNTS="/lustre:/lustre"
LOG_BASE="${SCRIPT_DIR}/experiments/r060_q30_nopack_valid_offline_w3_oci_$(date +%Y%m%d_%H%M%S)"
JOB_SUFFIX="${JOB_SUFFIX:--offline-w3}"

mkdir -p "${LOG_BASE}"

YAMLS=(
    grpo-qwen3-30ba3b-4n4g-nocg-nopack.yaml
    grpo-qwen3-30ba3b-4n4g-cg-attn-w3-nopack.yaml
    grpo-qwen3-30ba3b-4n4g-cg-moe-router-w3-nopack.yaml
    grpo-qwen3-30ba3b-4n4g-cg-attn-moe-router-w3-nopack.yaml
)

submit_one() {
    local yaml_name="$1"
    local config_file="${CONFIG_ROOT}/${yaml_name}"
    local base="${yaml_name%.yaml}"
    local job_name="${base}${JOB_SUFFIX}"
    local nodes="4"
    local gpus_per_node="4"

    local command
    command="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1 \
NRL_FORCE_REBUILD_VENVS=true CG_COUNT_LOG=1 uv run ./examples/run_grpo.py \
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

    echo "[SUBMIT] ${job_name}"
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
        --job-name="${job_name}" \
        --partition="${PARTITION}" \
        --time=04:00:00 \
        --gres="gpu:${gpus_per_node}" \
        --segment "${nodes}" \
        "${RAY_SUB}"
}

for yaml_name in "${YAMLS[@]}"; do
    submit_one "${yaml_name}"
done
