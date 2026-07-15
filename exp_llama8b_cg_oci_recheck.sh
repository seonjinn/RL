#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

REFERENCE_DIR="${SCRIPT_DIR}"
RAY_SUB="${REFERENCE_DIR}/ray.sub"
ASSET_DIR="${REFERENCE_DIR}"
if [[ ! -f "${ASSET_DIR}/cluster_config.sh" || ! -e "${ASSET_DIR}/nemo_rl_nightly.sqsh" ]]; then
    ASSET_DIR="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl"
fi
LOG_BASE="${SCRIPT_DIR}/experiments/llama8_oci_recheck_$(date +%Y%m%d)"

source "${ASSET_DIR}/cluster_config.sh"
setup_cluster_config "batch"
export_cluster_config

CONTAINER="${ASSET_DIR}/nemo_rl_nightly.sqsh"
export CONTAINER

ACCOUNT="coreai_dlalgo_nemorl"
NUM_NODES=1
GPUS_PER_NODE=4
export GPUS_PER_NODE

mkdir -p "${LOG_BASE}"

submit_one() {
    local name="$1"
    local config_file="$2"
    local extra="$3"

    local command
    command="NRL_FORCE_REBUILD_VENVS=true NRL_IGNORE_VERSION_MISMATCH=1 CG_COUNT_LOG=1 uv run ./examples/run_grpo.py \
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
        ${GRES_FLAG} \
        "${RAY_SUB}"
}

submit_one "ll8_nocg_m20_script" "examples/configs/recipes/llm/performance/grpo-llama3.1-8b-instruct-1n4g-nocg.yaml" ""
submit_one "ll8_attn_w3_m20_script" "examples/configs/recipes/llm/performance/grpo-llama3.1-8b-instruct-1n4g-cg.yaml" ""
submit_one "ll8_attn_w6_m20_script" "examples/configs/recipes/llm/performance/grpo-llama3.1-8b-instruct-1n4g-cg.yaml" "policy.megatron_cfg.cuda_graph_warmup_steps=6"
submit_one "ll8_mlp_w3_m20_script" "examples/configs/recipes/llm/performance/grpo-llama3.1-8b-instruct-1n4g-cg-mlp-w3.yaml" ""
submit_one "ll8_mlp_w6_m20_script" "examples/configs/recipes/llm/performance/grpo-llama3.1-8b-instruct-1n4g-cg-mlp-w3.yaml" "policy.megatron_cfg.cuda_graph_warmup_steps=6"
submit_one "ll8_attnmlp_w3_m20_script" "examples/configs/recipes/llm/performance/grpo-llama3.1-8b-instruct-1n4g-cg-attn-mlp-w3.yaml" ""
submit_one "ll8_attnmlp_w6_m20_script" "examples/configs/recipes/llm/performance/grpo-llama3.1-8b-instruct-1n4g-cg-attn-mlp-w6.yaml" ""
