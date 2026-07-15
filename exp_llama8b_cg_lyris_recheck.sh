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
LOG_BASE="${SCRIPT_DIR}/experiments/llama8_lyris_recheck_$(date +%Y%m%d)"

mkdir -p "${LOG_BASE}"

submit_one() {
    local name="$1"
    local config_file="$2"
    local extra="$3"

    local command
    command="NRL_FORCE_REBUILD_VENVS=true CG_COUNT_LOG=1 uv run ./examples/run_grpo.py \
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

submit_one "ll8_nocg_m20_lyris" "examples/configs/recipes/llm/performance/grpo-llama3.1-8b-instruct-1n4g-nocg.yaml" ""
submit_one "ll8_attn_w3_m20_lyris" "examples/configs/recipes/llm/performance/grpo-llama3.1-8b-instruct-1n4g-cg.yaml" ""
submit_one "ll8_attn_w6_m20_lyris" "examples/configs/recipes/llm/performance/grpo-llama3.1-8b-instruct-1n4g-cg.yaml" "policy.megatron_cfg.cuda_graph_warmup_steps=6"
submit_one "ll8_mlp_w3_m20_lyris" "examples/configs/recipes/llm/performance/grpo-llama3.1-8b-instruct-1n4g-cg.yaml" "policy.megatron_cfg.cuda_graph_scope=mlp policy.megatron_cfg.cuda_graph_warmup_steps=3"
submit_one "ll8_mlp_w6_m20_lyris" "examples/configs/recipes/llm/performance/grpo-llama3.1-8b-instruct-1n4g-cg.yaml" "policy.megatron_cfg.cuda_graph_scope=mlp policy.megatron_cfg.cuda_graph_warmup_steps=6"
submit_one "ll8_attnmlp_w3_m20_lyris" "examples/configs/recipes/llm/performance/grpo-llama3.1-8b-instruct-1n4g-cg-attn-mlp-w6.yaml" "policy.megatron_cfg.cuda_graph_warmup_steps=3"
submit_one "ll8_attnmlp_w6_m20_lyris" "examples/configs/recipes/llm/performance/grpo-llama3.1-8b-instruct-1n4g-cg-attn-mlp-w6.yaml" ""
