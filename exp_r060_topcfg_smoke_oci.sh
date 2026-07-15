#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

RAY_SUB="${SCRIPT_DIR}/ray.sub"
CONTAINER="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/nemo_rl_nightly_20260502.sqsh"
ACCOUNT="coreai_dlalgo_nemorl"
PARTITION="batch"
HF_HOME="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home"
HF_DATASETS_CACHE="${HF_HOME}/cache"
MOUNTS="/lustre:/lustre"
LOG_BASE="${SCRIPT_DIR}/experiments/r060_sigprobe_smoke_oci_$(date +%Y%m%d_%H%M%S)"

mkdir -p "${LOG_BASE}"

submit_one() {
    local name="$1"
    local num_nodes="$2"
    local gpus_per_node="$3"
    local config_file="$4"
    local extra="$5"

    local command
    command="NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo.py \
--config ${config_file} \
cluster.num_nodes=${num_nodes} \
cluster.gpus_per_node=${gpus_per_node} \
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
    GPUS_PER_NODE="${gpus_per_node}" \
    sbatch \
        --nodes="${num_nodes}" \
        --account="${ACCOUNT}" \
        --job-name="${name}" \
        --partition="${PARTITION}" \
        --time=04:00:00 \
        --gres="gpu:${gpus_per_node}" \
        --segment "${num_nodes}" \
        "${RAY_SUB}"
}

submit_one \
    "r060sig-oci-ll8" \
    "1" \
    "4" \
    "examples/configs/grpo_math_8B_megatron.yaml" \
    "policy.megatron_cfg.pipeline_model_parallel_size=1"

submit_one \
    "r060sig-oci-qw8" \
    "1" \
    "4" \
    "examples/configs/grpo_math_8B_megatron.yaml" \
    "policy.model_name=Qwen/Qwen3-8B-Base policy.tokenizer.name=Qwen/Qwen3-8B-Base policy.megatron_cfg.converter_type=Qwen3ForCausalLM policy.megatron_cfg.pipeline_model_parallel_size=1"

submit_one \
    "r060sig-oci-q30" \
    "4" \
    "4" \
    "examples/configs/grpo_math_qwen30ba3b_megatron.yaml" \
    ""
