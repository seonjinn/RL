#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
cd "${SCRIPT_DIR}"

REFERENCE_DIR="${SCRIPT_DIR}"
ASSET_DIR="${REFERENCE_DIR}"
if [[ ! -f "${ASSET_DIR}/cluster_config.sh" ]]; then
    ASSET_DIR="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl"
fi
RAY_SUB_DEFAULT="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/ray.sub"
RAY_SUB="${RAY_SUB_OVERRIDE:-${RAY_SUB_DEFAULT}}"
if [[ ! -f "${RAY_SUB}" ]]; then
    RAY_SUB="${REFERENCE_DIR}/ray.sub"
fi
CONTAINER_DEFAULT="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/HybridEP_test/nemo_rl_nightly_20260409.sqsh"
UV_CACHE_DIR_OVERRIDE_DEFAULT="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/.cache/uv"
PROJECT_ROOT="${PROJECT_ROOT_OVERRIDE:-${SCRIPT_DIR}}"
LOG_ROOT="${LOG_ROOT_OVERRIDE:-${PROJECT_ROOT}}"
LOG_BASE="${LOG_ROOT}/experiments/qwen8_oci_recheck_$(date +%Y%m%d)"
JOB_FILTER="${JOB_FILTER:-}"
MAX_STEPS="${MAX_STEPS:-20}"
REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-true}"

source "${ASSET_DIR}/cluster_config.sh"
setup_cluster_config "batch"
export_cluster_config

CONTAINER="${CONTAINER_OVERRIDE:-${CONTAINER_DEFAULT}}"
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
    local job_venv_dir="${PROJECT_ROOT}/job_venvs/${name}"

    if [[ -n "${JOB_FILTER}" && "${name}" != *"${JOB_FILTER}"* ]]; then
        return
    fi

    local command
    command="cd ${PROJECT_ROOT} && uv run ./examples/run_grpo.py \
--config ${config_file} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
grpo.max_num_steps=${MAX_STEPS} \
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
    NEMO_RL_VENV_DIR="${job_venv_dir}" \
    NRL_FORCE_REBUILD_VENVS="${REBUILD_VENVS}" \
    NRL_IGNORE_VERSION_MISMATCH=1 \
    CG_COUNT_LOG=1 \
    UV_CACHE_DIR_OVERRIDE="${UV_CACHE_DIR_OVERRIDE:-${UV_CACHE_DIR_OVERRIDE_DEFAULT}}" \
    sbatch \
        --nodes="${NUM_NODES}" \
        --account="${ACCOUNT}" \
        --job-name="${name}" \
        --partition="${PARTITION}" \
        --time=04:00:00 \
        ${GRES_FLAG} \
        "${RAY_SUB}"
}

submit_one "qw8_nocg_m20_script" "examples/configs/recipes/llm/performance/grpo-qwen3-8b-1n4g-nocg.yaml" ""
submit_one "qw8_attn_w3_m20_script" "examples/configs/recipes/llm/performance/grpo-qwen3-8b-1n4g-cg-attn-w3.yaml" ""
submit_one "qw8_attn_w6_m20_script" "examples/configs/recipes/llm/performance/grpo-qwen3-8b-1n4g-cg-attn-w6.yaml" ""
submit_one "qw8_mlp_w3_m20_script" "examples/configs/recipes/llm/performance/grpo-qwen3-8b-1n4g-cg-attn-mlp-w3.yaml" "policy.megatron_cfg.cuda_graph_scope=mlp"
submit_one "qw8_mlp_w6_m20_script" "examples/configs/recipes/llm/performance/grpo-qwen3-8b-1n4g-cg-attn-mlp-w6.yaml" "policy.megatron_cfg.cuda_graph_scope=mlp"
submit_one "qw8_attnmlp_w3_m20_script" "examples/configs/recipes/llm/performance/grpo-qwen3-8b-1n4g-cg-attn-mlp-w3.yaml" ""
submit_one "qw8_attnmlp_w6_m20_script" "examples/configs/recipes/llm/performance/grpo-qwen3-8b-1n4g-cg-attn-mlp-w6.yaml" ""
