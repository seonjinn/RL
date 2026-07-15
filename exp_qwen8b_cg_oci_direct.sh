#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
cd "${SCRIPT_DIR}"

ASSET_DIR="${SCRIPT_DIR}"
if [[ ! -f "${ASSET_DIR}/cluster_config.sh" ]]; then
    ASSET_DIR="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl"
fi

PROJECT_ROOT="${PROJECT_ROOT_OVERRIDE:-${SCRIPT_DIR}}"
LOG_ROOT="${LOG_ROOT_OVERRIDE:-${PROJECT_ROOT}}"
LOG_BASE="${LOG_ROOT}/experiments/qwen8_oci_direct_$(date +%Y%m%d)"
CONTAINER_DEFAULT="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/nemo_rl_nightly.sqsh"
UV_CACHE_DIR_OVERRIDE_DEFAULT="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/.cache/uv"
JOB_FILTER="${JOB_FILTER:-}"
MAX_STEPS="${MAX_STEPS:-20}"
REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-true}"

source "${ASSET_DIR}/cluster_config.sh"
setup_cluster_config "batch"
export_cluster_config

CONTAINER="${CONTAINER_OVERRIDE:-${CONTAINER_DEFAULT}}"
ACCOUNT="coreai_dlalgo_nemorl"
UV_CACHE_PATH="${UV_CACHE_DIR_OVERRIDE:-${UV_CACHE_DIR_OVERRIDE_DEFAULT}}"

mkdir -p "${LOG_BASE}"

submit_one() {
    local name="$1"
    local config_file="$2"
    shift 2
    local -a extra_args=("$@")
    local job_venv_dir="${PROJECT_ROOT}/job_venvs/${name}"
    local container_mounts="${MOUNTS}"
    local -a srun_cmd
    local -a sbatch_cmd
    local wrap_cmd

    if [[ -n "${JOB_FILTER}" && "${name}" != *"${JOB_FILTER}"* ]]; then
        return
    fi

    mkdir -p "${job_venv_dir}"
    mkdir -p "${UV_CACHE_PATH}"

    if [[ -n "${container_mounts}" ]]; then
        container_mounts+=",${UV_CACHE_PATH}:/root/.cache/uv"
    else
        container_mounts="${UV_CACHE_PATH}:/root/.cache/uv"
    fi

    srun_cmd=(
        srun
        --nodes=1
        --ntasks=1
        --no-container-mount-home
        "--container-image=${CONTAINER}"
        "--container-mounts=${container_mounts}"
        "--container-workdir=${PROJECT_ROOT}"
    )
    if [[ -n "${GRES_FLAG}" ]]; then
        srun_cmd+=(${GRES_FLAG})
    fi
    srun_cmd+=(
        "${PROJECT_ROOT}/cg_run_qwen8_oci_job.sh"
        "${PROJECT_ROOT}"
        "${config_file}"
        "${job_venv_dir}"
        "${MAX_STEPS}"
    )
    srun_cmd+=("${extra_args[@]}")

    printf -v wrap_cmd '%q ' "${srun_cmd[@]}"

    sbatch_cmd=(
        sbatch
        "--nodes=1"
        "--account=${ACCOUNT}"
        "--job-name=${name}"
        "--partition=${PARTITION}"
        "--time=04:00:00"
        "--output=${LOG_BASE}/slurm-%j.out"
        "--export=ALL,UV_CACHE_DIR_OVERRIDE=${UV_CACHE_PATH},NRL_FORCE_REBUILD_VENVS=${REBUILD_VENVS}"
    )
    if [[ -n "${GRES_FLAG}" ]]; then
        sbatch_cmd+=(${GRES_FLAG})
    fi
    sbatch_cmd+=(--wrap "${wrap_cmd}")

    echo "[SUBMIT] ${name} -> ${config_file} ${extra_args[*]}"
    "${sbatch_cmd[@]}"
}

submit_one "qw8_nocg_m20_direct" "examples/configs/recipes/llm/performance/grpo-qwen3-8b-1n4g-nocg.yaml"
submit_one "qw8_attn_w3_m20_direct" "examples/configs/recipes/llm/performance/grpo-qwen3-8b-1n4g-cg-attn-w3.yaml"
submit_one "qw8_attn_w6_m20_direct" "examples/configs/recipes/llm/performance/grpo-qwen3-8b-1n4g-cg-attn-w6.yaml"
submit_one "qw8_mlp_w3_m20_direct" "examples/configs/recipes/llm/performance/grpo-qwen3-8b-1n4g-cg-attn-mlp-w3.yaml" "policy.megatron_cfg.cuda_graph_scope=mlp"
submit_one "qw8_mlp_w6_m20_direct" "examples/configs/recipes/llm/performance/grpo-qwen3-8b-1n4g-cg-attn-mlp-w6.yaml" "policy.megatron_cfg.cuda_graph_scope=mlp"
submit_one "qw8_attnmlp_w3_m20_direct" "examples/configs/recipes/llm/performance/grpo-qwen3-8b-1n4g-cg-attn-mlp-w3.yaml"
submit_one "qw8_attnmlp_w6_m20_direct" "examples/configs/recipes/llm/performance/grpo-qwen3-8b-1n4g-cg-attn-mlp-w6.yaml"
