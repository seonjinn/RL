#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
cd "${SCRIPT_DIR}"

REFERENCE_DIR="${SCRIPT_DIR}"
if [[ ! -f "${REFERENCE_DIR}/cluster_config.sh" ]]; then
    REFERENCE_DIR="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl"
fi
# Use this repo's ray.sub (carries the GRES-suffix and Ray-CLI fixes).
RAY_SUB_DEFAULT="${SCRIPT_DIR}/ray.sub"
RAY_SUB="${RAY_SUB_OVERRIDE:-${RAY_SUB_DEFAULT}}"
if [[ ! -f "${RAY_SUB}" ]]; then
    RAY_SUB="${REFERENCE_DIR}/ray.sub"
fi
CONTAINER_DEFAULT="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/nemo_rl_nightly_20260624.sqsh"
UV_CACHE_DIR_OVERRIDE_DEFAULT="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/.cache/uv"
PROJECT_ROOT="${PROJECT_ROOT_OVERRIDE:-${SCRIPT_DIR}}"
LOG_ROOT="${LOG_ROOT_OVERRIDE:-${PROJECT_ROOT}}"
LOG_BASE="${LOG_ROOT}/experiments/qwen30_oci_script_$(date +%Y%m%d)"
JOB_FILTER="${JOB_FILTER:-}"
MAX_STEPS="${MAX_STEPS:-20}"
# Prebuilt container venvs by default: branch uv.lock does not resolve on
# py313 containers (nvidia-resiliency-ext lacks cp313 wheels).
REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-false}"

source "${REFERENCE_DIR}/cluster_config.sh"
setup_cluster_config "batch"
export_cluster_config

# Reuse the known-working OCI nightly and helper config path.
CONTAINER="${CONTAINER_OVERRIDE:-${CONTAINER_DEFAULT}}"
export CONTAINER

eval "$(python3 "${REFERENCE_DIR}/get_model_config.py" qwen30b gb200 gb200_4n4g)"

NUM_NODES=$((NUM_GPUS / GPUS_PER_NODE))
ACCOUNT="coreai_dlalgo_nemorl"

echo "============================================"
echo "Launching Qwen3-30B-A3B CG Sweep"
echo "Workspace: ${SCRIPT_DIR}"
echo "Reference helper dir: ${REFERENCE_DIR}"
echo "Partition: ${PARTITION}"
echo "Nodes: ${NUM_NODES}, GPUs/Node: ${GPUS_PER_NODE}"
echo "Container: ${CONTAINER}"
echo "============================================"

mkdir -p "${LOG_BASE}"

submit_one() {
    local name="$1"
    local config_file="$2"
    local job_venv_dir="${NEMO_RL_VENV_DIR_OVERRIDE:-${PROJECT_ROOT}/job_venvs/${name}}"

    if [[ -n "${JOB_FILTER}" && "${name}" != *"${JOB_FILTER}"* ]]; then
        return
    fi

    # NEMO_RL_VENV_DIR must be exported INSIDE the command: enroot drops
    # submit-time env vars in favor of the image's baked ENV (/opt/ray_venvs,
    # whose prebaked venvs are broken symlink farms in the 20260711 nightly).
    # A lustre venv dir makes prefetch build complete venvs from our uv.lock.
    local command
    command="cd ${PROJECT_ROOT} && export PYTHONPATH=${PROJECT_ROOT}/3rdparty/Megatron-LM-workspace/Megatron-LM:${PROJECT_ROOT}:\${PYTHONPATH:-} UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv NEMO_RL_VENV_DIR=${job_venv_dir} && uv pip install --reinstall --no-deps transformers==5.5.0 >/dev/null 2>&1 && uv run --no-sync python -m nemo_rl.utils.prefetch_venvs vllm_worker megatron_policy_worker && uv run --no-sync ./examples/run_grpo.py \
--config ${config_file} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.expert_parallel_size=${G_EP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
policy.megatron_cfg.sequence_parallel=${T_SP} \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
logger.wandb_enabled=true \
logger.wandb.project=nemo-rl-cudagraph \
logger.tensorboard_enabled=false \
grpo.num_prompts_per_step=${NUM_PROMPTS_OVERRIDE:-${NUM_PROMPTS}} \
grpo.num_generations_per_prompt=${NUM_GENERATIONS} \
policy.sequence_packing.enabled=True \
policy.train_global_batch_size=${TRAIN_GBS} \
grpo.max_num_steps=${MAX_STEPS}"

    echo "[SUBMIT] ${name} -> ${config_file}"
    COMMAND="${command}" \
    CONTAINER="${CONTAINER}" \
    HF_HOME="${HF_HOME}" \
    HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
    MOUNTS="${MOUNTS}" \
    BASE_LOG_DIR="${LOG_BASE}" \
    NEMO_RL_VENV_DIR="${job_venv_dir}" \
    NRL_REPAIR_TRANSFORMERS="${NRL_REPAIR_TRANSFORMERS:-5.5.0}" \
    NRL_FORCE_REBUILD_VENVS="${REBUILD_VENVS}" \
    UV_LINK_MODE=copy \
    NRL_SKIP_VENV_SYNC="${NRL_SKIP_VENV_SYNC:-1}" \
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
        --segment "${NUM_NODES}" \
        "${RAY_SUB}"
}

submit_one "q30_nocg_m20_script" "examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-nocg.yaml"
submit_one "q30_attn_w6_m20_script" "examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-cg-attn.yaml"
submit_one "q30_moerouter_w6_m20_script" "examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-cg-moe-router.yaml"
submit_one "q30_attnmoerouter_w6_m20_script" "examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-cg-attn-moe-router.yaml"
