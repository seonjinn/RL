#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

REFERENCE_DIR="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl"
RAY_SUB="${REFERENCE_DIR}/ray.sub"
LOG_BASE="${SCRIPT_DIR}/experiments/qwen30_oci_script_w3_$(date +%Y%m%d)"

source "${REFERENCE_DIR}/cluster_config.sh"
setup_cluster_config "batch"
export_cluster_config

CONTAINER="${REFERENCE_DIR}/nemo_rl_nightly.sqsh"
export CONTAINER

eval "$(python3 "${REFERENCE_DIR}/get_model_config.py" qwen30b gb200 gb200_4n4g)"

NUM_NODES=$((NUM_GPUS / GPUS_PER_NODE))
ACCOUNT="coreai_dlalgo_nemorl"

echo "============================================"
echo "Launching Qwen3-30B-A3B CG Sweep (warmup=3)"
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

    local command
    command="NRL_FORCE_REBUILD_VENVS=true NRL_IGNORE_VERSION_MISMATCH=1 CG_COUNT_LOG=1 uv run ./examples/run_grpo.py \
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
policy.megatron_cfg.cuda_graph_warmup_steps=3 \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
logger.wandb_enabled=true \
logger.wandb.project=nemo-rl-cudagraph \
logger.tensorboard_enabled=false \
grpo.num_prompts_per_step=${NUM_PROMPTS} \
grpo.num_generations_per_prompt=${NUM_GENERATIONS} \
policy.sequence_packing.enabled=True \
policy.train_global_batch_size=${TRAIN_GBS} \
grpo.max_num_steps=20"

    echo "[SUBMIT] ${name} -> ${config_file}"
    COMMAND="${command}" \
    CONTAINER="${CONTAINER}" \
    HF_HOME="${HF_HOME}" \
    HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
    MOUNTS="${MOUNTS}" \
    BASE_LOG_DIR="${LOG_BASE}" \
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

submit_one "q30_attn_w3_m20_script" "examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-cg-attn.yaml"
submit_one "q30_moerouter_w3_m20_script" "examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-cg-moe-router.yaml"
submit_one "q30_attnmoerouter_w3_m20_script" "examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-cg-attn-moe-router.yaml"
