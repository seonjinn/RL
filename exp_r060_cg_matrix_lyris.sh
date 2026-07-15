#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

RAY_SUB="${SCRIPT_DIR}/ray.sub"
CONTAINER="/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly.sqsh"
ACCOUNT="coreai_dlalgo_llm"
PARTITION="gb200"
HF_HOME="/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home"
HF_DATASETS_CACHE="${HF_HOME}/cache"
MOUNTS="/lustre:/lustre"
LOG_BASE="${SCRIPT_DIR}/experiments/r060_cg_matrix_lyris_$(date +%Y%m%d)"

mkdir -p "${LOG_BASE}"

submit_one() {
    local name="$1"
    local num_nodes="$2"
    local gpus_per_node="$3"
    local config_file="$4"
    local extra="$5"

    local command
    command="NRL_FORCE_REBUILD_VENVS=true CG_COUNT_LOG=1 uv run ./examples/run_grpo.py \
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

    echo "[SUBMIT] ${name}"
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
        --segment "${num_nodes}" \
        "${RAY_SUB}"
}

ll8_common="policy.model_name=meta-llama/Llama-3.1-8B-Instruct policy.tokenizer.name=meta-llama/Llama-3.1-8B-Instruct policy.megatron_cfg.converter_type=LlamaForCausalLM policy.megatron_cfg.pipeline_model_parallel_size=1"
qw8_common="policy.model_name=Qwen/Qwen3-8B-Base policy.tokenizer.name=Qwen/Qwen3-8B-Base policy.megatron_cfg.converter_type=Qwen3ForCausalLM policy.megatron_cfg.pipeline_model_parallel_size=1"
q30_common=""

cg_ll8="policy.megatron_cfg.cuda_graph_impl=transformer_engine policy.megatron_cfg.cuda_graph_warmup_steps=3 policy.megatron_cfg.cuda_graph_packed_seq=true policy.megatron_cfg.cuda_graph_buckets=[4096]"
cg_qw8="policy.megatron_cfg.cuda_graph_impl=transformer_engine policy.megatron_cfg.cuda_graph_warmup_steps=3 policy.megatron_cfg.cuda_graph_packed_seq=false policy.megatron_cfg.cuda_graph_buckets=[4096]"
cg_q30="policy.megatron_cfg.cuda_graph_impl=transformer_engine policy.megatron_cfg.cuda_graph_warmup_steps=3 policy.megatron_cfg.cuda_graph_packed_seq=false policy.megatron_cfg.cuda_graph_buckets=[4096]"

for seqpack in true false; do
    if [ "${seqpack}" = "true" ]; then
        sp_tag="sp1"
        ll8_sp=""
        qw8_sp=""
        q30_sp="policy.sequence_packing.enabled=true"
    else
        sp_tag="sp0"
        ll8_sp="policy.sequence_packing.enabled=false"
        qw8_sp="policy.sequence_packing.enabled=false"
        q30_sp=""
    fi

    submit_one "r060-ll8-${sp_tag}-nocg" 1 4 "examples/configs/grpo_math_8B_megatron.yaml" "${ll8_common} ${ll8_sp}"
    submit_one "r060-ll8-${sp_tag}-attn" 1 4 "examples/configs/grpo_math_8B_megatron.yaml" "${ll8_common} ${ll8_sp} ${cg_ll8} policy.megatron_cfg.cuda_graph_scope=attn"
    submit_one "r060-ll8-${sp_tag}-mlp" 1 4 "examples/configs/grpo_math_8B_megatron.yaml" "${ll8_common} ${ll8_sp} ${cg_ll8} policy.megatron_cfg.cuda_graph_scope=mlp"
    submit_one "r060-ll8-${sp_tag}-attnmlp" 1 4 "examples/configs/grpo_math_8B_megatron.yaml" "${ll8_common} ${ll8_sp} ${cg_ll8} policy.megatron_cfg.cuda_graph_scope=[attn,mlp]"

    submit_one "r060-qw8-${sp_tag}-nocg" 1 4 "examples/configs/grpo_math_8B_megatron.yaml" "${qw8_common} ${qw8_sp}"
    submit_one "r060-qw8-${sp_tag}-attn" 1 4 "examples/configs/grpo_math_8B_megatron.yaml" "${qw8_common} ${qw8_sp} ${cg_qw8} policy.megatron_cfg.cuda_graph_scope=attn"
    submit_one "r060-qw8-${sp_tag}-mlp" 1 4 "examples/configs/grpo_math_8B_megatron.yaml" "${qw8_common} ${qw8_sp} ${cg_qw8} policy.megatron_cfg.cuda_graph_scope=mlp"
    submit_one "r060-qw8-${sp_tag}-attnmlp" 1 4 "examples/configs/grpo_math_8B_megatron.yaml" "${qw8_common} ${qw8_sp} ${cg_qw8} policy.megatron_cfg.cuda_graph_scope=[attn,mlp]"

    submit_one "r060-q30-${sp_tag}-nocg" 4 4 "examples/configs/grpo_math_qwen30ba3b_megatron.yaml" "${q30_common} ${q30_sp}"
    submit_one "r060-q30-${sp_tag}-attn" 4 4 "examples/configs/grpo_math_qwen30ba3b_megatron.yaml" "${q30_common} ${q30_sp} ${cg_q30} policy.megatron_cfg.cuda_graph_scope=attn"
    submit_one "r060-q30-${sp_tag}-mlp" 4 4 "examples/configs/grpo_math_qwen30ba3b_megatron.yaml" "${q30_common} ${q30_sp} ${cg_q30} policy.megatron_cfg.cuda_graph_scope=mlp"
    submit_one "r060-q30-${sp_tag}-moerouter" 4 4 "examples/configs/grpo_math_qwen30ba3b_megatron.yaml" "${q30_common} ${q30_sp} ${cg_q30} policy.megatron_cfg.cuda_graph_scope=moe_router"
    submit_one "r060-q30-${sp_tag}-attnmlp" 4 4 "examples/configs/grpo_math_qwen30ba3b_megatron.yaml" "${q30_common} ${q30_sp} ${cg_q30} policy.megatron_cfg.cuda_graph_scope=[attn,mlp]"
    submit_one "r060-q30-${sp_tag}-attnmoerouter" 4 4 "examples/configs/grpo_math_qwen30ba3b_megatron.yaml" "${q30_common} ${q30_sp} ${cg_q30} policy.megatron_cfg.cuda_graph_scope=[attn,moe_router]"
    submit_one "r060-q30-${sp_tag}-mlpmoerouter" 4 4 "examples/configs/grpo_math_qwen30ba3b_megatron.yaml" "${q30_common} ${q30_sp} ${cg_q30} policy.megatron_cfg.cuda_graph_scope=[mlp,moe_router]"
    submit_one "r060-q30-${sp_tag}-attnmlpmoerouter" 4 4 "examples/configs/grpo_math_qwen30ba3b_megatron.yaml" "${q30_common} ${q30_sp} ${cg_q30} policy.megatron_cfg.cuda_graph_scope=[attn,mlp,moe_router]"
done
