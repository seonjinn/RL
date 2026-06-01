#!/bin/bash
# Qwen/Qwen3-235B-A22B performance script with SpecDec enabled.
# This is intentionally the original exp_qwen235b_a22b.sh flow plus only the
# vLLM speculative decoding overrides.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${ALLOW_DEPRECATED_QWEN235B_SPECDEC_SCRIPT:-false}" != "true" && "${ALLOW_DEPRECATED_QWEN235B_SPECDEC_SCRIPT:-false}" != "True" ]]; then
    echo "ERROR: Qwen235B_GB200_SpecDec.sh is deprecated because it defaults to a legacy 50K drafter and patches NeMo-RL in place." >&2
    echo "Use Qwen235B_GB200_Main_SpecDec.sh with explicit DRAFT_MODEL and DRAFT_MODEL_PROVENANCE instead." >&2
    echo "For a legacy diagnostic only, set ALLOW_DEPRECATED_QWEN235B_SPECDEC_SCRIPT=true." >&2
    exit 2
fi

# Source cluster configuration (auto-detect H100/GB200)
source "${SCRIPT_DIR}/cluster_config.sh"
setup_cluster_config "batch"
export_cluster_config

echo "============================================"
echo "Launching Qwen3-235B-A22B GRPO Training + SpecDec"
echo "Cluster: ${CLUSTER_TYPE}, GPUs/Node: ${GPUS_PER_NODE}"
echo "============================================"

# Load model-specific config from YAML
MODEL_CONFIG_CLUSTER="${CLUSTER_TYPE,,}"
if [[ "${MODEL_CONFIG_CLUSTER}" == "gb200_oci" ]]; then
    MODEL_CONFIG_CLUSTER="gb200"
fi
eval $(python3 "${SCRIPT_DIR}/get_model_config.py" qwen235b ${MODEL_CONFIG_CLUSTER})

# Calculate number of nodes
NUM_NODES=$((NUM_GPUS / GPUS_PER_NODE))

DRAFT_MODEL="${DRAFT_MODEL:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/speculators/eagle3_openmath_reasoning_cot_50k/checkpoints_train_50k_layers93_mlen8193_fsdpcache_compile/0}"
SPECDEC_METHOD="${SPECDEC_METHOD:-eagle3}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
DRAFT_TP="${DRAFT_TP:-1}"
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"
VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE="${VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE:-4}"

if [[ ! -s "${DRAFT_MODEL}/config.json" ]]; then
    echo "ERROR: DRAFT_MODEL is not a valid HF checkpoint: ${DRAFT_MODEL}" >&2
    exit 2
fi

# NeMo-RL normally sets vLLM load_format=dummy for training generation workers.
# With an external draft model, dummy load would leave the drafter random.
python3 - <<'PY'
from pathlib import Path

p = Path("nemo_rl/models/generation/__init__.py")
text = p.read_text(encoding="utf-8")
marker = "Speculative decoding is enabled. Setting vllm_cfg['load_format'] to 'auto'."
if marker not in text:
    needle = '        config["vllm_cfg"]["load_format"] = "auto" if is_eval else "dummy"\n\n        # Respect'
    replacement = '''        config["vllm_cfg"]["load_format"] = "auto" if is_eval else "dummy"
        is_spec = "speculative_config" in config.get("vllm_kwargs", {})
        if is_spec:
            warnings.warn(
                "Speculative decoding is enabled. Setting vllm_cfg['load_format'] to 'auto'. "
                "This may result in slower startup times as full model weights are loaded."
            )
            config["vllm_cfg"]["load_format"] = "auto"

        # Respect'''
    if needle not in text:
        raise SystemExit("Could not patch nemo_rl/models/generation/__init__.py")
    p.write_text(text.replace(needle, replacement), encoding="utf-8")
    print("Patched NeMo-RL generation config for SpecDec load_format=auto")
else:
    print("NeMo-RL generation config already handles SpecDec load_format=auto")
PY

# Print configuration
echo "[INFO] Model: ${MODEL_NAME}"
echo "[INFO] Nodes: ${NUM_NODES}, Total GPUs: ${NUM_GPUS}"
echo "[INFO] Generation: TP=${G_TP}, PP=${G_PP}"
echo "[INFO] Training: TP=${T_TP}, CP=${T_CP}, EP=${T_EP}, PP=${T_PP}"
echo "[INFO] Batch: Rollout=${ROLLOUT_GBS}, Train=${TRAIN_GBS}"
echo "[INFO] SpecDec: method=${SPECDEC_METHOD}, draft=${DRAFT_MODEL}, k=${NUM_SPECULATIVE_TOKENS}, draft_tp=${DRAFT_TP}"

account=coreai_dlalgo_nemorl
WANDB_PROJECT="sync-grpo-${CLUSTER_TYPE,,}-benchmark"

# Segment size (16 for large jobs)
SEGMENT=16

DRIVER_UV_PROJECT_ENVIRONMENT="${DRIVER_UV_PROJECT_ENVIRONMENT:-${SCRIPT_DIR}/.driver_venvs/qwen235b_math_specdec}"

COMMAND="NRL_FORCE_REBUILD_VENVS=true \
UV_PROJECT_ENVIRONMENT=${DRIVER_UV_PROJECT_ENVIRONMENT} \
uv run ./examples/run_grpo_math.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.expert_parallel_size=${G_EP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.generation.vllm_cfg.enforce_eager=false \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
policy.megatron_cfg.sequence_parallel=${T_SP} \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.num_prompts_per_step=${NUM_PROMPTS} \
grpo.num_generations_per_prompt=${NUM_GENERATIONS} \
policy.sequence_packing.enabled=True \
policy.train_global_batch_size=${TRAIN_GBS} \
grpo.max_num_steps=20 \
++policy.generation.vllm_kwargs.speculative_config.method=${SPECDEC_METHOD} \
++policy.generation.vllm_kwargs.speculative_config.model=${DRAFT_MODEL} \
++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=${NUM_SPECULATIVE_TOKENS} \
++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=${DRAFT_TP} \
++policy.generation.vllm_kwargs.attention_backend=${VLLM_ATTENTION_BACKEND} \
++policy.generation.vllm_kwargs.max_cudagraph_capture_size=${VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE} \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='Qwen235B_A22B_SpecDec_N${NUM_NODES}xG${GPUS_PER_NODE}_Ttp${T_TP}pp${T_PP}ep${T_EP}cp${T_CP}_Gtp${G_TP}pp${G_PP}ep${G_EP}_k${NUM_SPECULATIVE_TOKENS}'" \
CONTAINER=$CONTAINER \
HF_HOME=$HF_HOME \
HF_DATASETS_CACHE=$HF_DATASETS_CACHE \
WANDB_API_KEY=$WANDB_API_KEY \
MOUNTS="$MOUNTS" \
sbatch \
    --nodes=${NUM_NODES} \
    --account=${account} \
    --job-name=qwen235b-specdec-N${NUM_NODES}xG${GPUS_PER_NODE}-T.tp${T_TP}.pp${T_PP}.ep${T_EP}-G.tp${G_TP}.pp${G_PP} \
    --partition=${PARTITION} \
    --time=04:00:00 \
    ${GRES_FLAG} \
    --segment ${SEGMENT} \
    ray.sub
