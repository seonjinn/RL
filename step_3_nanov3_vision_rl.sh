#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMORL="${NEMORL:-${SCRIPT_DIR}}"

OMNI_SHARED_ROOT="${OMNI_SHARED_ROOT:-/lustre/fs1/portfolios/coreai/users/aroshanghias}"
USER_ROOT="${USER_ROOT:-/lustre/fs1/portfolios/coreai/users/${USER:-aroshanghias}}"
DATASET_ROOT="${DATASET_ROOT:-${OMNI_SHARED_ROOT}/data/mmpr_miniscule/processed}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${OMNI_SHARED_ROOT}/checkpoints}"
CONTAINER_ROOT="${CONTAINER_ROOT:-${OMNI_SHARED_ROOT}/containers}"
CONFIG_PATH="${CONFIG_PATH:-examples/configs/nanov3_vision_rl_truncated.yaml}"

JOB_NAME_BASE="${JOB_NAME_BASE:-step-3-nemotron-vl-super-debug}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S-%3N)}"
JOB_NAME="${JOB_NAME:-${JOB_NAME_BASE}-${RUN_ID}}"
SEED="${SEED:-42}"
NUM_NODES="${NUM_NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-${CP_SIZE:-1}}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-1}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
TRAIN_GLOBAL_BATCH_SIZE="${TRAIN_GLOBAL_BATCH_SIZE:-1}"
NUM_PROMPTS_PER_STEP="${NUM_PROMPTS_PER_STEP:-1}"
NUM_GENERATIONS_PER_PROMPT="${NUM_GENERATIONS_PER_PROMPT:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-1}"
MODEL_NAME="${MODEL_NAME:-${CHECKPOINT_ROOT}/mpo-nanov3omni-mmpr-nanov2-filtered-conv3d-truncated}"
PRODUCTION_MODEL_NAME="${PRODUCTION_MODEL_NAME:-${CHECKPOINT_ROOT}/mpo-nanov3omni-mmpr-nanov2-filtered-conv3d-0303/step_400}"
WANDB_PROJECT="${WANDB_PROJECT:-grpo-nanov3vl}"
RESULTS_ROOT="${RESULTS_ROOT:-${NEMORL}/results}"
RESULTS_DIR="${RESULTS_ROOT}/${JOB_NAME}"

FIXTURE_SAMPLE_ID="${FIXTURE_SAMPLE_ID:-10189}"
FIXTURE_IMAGE="${FIXTURE_IMAGE:-${DATASET_ROOT}/MMPR-Tiny/images/10189_0.png}"
if [[ -z "${FIXTURE_PROMPT:-}" ]]; then
  FIXTURE_PROMPT=$'<image>\nWhile hanging Christmas lights for neighbors, Bella counted the number of broken lights on each string. How many strings had exactly 16 broken lights?\nPlease answer the question and put the final answer within \\boxed{}.'
fi

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-coreai_dlalgo_nemorl}"
SBATCH_PARTITION="${SBATCH_PARTITION:-${PARTITION:-batch}}"
SBATCH_DEPENDENCY="${SBATCH_DEPENDENCY:-singleton}"
SBATCH_TIME="${SBATCH_TIME:-4:00:00}"

export CONTAINER="${CONTAINER:-${CONTAINER_ROOT}/nemo-rl-nano-v3-vl-b65b6cde.sqsh}"
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-true}"
export CACHE_ROOT="${CACHE_ROOT:-${USER_ROOT}/.cache}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-${HF_HOME}/nemo_rl}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
export TMPDIR="${TMPDIR:-/tmp/nrl-${USER:-u}}"

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NVTE_FWD_LAYERNORM_SM_MARGIN="${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}"
export NVTE_BWD_LAYERNORM_SM_MARGIN="${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}"
export NEMO_RL_LOG_GPU_MEMORY="${NEMO_RL_LOG_GPU_MEMORY:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NRL_IGNORE_VERSION_MISMATCH="${NRL_IGNORE_VERSION_MISMATCH:-true}"
export NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR:-${NEMORL}/../tmp}"

export NRL_NEMOTRON_VL_DEBUG="${NRL_NEMOTRON_VL_DEBUG:-1}"
export NRL_NEMOTRON_VL_DEBUG_DIR="${NRL_NEMOTRON_VL_DEBUG_DIR:-/tmp/nrl_nemotron_vl_debug/super}"
export NRL_NEMOTRON_VL_RUN_LABEL="${NRL_NEMOTRON_VL_RUN_LABEL:-super}"
export NRL_VLLM_USE_V1="${NRL_VLLM_USE_V1:-0}"
export NRL_NEMOTRON_VL_FIXTURE_SAMPLE_ID="${NRL_NEMOTRON_VL_FIXTURE_SAMPLE_ID:-${FIXTURE_SAMPLE_ID}}"
export NRL_NEMOTRON_VL_FIXTURE_IMAGE="${NRL_NEMOTRON_VL_FIXTURE_IMAGE:-${FIXTURE_IMAGE}}"
export NRL_NEMOTRON_VL_FIXTURE_PROMPT="${NRL_NEMOTRON_VL_FIXTURE_PROMPT:-${FIXTURE_PROMPT}}"
export NRL_NEMOTRON_VL_PRODUCTION_MODEL="${NRL_NEMOTRON_VL_PRODUCTION_MODEL:-${PRODUCTION_MODEL_NAME}}"

if [[ ! -f "${NEMORL}/ray.sub" ]]; then
  echo "ray.sub not found under NEMORL=${NEMORL}" >&2
  exit 1
fi

if [[ ! -f "${FIXTURE_IMAGE}" ]]; then
  echo "Fixture image not found: ${FIXTURE_IMAGE}" >&2
  exit 1
fi

if [[ "${CONFIG_PATH}" = /* ]]; then
  CONFIG_ABS_PATH="${CONFIG_PATH}"
else
  CONFIG_ABS_PATH="${NEMORL}/${CONFIG_PATH}"
fi

if [[ ! -f "${CONFIG_ABS_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

echo "Using Super parity launcher"
echo "  branch_role=implementation"
echo "  model_name=${MODEL_NAME}"
echo "  production_model_name=${PRODUCTION_MODEL_NAME}"
echo "  dataset_root=${DATASET_ROOT}"
echo "  fixture_sample_id=${FIXTURE_SAMPLE_ID}"
echo "  fixture_image=${FIXTURE_IMAGE}"
echo "  seed=${SEED}"
echo "  debug_dir=${NRL_NEMOTRON_VL_DEBUG_DIR}"
echo "  note=Gate-0 scaffold; expected to stay structurally parallel to Omni until the Nemotron recipe and dataset wiring land in Super."

EXTRA_OVERRIDES="\
policy.megatron_cfg.context_parallel_size=${CONTEXT_PARALLEL_SIZE} \
policy.megatron_cfg.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE} \
policy.megatron_cfg.expert_model_parallel_size=${EXPERT_MODEL_PARALLEL_SIZE} \
policy.generation.vllm_cfg.tensor_parallel_size=${VLLM_TENSOR_PARALLEL_SIZE} \
policy.generation.vllm_cfg.async_engine=false \
policy.model_name='${MODEL_NAME}' \
policy.train_global_batch_size=${TRAIN_GLOBAL_BATCH_SIZE} \
grpo.num_prompts_per_step=${NUM_PROMPTS_PER_STEP} \
grpo.num_generations_per_prompt=${NUM_GENERATIONS_PER_PROMPT} \
grpo.seed=${SEED} \
grpo.max_num_steps=${MAX_NUM_STEPS:-1} \
grpo.deduplicate_multimodal_data=${DEDUPLICATE_MULTIMODAL_DATA:-false} \
data.train.cache_dir='${DATASET_ROOT}' \
policy.generation.max_new_tokens=${MAX_NEW_TOKENS} \
policy.generation.temperature=${TEMPERATURE} \
policy.generation.top_p=${TOP_P} \
policy.generation.top_k=${TOP_K} \
checkpointing.checkpoint_dir='${RESULTS_DIR}' \
logger.log_dir='${RESULTS_DIR}' \
logger.wandb_enabled=false \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${JOB_NAME}'"

export COMMAND="\
mkdir -p ${HF_HOME} ${HF_MODULES_CACHE} ${NRL_MEGATRON_CHECKPOINT_DIR} ${TRITON_CACHE_DIR} ${TMPDIR} ${RESULTS_DIR} && \
uv run examples/run_vlm_grpo.py --config ${CONFIG_PATH} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
${EXTRA_OVERRIDES}"

cd "${NEMORL}"

MOUNTS="${MOUNTS:-/lustre:/lustre}" \
sbatch \
    --nodes=${NUM_NODES} \
    --account=${SBATCH_ACCOUNT} \
    --job-name=${JOB_NAME} \
    --partition=${SBATCH_PARTITION} \
    --dependency=${SBATCH_DEPENDENCY} \
    --time=${SBATCH_TIME} \
    --gres=gpu:${GPUS_PER_NODE} \
    ray.sub
