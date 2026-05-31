#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NUM_NODES="${NUM_NODES:-32}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
PARTITION="${PARTITION:-batch}"
ACCOUNT="${ACCOUNT:-coreai_dlalgo_nemorl}"
GRES_FLAG="${GRES_FLAG:---gres=gpu:4}"
SEGMENT="${SEGMENT:-16}"
CPUS_PER_WORKER="${CPUS_PER_WORKER:-$((GPUS_PER_NODE * 16))}"
SBATCH_RESOURCE_ARGS="${SBATCH_RESOURCE_ARGS:---ntasks-per-node=1 --cpus-per-task=${CPUS_PER_WORKER} --mem=0}"
SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:-}"
JOB_TAG="${JOB_TAG:-main-specdec-bgate}"

CONFIG_FILE="${CONFIG_FILE:-examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n8g.yaml}"
CONTAINER="${CONTAINER:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl-main-20260526/nemo_rl_nightly.sqsh}"
HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/cache}"
MOUNTS="${MOUNTS:-/lustre:/lustre}"
NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-${SCRIPT_DIR}/nrl_megatron_ckpts_20260526}"
WANDB_PROJECT="${WANDB_PROJECT:-sync-grpo-gb200_oci-benchmark}"
UV_PYTHON="${UV_PYTHON:-3.12.13}"
RAY_VERSION="${RAY_VERSION:-2.49.2}"
DRIVER_UV_PROJECT_ENVIRONMENT="${DRIVER_UV_PROJECT_ENVIRONMENT:-${SCRIPT_DIR}/.driver_venvs/qwen235b_main_specdec_py312}"
NUM_PROMPTS="${NUM_PROMPTS:-16}"
NUM_GENERATIONS="${NUM_GENERATIONS:-32}"
TRAIN_GLOBAL_BATCH_SIZE="${TRAIN_GLOBAL_BATCH_SIZE:-$((NUM_PROMPTS * NUM_GENERATIONS))}"
MAX_STEPS="${MAX_STEPS:-20}"

DRAFT_MODEL="${DRAFT_MODEL:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/speculators/eagle3_openmath50k_dapo16k_continued/checkpoints_from50k_dapo16k_e3_lr5e5_layers93_mlen8193_cachefix_r2/2}"
SPECDEC_METHOD="${SPECDEC_METHOD:-eagle3}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
DRAFT_TP="${DRAFT_TP:-1}"
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
SPECDEC_BATCH_SIZE_GATE_THRESHOLD="${SPECDEC_BATCH_SIZE_GATE_THRESHOLD:-4}"
RAY_raylet_start_wait_time_s="${RAY_raylet_start_wait_time_s:-180}"
RAY_LOG_SYNC_FREQUENCY="${RAY_LOG_SYNC_FREQUENCY:-30}"

mkdir -p "${NRL_MEGATRON_CHECKPOINT_DIR}"

if [[ ! -s "${CONTAINER}" ]]; then
  echo "ERROR: container not found: ${CONTAINER}" >&2
  exit 2
fi
if [[ -d "${DRAFT_MODEL}" || "${DRAFT_MODEL}" == /* ]]; then
  if [[ ! -s "${DRAFT_MODEL}/config.json" ]]; then
    echo "ERROR: DRAFT_MODEL is not a valid local HF checkpoint: ${DRAFT_MODEL}" >&2
    exit 2
  fi
else
  echo "Using Hugging Face draft model id: ${DRAFT_MODEL}"
fi

COMMAND="NRL_FORCE_REBUILD_VENVS=true \
UV_PYTHON=${UV_PYTHON} \
UV_PROJECT_ENVIRONMENT=${DRIVER_UV_PROJECT_ENVIRONMENT} \
PYTHONPATH=${SCRIPT_DIR}:${PYTHONPATH:-} \
NRL_MEGATRON_CHECKPOINT_DIR=${NRL_MEGATRON_CHECKPOINT_DIR} \
VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND} \
VLLM_SPECDEC_BATCH_SIZE_GATE_THRESHOLD=${SPECDEC_BATCH_SIZE_GATE_THRESHOLD} \
uv run --python ${UV_PYTHON} --locked --extra mcore --directory ${SCRIPT_DIR} python ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=16 \
policy.generation.vllm_cfg.expert_parallel_size=1 \
policy.generation.vllm_cfg.pipeline_parallel_size=1 \
policy.generation.vllm_cfg.enforce_eager=false \
policy.megatron_cfg.tensor_model_parallel_size=2 \
policy.megatron_cfg.expert_model_parallel_size=16 \
policy.megatron_cfg.pipeline_model_parallel_size=8 \
policy.megatron_cfg.context_parallel_size=2 \
policy.megatron_cfg.sequence_parallel=true \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.num_prompts_per_step=${NUM_PROMPTS} \
grpo.num_generations_per_prompt=${NUM_GENERATIONS} \
policy.sequence_packing.enabled=true \
policy.train_global_batch_size=${TRAIN_GLOBAL_BATCH_SIZE} \
grpo.max_num_steps=${MAX_STEPS} \
++policy.generation.vllm_kwargs.speculative_config.method=${SPECDEC_METHOD} \
++policy.generation.vllm_kwargs.speculative_config.model=${DRAFT_MODEL} \
++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=${NUM_SPECULATIVE_TOKENS} \
++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=${DRAFT_TP} \
logger.wandb_enabled=true \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='Qwen235B_A22B_Main_N${NUM_NODES}xG${GPUS_PER_NODE}_specdec_k${NUM_SPECULATIVE_TOKENS}_bgate${SPECDEC_BATCH_SIZE_GATE_THRESHOLD}_${MAX_STEPS}step'"

CONTAINER="${CONTAINER}" \
HF_HOME="${HF_HOME}" \
HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
WANDB_API_KEY="${WANDB_API_KEY:-}" \
NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR}" \
UV_PYTHON="${UV_PYTHON}" \
RAY_VERSION="${RAY_VERSION}" \
DRIVER_UV_PROJECT_ENVIRONMENT="${DRIVER_UV_PROJECT_ENVIRONMENT}" \
GPUS_PER_NODE="${GPUS_PER_NODE}" \
MOUNTS="${MOUNTS}" \
RAY_raylet_start_wait_time_s="${RAY_raylet_start_wait_time_s}" \
RAY_LOG_SYNC_FREQUENCY="${RAY_LOG_SYNC_FREQUENCY}" \
COMMAND="${COMMAND}" \
sbatch \
  --nodes="${NUM_NODES}" \
  --account="${ACCOUNT}" \
  --job-name="qwen235b-${JOB_TAG}-N${NUM_NODES}xG${GPUS_PER_NODE}" \
  --partition="${PARTITION}" \
  --time=04:00:00 \
  ${GRES_FLAG} \
  ${SBATCH_RESOURCE_ARGS} \
  ${SBATCH_EXTRA_ARGS} \
  --segment "${SEGMENT}" \
  "${SCRIPT_DIR}/ray.sub"
