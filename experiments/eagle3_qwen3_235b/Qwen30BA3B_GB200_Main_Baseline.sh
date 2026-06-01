#!/bin/bash
set -euo pipefail

SCRIPT_PATH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${NEMO_RL_DIR:-}" ]]; then
  SCRIPT_DIR="${NEMO_RL_DIR}"
elif [[ -f "${SCRIPT_PATH_DIR}/examples/run_grpo.py" ]]; then
  SCRIPT_DIR="${SCRIPT_PATH_DIR}"
else
  SCRIPT_DIR="$(cd "${SCRIPT_PATH_DIR}/../.." && pwd)"
fi
if [[ ! -f "${SCRIPT_DIR}/examples/run_grpo.py" ]]; then
  echo "ERROR: could not locate NeMo-RL repo root from ${SCRIPT_PATH_DIR}; set NEMO_RL_DIR" >&2
  exit 2
fi

NUM_NODES="${NUM_NODES:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
PARTITION="${PARTITION:-batch}"
ACCOUNT="${ACCOUNT:-coreai_dlalgo_nemorl}"
GRES_FLAG="${GRES_FLAG:---gres=gpu:4}"
SEGMENT="${SEGMENT:-${NUM_NODES}}"
CPUS_PER_WORKER="${CPUS_PER_WORKER:-$((GPUS_PER_NODE * 16))}"
SBATCH_RESOURCE_ARGS="${SBATCH_RESOURCE_ARGS:---ntasks-per-node=1 --cpus-per-task=${CPUS_PER_WORKER} --mem=0}"
SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:-}"
JOB_TAG="${JOB_TAG:-main-baseline}"

CONFIG_FILE="${CONFIG_FILE:-examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml}"
CONTAINER="${CONTAINER:-${SCRIPT_DIR}/nemo_rl_nightly.sqsh}"
HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/cache}"
MOUNTS="${MOUNTS:-/lustre:/lustre}"
NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-${SCRIPT_DIR}/nrl_megatron_ckpts_20260526_qwen30ba3b}"
WANDB_PROJECT="${WANDB_PROJECT:-sync-grpo-gb200_oci-benchmark}"
UV_PYTHON="${UV_PYTHON:-3.12.13}"
DRIVER_UV_PROJECT_ENVIRONMENT="${DRIVER_UV_PROJECT_ENVIRONMENT:-${SCRIPT_DIR}/.driver_venvs/qwen30ba3b_main_baseline_py312}"
NRL_VLLM_DISABLE_LOG_STATS="${NRL_VLLM_DISABLE_LOG_STATS:-false}"
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
NUM_GENERATIONS="${NUM_GENERATIONS:-32}"
TRAIN_GLOBAL_BATCH_SIZE="${TRAIN_GLOBAL_BATCH_SIZE:-$((NUM_PROMPTS * NUM_GENERATIONS))}"
MAX_STEPS="${MAX_STEPS:-20}"
NRL_MEGATRON_NCCL_TIMEOUT_SECONDS="${NRL_MEGATRON_NCCL_TIMEOUT_SECONDS:-1800}"

mkdir -p "${NRL_MEGATRON_CHECKPOINT_DIR}"

if [[ ! -s "${CONTAINER}" ]]; then
  echo "ERROR: container not found: ${CONTAINER}" >&2
  exit 2
fi

VLLM_EXTRA_OVERRIDES=""
if [[ -n "${VLLM_MAX_NUM_SEQS}" ]]; then
  VLLM_EXTRA_OVERRIDES+=" ++policy.generation.vllm_kwargs.max_num_seqs=${VLLM_MAX_NUM_SEQS}"
fi
if [[ -n "${VLLM_MAX_NUM_BATCHED_TOKENS}" ]]; then
  VLLM_EXTRA_OVERRIDES+=" ++policy.generation.vllm_kwargs.max_num_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS}"
fi

COMMAND="NRL_FORCE_REBUILD_VENVS=true \
UV_PYTHON=${UV_PYTHON} \
UV_PROJECT_ENVIRONMENT=${DRIVER_UV_PROJECT_ENVIRONMENT} \
PYTHONPATH=${SCRIPT_DIR}:${PYTHONPATH:-} \
NRL_MEGATRON_CHECKPOINT_DIR=${NRL_MEGATRON_CHECKPOINT_DIR} \
NRL_MEGATRON_NCCL_TIMEOUT_SECONDS=${NRL_MEGATRON_NCCL_TIMEOUT_SECONDS} \
VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND} \
NRL_VLLM_DISABLE_LOG_STATS=${NRL_VLLM_DISABLE_LOG_STATS} \
uv run --python ${UV_PYTHON} --locked --extra mcore --directory ${SCRIPT_DIR} python ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.enforce_eager=false \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.num_prompts_per_step=${NUM_PROMPTS} \
grpo.num_generations_per_prompt=${NUM_GENERATIONS} \
policy.train_global_batch_size=${TRAIN_GLOBAL_BATCH_SIZE} \
grpo.max_num_steps=${MAX_STEPS} \
${VLLM_EXTRA_OVERRIDES} \
logger.wandb_enabled=true \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='Qwen30B_A3B_Main_N${NUM_NODES}xG${GPUS_PER_NODE}_baseline_p${NUM_PROMPTS}_g${NUM_GENERATIONS}_${MAX_STEPS}step'"

CONTAINER="${CONTAINER}" \
HF_HOME="${HF_HOME}" \
HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
WANDB_API_KEY="${WANDB_API_KEY:-}" \
NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR}" \
UV_PYTHON="${UV_PYTHON}" \
DRIVER_UV_PROJECT_ENVIRONMENT="${DRIVER_UV_PROJECT_ENVIRONMENT}" \
GPUS_PER_NODE="${GPUS_PER_NODE}" \
MOUNTS="${MOUNTS}" \
COMMAND="${COMMAND}" \
sbatch \
  --nodes="${NUM_NODES}" \
  --account="${ACCOUNT}" \
  --job-name="qwen30ba3b-${JOB_TAG}-N${NUM_NODES}xG${GPUS_PER_NODE}" \
  --partition="${PARTITION}" \
  --time=04:00:00 \
  ${GRES_FLAG} \
  ${SBATCH_RESOURCE_ARGS} \
  ${SBATCH_EXTRA_ARGS} \
  --segment "${SEGMENT}" \
  ray.sub
