#!/usr/bin/env bash
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

NUM_NODES="${NUM_NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
PARTITION="${PARTITION:-batch}"
ACCOUNT="${ACCOUNT:-coreai_dlalgo_nemorl}"
GRES_FLAG="${GRES_FLAG:---gres=gpu:4}"
SEGMENT="${SEGMENT:-${NUM_NODES}}"
CPUS_PER_WORKER="${CPUS_PER_WORKER:-$((GPUS_PER_NODE * 16))}"
SBATCH_RESOURCE_ARGS="${SBATCH_RESOURCE_ARGS:---ntasks-per-node=1 --cpus-per-task=${CPUS_PER_WORKER} --mem=0}"
SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:-}"
JOB_TAG="${JOB_TAG:-qwen3-8b-publichf-specdec}"

CONFIG_FILE="${CONFIG_FILE:-examples/configs/recipes/llm/grpo-qwen3-8b-base-1n8g-fp8-kvcache-megatron.yaml}"
TARGET_MODEL_ID="${TARGET_MODEL_ID:-Qwen/Qwen3-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-RedHatAI/Qwen3-8B-speculator.eagle3}"
DRAFT_MODEL_PROVENANCE="${DRAFT_MODEL_PROVENANCE:-RedHatAI_Qwen3-8B-speculator.eagle3_public_hf}"
SPECDEC_METHOD="${SPECDEC_METHOD:-eagle3}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
DRAFT_TP="${DRAFT_TP:-1}"
CONTAINER="${CONTAINER:-${SCRIPT_DIR}/nemo_rl_nightly.sqsh}"
HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/cache}"
MOUNTS="${MOUNTS:-/lustre:/lustre}"
NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-${SCRIPT_DIR}/nrl_megatron_ckpts_20260602_qwen3_8b}"
WANDB_PROJECT="${WANDB_PROJECT:-sync-grpo-gb200_oci-benchmark}"
UV_PYTHON="${UV_PYTHON:-3.12.13}"
RAY_VERSION="${RAY_VERSION:-2.49.2}"
DRIVER_UV_PROJECT_ENVIRONMENT="${DRIVER_UV_PROJECT_ENVIRONMENT:-${SCRIPT_DIR}/.driver_venvs/qwen3_8b_publichf_specdec_py312}"
DRIVER_SRUN_CPUS_PER_TASK="${DRIVER_SRUN_CPUS_PER_TASK:-8}"
DRIVER_SRUN_MEM="${DRIVER_SRUN_MEM:-128G}"
MAX_JOBS="${MAX_JOBS:-4}"
CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
NVTE_BUILD_MAX_JOBS="${NVTE_BUILD_MAX_JOBS:-4}"
NINJAFLAGS="${NINJAFLAGS:--j4}"
MAKEFLAGS="${MAKEFLAGS:--j4}"
NRL_MEGATRON_NCCL_TIMEOUT_SECONDS="${NRL_MEGATRON_NCCL_TIMEOUT_SECONDS:-1800}"
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"
SPECDEC_SCHEDULER_PATCH_GATE_THRESHOLD="${SPECDEC_SCHEDULER_PATCH_GATE_THRESHOLD:-8}"
SPECDEC_SCHEDULER_PATCH_GATE_TOKEN_THRESHOLD="${SPECDEC_SCHEDULER_PATCH_GATE_TOKEN_THRESHOLD:-4096}"
ENABLE_RUNTIME_SPECDEC_GATE_PATCH="${ENABLE_RUNTIME_SPECDEC_GATE_PATCH:-true}"
VLLM_SPECDEC_BATCH_GATE_LOG_INTERVAL="${VLLM_SPECDEC_BATCH_GATE_LOG_INTERVAL:-64}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-}"
SPECDEC_EXTRA_OVERRIDES="${SPECDEC_EXTRA_OVERRIDES:-${EXTRA_OVERRIDES:-}}"
NRL_VLLM_DISABLE_LOG_STATS="${NRL_VLLM_DISABLE_LOG_STATS:-false}"
NRL_VLLM_OMIT_GENERATION_LOGPROBS="${NRL_VLLM_OMIT_GENERATION_LOGPROBS:-true}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
NUM_GENERATIONS="${NUM_GENERATIONS:-32}"
TRAIN_GLOBAL_BATCH_SIZE="${TRAIN_GLOBAL_BATCH_SIZE:-$((NUM_PROMPTS * NUM_GENERATIONS))}"
MAX_STEPS="${MAX_STEPS:-20}"
WALLTIME="${WALLTIME:-04:00:00}"
WANDB_NAME="${WANDB_NAME:-Qwen3_8B_N${NUM_NODES}xG${GPUS_PER_NODE}_publichf_eagle3_k${NUM_SPECULATIVE_TOKENS}_req${SPECDEC_SCHEDULER_PATCH_GATE_THRESHOLD}_p${NUM_PROMPTS}_g${NUM_GENERATIONS}_${MAX_STEPS}step}"

mkdir -p "${NRL_MEGATRON_CHECKPOINT_DIR}"

if [[ ! -s "${CONTAINER}" ]]; then
  echo "ERROR: container not found: ${CONTAINER}" >&2
  exit 2
fi
if [[ ! -s "${SCRIPT_DIR}/ray.sub" ]]; then
  echo "ERROR: patched ray.sub not found at ${SCRIPT_DIR}/ray.sub" >&2
  exit 2
fi
require_patch_marker() {
  local file="$1"
  local marker="$2"
  if [[ ! -s "${file}" ]] || ! grep -q "${marker}" "${file}"; then
    echo "ERROR: required SpecDec-RL patch marker '${marker}' is missing from ${file}" >&2
    exit 2
  fi
}
require_patch_marker "${SCRIPT_DIR}/nemo_rl/models/generation/vllm/vllm_worker.py" "NRL_SPECDEC_BATCH_GATE_PATCH_V9"
require_patch_marker "${SCRIPT_DIR}/nemo_rl/models/generation/vllm/vllm_worker.py" "NRL_SPECDEC_BATCH_GATE_ZERO_DRAFTS_ON_DISABLE_V1"
require_patch_marker "${SCRIPT_DIR}/nemo_rl/models/generation/vllm/vllm_generation.py" "acceptance_rate_reliable"

if [[ -n "${VLLM_MAX_NUM_SEQS}" ]]; then
  SPECDEC_EXTRA_OVERRIDES+=" ++policy.generation.vllm_kwargs.max_num_seqs=${VLLM_MAX_NUM_SEQS}"
fi
if [[ -n "${VLLM_MAX_NUM_BATCHED_TOKENS}" ]]; then
  SPECDEC_EXTRA_OVERRIDES+=" ++policy.generation.vllm_kwargs.max_num_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS}"
fi

COMMAND="NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS:-true} \
UV_PYTHON=${UV_PYTHON} \
UV_PROJECT_ENVIRONMENT=${DRIVER_UV_PROJECT_ENVIRONMENT} \
PYTHONPATH=${SCRIPT_DIR}:${PYTHONPATH:-} \
MAX_JOBS=${MAX_JOBS} \
CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL} \
NVTE_BUILD_MAX_JOBS=${NVTE_BUILD_MAX_JOBS} \
NINJAFLAGS=${NINJAFLAGS} \
MAKEFLAGS=${MAKEFLAGS} \
NRL_MEGATRON_CHECKPOINT_DIR=${NRL_MEGATRON_CHECKPOINT_DIR} \
NRL_MEGATRON_NCCL_TIMEOUT_SECONDS=${NRL_MEGATRON_NCCL_TIMEOUT_SECONDS} \
VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND} \
VLLM_SPECDEC_BATCH_SIZE_GATE_THRESHOLD=${SPECDEC_SCHEDULER_PATCH_GATE_THRESHOLD:-0} \
VLLM_SPECDEC_BATCH_TOKEN_GATE_THRESHOLD=${SPECDEC_SCHEDULER_PATCH_GATE_TOKEN_THRESHOLD:-0} \
VLLM_ENABLE_RUNTIME_SPECDEC_BATCH_GATE_PATCH=${ENABLE_RUNTIME_SPECDEC_GATE_PATCH} \
VLLM_SPECDEC_BATCH_GATE_LOG_INTERVAL=${VLLM_SPECDEC_BATCH_GATE_LOG_INTERVAL} \
NRL_VLLM_DISABLE_LOG_STATS=${NRL_VLLM_DISABLE_LOG_STATS} \
NRL_VLLM_OMIT_GENERATION_LOGPROBS=${NRL_VLLM_OMIT_GENERATION_LOGPROBS} \
NRL_SPECDEC_DRAFT_MODEL_PROVENANCE=${DRAFT_MODEL_PROVENANCE} \
uv run --python ${UV_PYTHON} --locked --extra mcore --directory ${SCRIPT_DIR} python ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.model_name=${TARGET_MODEL_ID} \
policy.tokenizer.name=${TARGET_MODEL_ID} \
policy.generation.vllm_cfg.enforce_eager=false \
policy.generation.vllm_cfg.async_engine=false \
policy.sequence_packing.enabled=true \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.num_prompts_per_step=${NUM_PROMPTS} \
grpo.num_generations_per_prompt=${NUM_GENERATIONS} \
policy.train_global_batch_size=${TRAIN_GLOBAL_BATCH_SIZE} \
 ++policy.megatron_cfg.moe_enable_deepep=false \
 ++policy.megatron_cfg.moe_token_dispatcher_type=alltoall \
 ++policy.megatron_cfg.moe_shared_expert_overlap=false \
grpo.max_num_steps=${MAX_STEPS} \
++policy.generation.vllm_kwargs.speculative_config.method=${SPECDEC_METHOD} \
++policy.generation.vllm_kwargs.speculative_config.model=${DRAFT_MODEL} \
++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=${NUM_SPECULATIVE_TOKENS} \
++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=${DRAFT_TP} \
${SPECDEC_EXTRA_OVERRIDES} \
logger.wandb_enabled=true \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"

CONTAINER="${CONTAINER}" \
HF_HOME="${HF_HOME}" \
HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
WANDB_API_KEY="${WANDB_API_KEY:-}" \
NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR}" \
UV_PYTHON="${UV_PYTHON}" \
RAY_VERSION="${RAY_VERSION}" \
DRIVER_UV_PROJECT_ENVIRONMENT="${DRIVER_UV_PROJECT_ENVIRONMENT}" \
DRIVER_SRUN_CPUS_PER_TASK="${DRIVER_SRUN_CPUS_PER_TASK}" \
DRIVER_SRUN_MEM="${DRIVER_SRUN_MEM}" \
GPUS_PER_NODE="${GPUS_PER_NODE}" \
MOUNTS="${MOUNTS}" \
COMMAND="${COMMAND}" \
sbatch \
  --nodes="${NUM_NODES}" \
  --account="${ACCOUNT}" \
  --job-name="qwen3-8b-${JOB_TAG}-N${NUM_NODES}xG${GPUS_PER_NODE}" \
  --partition="${PARTITION}" \
  --time="${WALLTIME}" \
  ${GRES_FLAG} \
  ${SBATCH_RESOURCE_ARGS} \
  ${SBATCH_EXTRA_ARGS} \
  --segment "${SEGMENT}" \
  "${SCRIPT_DIR}/ray.sub"
