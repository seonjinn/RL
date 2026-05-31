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
JOB_TAG="${JOB_TAG:-main-specdec}"

CONFIG_FILE="${CONFIG_FILE:-examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml}"
CONTAINER="${CONTAINER:-${SCRIPT_DIR}/nemo_rl_nightly.sqsh}"
HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/cache}"
MOUNTS="${MOUNTS:-/lustre:/lustre}"
NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-${SCRIPT_DIR}/nrl_megatron_ckpts_20260526_qwen30ba3b}"
WANDB_PROJECT="${WANDB_PROJECT:-sync-grpo-gb200_oci-benchmark}"
UV_PYTHON="${UV_PYTHON:-3.12.13}"
RAY_VERSION="${RAY_VERSION:-2.49.2}"
DRIVER_UV_PROJECT_ENVIRONMENT="${DRIVER_UV_PROJECT_ENVIRONMENT:-${SCRIPT_DIR}/.driver_venvs/qwen30ba3b_main_specdec_py312}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
NUM_GENERATIONS="${NUM_GENERATIONS:-32}"
TRAIN_GLOBAL_BATCH_SIZE="${TRAIN_GLOBAL_BATCH_SIZE:-$((NUM_PROMPTS * NUM_GENERATIONS))}"
MAX_STEPS="${MAX_STEPS:-20}"

DRAFT_ROOT="${DRAFT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/speculators/eagle3_qwen3_30ba3b_openmath_reasoning_cot_50k/checkpoints_train_50k_layers48_mlen8193_finalpatch}"
DRAFT_MODEL="${DRAFT_MODEL:-}"
SPECDEC_METHOD="${SPECDEC_METHOD:-eagle3}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
DRAFT_TP="${DRAFT_TP:-1}"
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"
SPECDEC_DISABLE_BY_BATCH_SIZE="${SPECDEC_DISABLE_BY_BATCH_SIZE:-}"
SPECDEC_SCHEDULER_PATCH_GATE_THRESHOLD="${SPECDEC_SCHEDULER_PATCH_GATE_THRESHOLD:-}"
SPECDEC_SCHEDULER_PATCH_GATE_TOKEN_THRESHOLD="${SPECDEC_SCHEDULER_PATCH_GATE_TOKEN_THRESHOLD:-}"
ENABLE_RUNTIME_SPECDEC_GATE_PATCH="${ENABLE_RUNTIME_SPECDEC_GATE_PATCH:-false}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-}"
NRL_VLLM_DISABLE_LOG_STATS="${NRL_VLLM_DISABLE_LOG_STATS:-false}"
NRL_VLLM_OMIT_GENERATION_LOGPROBS="${NRL_VLLM_OMIT_GENERATION_LOGPROBS:-false}"

if {
  [[ -n "${SPECDEC_SCHEDULER_PATCH_GATE_THRESHOLD}" && "${SPECDEC_SCHEDULER_PATCH_GATE_THRESHOLD}" != "0" ]] ||
  [[ -n "${SPECDEC_SCHEDULER_PATCH_GATE_TOKEN_THRESHOLD}" && "${SPECDEC_SCHEDULER_PATCH_GATE_TOKEN_THRESHOLD}" != "0" ]]
}; then
  case "${ENABLE_RUNTIME_SPECDEC_GATE_PATCH}" in
    1|true|TRUE|yes|YES|y|Y|on|ON) ;;
    *)
      echo "ERROR: SpecDec scheduler gate thresholds require ENABLE_RUNTIME_SPECDEC_GATE_PATCH=true" >&2
      exit 2
      ;;
  esac
fi
if [[ -n "${SPECDEC_DISABLE_BY_BATCH_SIZE}" ]]; then
  echo "ERROR: SPECDEC_DISABLE_BY_BATCH_SIZE is not the NeMo-RL long-tail gate. Use SPECDEC_SCHEDULER_PATCH_GATE_THRESHOLD or SPECDEC_SCHEDULER_PATCH_GATE_TOKEN_THRESHOLD with ENABLE_RUNTIME_SPECDEC_GATE_PATCH=true." >&2
  exit 2
fi

mkdir -p "${NRL_MEGATRON_CHECKPOINT_DIR}"

if [[ ! -s "${CONTAINER}" ]]; then
  echo "ERROR: container not found: ${CONTAINER}" >&2
  exit 2
fi
if [[ ! -s "${SCRIPT_DIR}/ray.sub" ]]; then
  echo "ERROR: patched ray.sub not found at ${SCRIPT_DIR}/ray.sub" >&2
  exit 2
fi

if [[ -z "${DRAFT_MODEL}" ]]; then
  DRAFT_MODEL="$(
    find "${DRAFT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' \
      -exec sh -c 'test -s "$1/config.json"' sh {} \; -print 2>/dev/null \
      | sort -V \
      | tail -n 1
  )"
fi

if [[ -z "${DRAFT_MODEL}" || ! -s "${DRAFT_MODEL}/config.json" ]]; then
  echo "ERROR: DRAFT_MODEL is not ready. Set DRAFT_MODEL or wait for a checkpoint under: ${DRAFT_ROOT}" >&2
  exit 2
fi

SPECDEC_EXTRA_OVERRIDES=""
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
NRL_MEGATRON_CHECKPOINT_DIR=${NRL_MEGATRON_CHECKPOINT_DIR} \
VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND} \
VLLM_SPECDEC_BATCH_SIZE_GATE_THRESHOLD=${SPECDEC_SCHEDULER_PATCH_GATE_THRESHOLD:-0} \
VLLM_SPECDEC_BATCH_TOKEN_GATE_THRESHOLD=${SPECDEC_SCHEDULER_PATCH_GATE_TOKEN_THRESHOLD:-0} \
VLLM_ENABLE_RUNTIME_SPECDEC_BATCH_GATE_PATCH=${ENABLE_RUNTIME_SPECDEC_GATE_PATCH} \
NRL_VLLM_DISABLE_LOG_STATS=${NRL_VLLM_DISABLE_LOG_STATS} \
NRL_VLLM_OMIT_GENERATION_LOGPROBS=${NRL_VLLM_OMIT_GENERATION_LOGPROBS} \
uv run --python ${UV_PYTHON} --locked --extra mcore --directory ${SCRIPT_DIR} python ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.enforce_eager=false \
policy.generation.vllm_cfg.async_engine=false \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.num_prompts_per_step=${NUM_PROMPTS} \
grpo.num_generations_per_prompt=${NUM_GENERATIONS} \
policy.train_global_batch_size=${TRAIN_GLOBAL_BATCH_SIZE} \
grpo.max_num_steps=${MAX_STEPS} \
++policy.generation.vllm_kwargs.speculative_config.method=${SPECDEC_METHOD} \
++policy.generation.vllm_kwargs.speculative_config.model=${DRAFT_MODEL} \
++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=${NUM_SPECULATIVE_TOKENS} \
++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=${DRAFT_TP} \
${SPECDEC_EXTRA_OVERRIDES} \
logger.wandb_enabled=true \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='Qwen30B_A3B_Main_N${NUM_NODES}xG${GPUS_PER_NODE}_specdec_k${NUM_SPECULATIVE_TOKENS}_p${NUM_PROMPTS}_g${NUM_GENERATIONS}_${MAX_STEPS}step'"

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
  "${SCRIPT_DIR}/ray.sub"
