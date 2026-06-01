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
if [[ "$(cd "${SCRIPT_DIR}" && pwd -P)" == *"/remote_worktree_edit"* ]]; then
  echo "ERROR: refusing to launch from stale scratch tree: ${SCRIPT_DIR}" >&2
  echo "Use the maintained SpecDec-RL checkout/overlay instead." >&2
  exit 2
fi

NUM_NODES="${NUM_NODES:-32}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
PARTITION="${PARTITION:-batch}"
ACCOUNT="${ACCOUNT:-coreai_dlalgo_nemorl}"
GRES_FLAG="${GRES_FLAG:---gres=gpu:4}"
SEGMENT="${SEGMENT:-16}"
CPUS_PER_WORKER="${CPUS_PER_WORKER:-$((GPUS_PER_NODE * 16))}"
SBATCH_RESOURCE_ARGS="${SBATCH_RESOURCE_ARGS:---ntasks-per-node=1 --cpus-per-task=${CPUS_PER_WORKER} --mem=0}"
SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:-}"
JOB_TAG="${JOB_TAG:-main-baseline}"

CONFIG_FILE="${CONFIG_FILE:-examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n8g.yaml}"
TARGET_MODEL_ID="${TARGET_MODEL_ID:-Qwen/Qwen3-235B-A22B-Thinking-2507}"
CONTAINER="${CONTAINER:-${SCRIPT_DIR}/nemo_rl_nightly.sqsh}"
HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/cache}"
MOUNTS="${MOUNTS:-/lustre:/lustre}"
NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-${SCRIPT_DIR}/nrl_megatron_ckpts_20260526}"
WANDB_PROJECT="${WANDB_PROJECT:-sync-grpo-gb200_oci-benchmark}"
UV_PYTHON="${UV_PYTHON:-3.12.13}"
RAY_VERSION="${RAY_VERSION:-2.49.2}"
DRIVER_UV_PROJECT_ENVIRONMENT="${DRIVER_UV_PROJECT_ENVIRONMENT:-${SCRIPT_DIR}/.driver_venvs/qwen235b_main_baseline_py312}"
NUM_PROMPTS="${NUM_PROMPTS:-16}"
NUM_GENERATIONS="${NUM_GENERATIONS:-32}"
TRAIN_GLOBAL_BATCH_SIZE="${TRAIN_GLOBAL_BATCH_SIZE:-$((NUM_PROMPTS * NUM_GENERATIONS))}"
MAX_STEPS="${MAX_STEPS:-20}"
NRL_MEGATRON_NCCL_TIMEOUT_SECONDS="${NRL_MEGATRON_NCCL_TIMEOUT_SECONDS:-1800}"
NRL_VLLM_DISABLE_LOG_STATS="${NRL_VLLM_DISABLE_LOG_STATS:-false}"
NRL_VLLM_OMIT_GENERATION_LOGPROBS="${NRL_VLLM_OMIT_GENERATION_LOGPROBS:-true}"
NRL_STOP_AFTER_GENERATION="${NRL_STOP_AFTER_GENERATION:-false}"
REQUIRE_SPECDEC_RL_PATCHES="${REQUIRE_SPECDEC_RL_PATCHES:-true}"
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-}"

mkdir -p "${NRL_MEGATRON_CHECKPOINT_DIR}"

if [[ ! -s "${CONTAINER}" ]]; then
  echo "ERROR: container not found: ${CONTAINER}" >&2
  exit 2
fi
if [[ ! -s "${SCRIPT_DIR}/ray.sub" ]]; then
  echo "ERROR: patched ray.sub not found at ${SCRIPT_DIR}/ray.sub" >&2
  exit 2
fi
if [[ "${REQUIRE_SPECDEC_RL_PATCHES}" == "true" || "${REQUIRE_SPECDEC_RL_PATCHES}" == "True" ]]; then
  require_patch_marker() {
    local file="$1"
    local marker="$2"
    if [[ ! -s "${file}" ]] || ! grep -q "${marker}" "${file}"; then
      echo "ERROR: required SpecDec-RL parity marker '${marker}' is missing from ${file}" >&2
      echo "Use NEMO_RL_DIR=/lustre/fs1/.../SpecDec-RL with the current patch bundle, or set REQUIRE_SPECDEC_RL_PATCHES=false only for diagnostics." >&2
      exit 2
    fi
  }
  require_patch_marker "${SCRIPT_DIR}/nemo_rl/models/generation/vllm/vllm_worker.py" "NRL_VLLM_OMIT_GENERATION_LOGPROBS"
  require_patch_marker "${SCRIPT_DIR}/nemo_rl/algorithms/grpo.py" "_repair_specdec_generation_logprobs_if_safe"
fi

VLLM_EXTRA_OVERRIDES=""
if [[ -n "${VLLM_MAX_NUM_SEQS}" ]]; then
  VLLM_EXTRA_OVERRIDES+=" ++policy.generation.vllm_kwargs.max_num_seqs=${VLLM_MAX_NUM_SEQS}"
fi
if [[ -n "${VLLM_MAX_NUM_BATCHED_TOKENS}" ]]; then
  VLLM_EXTRA_OVERRIDES+=" ++policy.generation.vllm_kwargs.max_num_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS}"
fi

COMMAND="NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS:-true} \
UV_PYTHON=${UV_PYTHON} \
UV_PROJECT_ENVIRONMENT=${DRIVER_UV_PROJECT_ENVIRONMENT} \
PYTHONPATH=${SCRIPT_DIR}:${PYTHONPATH:-} \
NRL_MEGATRON_CHECKPOINT_DIR=${NRL_MEGATRON_CHECKPOINT_DIR} \
NRL_MEGATRON_NCCL_TIMEOUT_SECONDS=${NRL_MEGATRON_NCCL_TIMEOUT_SECONDS} \
VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND} \
NRL_VLLM_DISABLE_LOG_STATS=${NRL_VLLM_DISABLE_LOG_STATS} \
NRL_VLLM_OMIT_GENERATION_LOGPROBS=${NRL_VLLM_OMIT_GENERATION_LOGPROBS} \
NRL_STOP_AFTER_GENERATION=${NRL_STOP_AFTER_GENERATION} \
uv run --python ${UV_PYTHON} --locked --extra mcore --directory ${SCRIPT_DIR} python ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.model_name=${TARGET_MODEL_ID} \
policy.generation.vllm_cfg.tensor_parallel_size=16 \
policy.generation.vllm_cfg.expert_parallel_size=1 \
policy.generation.vllm_cfg.pipeline_parallel_size=1 \
policy.generation.vllm_cfg.enforce_eager=false \
policy.generation.vllm_cfg.async_engine=false \
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
${VLLM_EXTRA_OVERRIDES} \
logger.wandb_enabled=true \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='Qwen235B_A22B_Main_N${NUM_NODES}xG${GPUS_PER_NODE}_baseline_p${NUM_PROMPTS}_g${NUM_GENERATIONS}_${MAX_STEPS}step'"

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
  --job-name="qwen235b-${JOB_TAG}-N${NUM_NODES}xG${GPUS_PER_NODE}" \
  --partition="${PARTITION}" \
  --time=04:00:00 \
  ${GRES_FLAG} \
  ${SBATCH_RESOURCE_ARGS} \
  ${SBATCH_EXTRA_ARGS} \
  --segment "${SEGMENT}" \
  "${SCRIPT_DIR}/ray.sub"
