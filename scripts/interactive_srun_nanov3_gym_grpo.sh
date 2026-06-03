#!/usr/bin/env bash
set -euo pipefail

# In-container one-node runner matching batch_nanov3_gym_grpo.sh, scaled down
# for interactive smoke testing.
#
# Start an interactive container shell first, for example:
#
#   NEMORL=/lustre/fsw/portfolios/llmservice/users/matthieul/repos_rl/nemo-rl-super-baseline
#   srun -A nemotron_omni_vision --job-name nemotron_omni_vision:baseline-gym \
#     --container-image=/lustre/fsw/portfolios/llmservice/users/smohsenitahe/sqsh/super-omni-vllm20-super-vlm2-20260507-0905b74.sqsh \
#     --no-container-mount-home \
#     --container-mounts="/lustre:/lustre,${NEMORL}:/opt/nemo-rl" \
#     --gpus-per-node=8 --partition=interactive --time=4:00:00 \
#     --pty /bin/bash -l
#
# Then run this from inside that shell:
#
#   NEMORL=/opt/nemo-rl bash /opt/nemo-rl/scripts/interactive_srun_nanov3_gym_grpo.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
NEMORL="${NEMORL:-${REPO_ROOT}}"
cd "${NEMORL}"

set +u
if [[ -f "${HOME}/.bashrc" ]]; then
  # shellcheck disable=SC1090
  source "${HOME}/.bashrc"
fi
set -u

if [[ -n "${GITLAB_FLASHINFER_TOKEN:-}" ]]; then
  export UV_INDEX_FLASHINFER_INTERNAL_PYPI_USERNAME="${UV_INDEX_FLASHINFER_INTERNAL_PYPI_USERNAME:-__token__}"
  export UV_INDEX_FLASHINFER_INTERNAL_PYPI_PASSWORD="${UV_INDEX_FLASHINFER_INTERNAL_PYPI_PASSWORD:-${GITLAB_FLASHINFER_TOKEN}}"
fi

DATETIME="$(date +'date_%y-%m-%d_time_%H-%M-%S')"

CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/smohsenitahe/sqsh/super-omni-vllm20-super-vlm2-20260507-0905b74.sqsh}"
JOB_NAME="${JOB_NAME:-interactive_grpo_nanov3omni_gym_super_branch_v6_w_gui_test_${DATETIME}}"
CONFIG="${CONFIG:-examples/nemo_gym/grpo_nanov3omni.yaml}"
MODEL_NAME="/lustre/fsw/portfolios/llmservice/users/smohsenitahe/checkpoints/sft_omni_300k_rebalanced_0301_trunc/"
POLICY_CHAT_TEMPLATE="${POLICY_CHAT_TEMPLATE:-/lustre/fsw/portfolios/llmservice/users/smohsenitahe/checkpoint/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/chat_template.jinja}"
VLLM_CHAT_TEMPLATE="${VLLM_CHAT_TEMPLATE:-${POLICY_CHAT_TEMPLATE}}"
DATA_PATH="${DATA_PATH:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/eagle-next/image_data/rl_data/random_blend_v6_w_gui_mmlongbench_gym.jsonl}"

NUM_PROMPTS_PER_STEP="${NUM_PROMPTS_PER_STEP:-8}"
NUM_GENERATIONS_PER_PROMPT="${NUM_GENERATIONS_PER_PROMPT:-8}"
TRAIN_GLOBAL_BATCH_SIZE="${TRAIN_GLOBAL_BATCH_SIZE:-32}"
GRPO_MAX_NUM_STEPS="${GRPO_MAX_NUM_STEPS:-2}"
MAX_TOTAL_SEQUENCE_LENGTH="${MAX_TOTAL_SEQUENCE_LENGTH:-16384}"
GENERATION_MAX_NEW_TOKENS="${GENERATION_MAX_NEW_TOKENS:-4096}"
SEED="${SEED:-0}"

POLICY_TP="${POLICY_TP:-2}"
POLICY_EP="${POLICY_EP:-2}"
POLICY_CP="${POLICY_CP:-1}"
VLLM_TP="${VLLM_TP:-2}"
POLICY_VLLM_LOAD_FORMAT="${POLICY_VLLM_LOAD_FORMAT:-}"
SEQUENCE_PACKING_ENABLED="${SEQUENCE_PACKING_ENABLED:-true}"
ROLLOUT_NUM_SAMPLES_IN_PARALLEL="${ROLLOUT_NUM_SAMPLES_IN_PARALLEL:-32}"

WANDB_ENABLED="${WANDB_ENABLED:-false}"
TENSORBOARD_ENABLED="${TENSORBOARD_ENABLED:-false}"
WANDB_ENTITY="${WANDB_ENTITY:-adlr}"
WANDB_PROJECT="${WANDB_PROJECT:-grpo-nanov3vl}"
NEMO_GYM_LOG_RESPONSES="${NEMO_GYM_LOG_RESPONSES:-false}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES//$'\n'/ }"

RESULTS_DIR="${RESULTS_DIR:-results/${JOB_NAME}}"
APP_LOG_DIR="${APP_LOG_DIR:-${RESULTS_DIR}/logs}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-${RESULTS_DIR}/slurm}"
RUN_METADATA_DIR="${RUN_METADATA_DIR:-${SLURM_LOG_DIR}/metadata}"

CACHE_ROOT="${CACHE_ROOT:-/lustre/fsw/portfolios/llmservice/users/matthieul/cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/llmservice/users/matthieul/cache/huggingface/}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${CACHE_ROOT}/uv}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export OPENAI_BASE_URL="https://inference-api.nvidia.com/v1/chat/completions"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export HF_TOKEN="${HF_TOKEN:-}"

export CONTAINER
export VLLM_PRECOMPILED_WHEEL_LOCATION="${VLLM_PRECOMPILED_WHEEL_LOCATION:-https://github.com/vllm-project/vllm/releases/download/v0.20.1/vllm-0.20.1%2Bcu129-cp38-abi3-manylinux_2_31_x86_64.whl}"
export PYTHONPATH="${HF_MODULES_CACHE}:${NEMORL}:${NEMORL}/3rdparty/Gym-workspace/Gym:${NEMORL}/3rdparty/vllm:${NEMORL}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${NEMORL}/3rdparty/Megatron-LM-workspace/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-false}"
export NRL_VENVS_TRUST_EXISTING="${NRL_VENVS_TRUST_EXISTING:-1}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR:-/opt/ray_venvs}"
export NEMO_RL_VLLM_PY_EXECUTABLE_SYSTEM="${NEMO_RL_VLLM_PY_EXECUTABLE_SYSTEM:-1}"
export NEMO_RL_MCORE_PY_EXECUTABLE_SYSTEM="${NEMO_RL_MCORE_PY_EXECUTABLE_SYSTEM:-0}"
export NEMO_RL_NEMO_GYM_PY_EXECUTABLE_SYSTEM="${NEMO_RL_NEMO_GYM_PY_EXECUTABLE_SYSTEM:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NVTE_FWD_LAYERNORM_SM_MARGIN="${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}"
export NVTE_BWD_LAYERNORM_SM_MARGIN="${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}"
export NEMO_RL_LOG_GPU_MEMORY="${NEMO_RL_LOG_GPU_MEMORY:-1}"
export NEMO_RL_PER_WORKER_COMPILER_CACHE="${NEMO_RL_PER_WORKER_COMPILER_CACHE:-1}"
export NEMO_RL_COMPILER_CACHE_ROOT="${NEMO_RL_COMPILER_CACHE_ROOT:-/tmp/nemo_rl_compiler_cache/${USER}/${JOB_NAME}}"
export NRL_IGNORE_VERSION_MISMATCH="${NRL_IGNORE_VERSION_MISMATCH:-true}"
export MIN_WORKER_PORT="${MIN_WORKER_PORT:-2000}"
export MAX_WORKER_PORT="${MAX_WORKER_PORT:-2999}"

mkdir -p \
  "${XDG_CACHE_HOME}" \
  "${HUGGINGFACE_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${HF_MODULES_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${TORCH_HOME}" \
  "${TRITON_CACHE_DIR}" \
  "${UV_CACHE_DIR}" \
  "${APP_LOG_DIR}" \
  "${SLURM_LOG_DIR}" \
  "${RUN_METADATA_DIR}"

if [[ ! -s "${DATA_PATH}" ]]; then
  echo "[ERROR] DATA_PATH is missing or empty: ${DATA_PATH}" >&2
  exit 1
fi

cmd=(
  uv run --no-sync examples/nemo_gym/run_grpo_nemo_gym.py --config "${CONFIG}"
  cluster.num_nodes=1
  cluster.gpus_per_node=8
  policy.model_name="${MODEL_NAME}"
  policy.megatron_cfg.tensor_model_parallel_size="${POLICY_TP}"
  policy.tokenizer.chat_template="${POLICY_CHAT_TEMPLATE}"
  policy.generation.vllm_cfg.http_server_serving_chat_kwargs.chat_template="${VLLM_CHAT_TEMPLATE}"
  policy.megatron_cfg.expert_model_parallel_size="${POLICY_EP}"
  policy.megatron_cfg.context_parallel_size="${POLICY_CP}"
  policy.generation.vllm_cfg.tensor_parallel_size="${VLLM_TP}"
  +policy.generation.vllm_cfg.logprobs_mode=raw_logprobs
  +policy.generation.vllm_kwargs.mm_processor_cache_gb=0
  policy.sequence_packing.enabled="${SEQUENCE_PACKING_ENABLED}"
  grpo.seed="${SEED}"
  grpo.num_prompts_per_step="${NUM_PROMPTS_PER_STEP}"
  grpo.num_generations_per_prompt="${NUM_GENERATIONS_PER_PROMPT}"
  grpo.max_num_steps="${GRPO_MAX_NUM_STEPS}"
  policy.megatron_cfg.scheduler.lr_warmup_iters=1
  policy.train_global_batch_size="${TRAIN_GLOBAL_BATCH_SIZE}"
  checkpointing.checkpoint_dir="${RESULTS_DIR}"
  logger.log_dir="${APP_LOG_DIR}"
  logger.wandb_enabled="${WANDB_ENABLED}"
  logger.tensorboard_enabled="${TENSORBOARD_ENABLED}"
  +logger.wandb.entity="${WANDB_ENTITY}"
  logger.wandb.project="${WANDB_PROJECT}"
  logger.wandb.name="${JOB_NAME}"
  policy.max_total_sequence_length="${MAX_TOTAL_SEQUENCE_LENGTH}"
  policy.generation.max_new_tokens="${GENERATION_MAX_NEW_TOKENS}"
  policy.generation.vllm_cfg.max_model_len="${MAX_TOTAL_SEQUENCE_LENGTH}"
  policy.generation.vllm_kwargs.max_num_batched_tokens="${MAX_TOTAL_SEQUENCE_LENGTH}"
  data.train.data_path="${DATA_PATH}"
  data.validation.data_path="${DATA_PATH}"
  +env.nemo_gym.skip_venv_if_present=true
  +env.nemo_gym.policy_model.responses_api_models.vllm_model.max_input_tokens="${MAX_TOTAL_SEQUENCE_LENGTH}"
  +env.nemo_gym.rollout_num_samples_in_parallel="${ROLLOUT_NUM_SAMPLES_IN_PARALLEL}"
  env.should_log_nemo_gym_responses="${NEMO_GYM_LOG_RESPONSES}"
)

if [[ -n "${POLICY_VLLM_LOAD_FORMAT}" ]]; then
  cmd+=(policy.generation.vllm_cfg.load_format="${POLICY_VLLM_LOAD_FORMAT}")
fi

if [[ -n "${EXTRA_OVERRIDES}" ]]; then
  # shellcheck disable=SC2206
  extra_args=( ${EXTRA_OVERRIDES} )
  cmd+=("${extra_args[@]}")
fi

RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
{
  printf '#!/usr/bin/env bash\n'
  printf 'set -euo pipefail\n'
  printf '%q ' "${cmd[@]}"
  printf '\n'
} > "${RUN_METADATA_DIR}/interactive_srun_command_${RUN_TS}.sh"
chmod +x "${RUN_METADATA_DIR}/interactive_srun_command_${RUN_TS}.sh"

echo "[INFO] Running one-node interactive NanoV3 Omni Gym GRPO job: ${JOB_NAME}"
echo "[INFO] repo=${NEMORL}"
echo "[INFO] expected_container=${CONTAINER}"
echo "[INFO] config=${CONFIG}"
echo "[INFO] model=${MODEL_NAME}"
echo "[INFO] data=${DATA_PATH}"
echo "[INFO] chat_template=${POLICY_CHAT_TEMPLATE}"
echo "[INFO] policy_tp=${POLICY_TP} policy_ep=${POLICY_EP} policy_cp=${POLICY_CP} vllm_tp=${VLLM_TP}"
echo "[INFO] prompts=${NUM_PROMPTS_PER_STEP} generations=${NUM_GENERATIONS_PER_PROMPT} train_gbs=${TRAIN_GLOBAL_BATCH_SIZE}"
echo "[INFO] max_steps=${GRPO_MAX_NUM_STEPS} sequence_packing=${SEQUENCE_PACKING_ENABLED}"
echo "[INFO] rollout_num_samples_in_parallel=${ROLLOUT_NUM_SAMPLES_IN_PARALLEL}"
echo "[INFO] wandb_enabled=${WANDB_ENABLED}"
echo "[INFO] app_logs=${APP_LOG_DIR}"
echo "[INFO] run_logs=${SLURM_LOG_DIR}/interactive_srun_${RUN_TS}.log"
echo "[INFO] python_runner=uv run --no-sync"
"${cmd[@]}" 2>&1 | tee "${SLURM_LOG_DIR}/interactive_srun_${RUN_TS}.log"
