#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMORL="${NEMORL:-${SCRIPT_DIR}}"
cd "${NEMORL}"

set +u
if [[ -f "${HOME}/.bashrc" ]]; then
  # shellcheck disable=SC1090
  source "${HOME}/.bashrc"
fi

set -a
if [[ -f /lustre/fsw/portfolios/llmservice/users/smohsenitahe/.env ]]; then
  # shellcheck disable=SC1091
  source /lustre/fsw/portfolios/llmservice/users/smohsenitahe/.env
elif [[ -f "${NEMORL}/.env" ]]; then
  # shellcheck disable=SC1091
  source "${NEMORL}/.env"
fi
set +a
set -u

if [[ -n "${GITLAB_FLASHINFER_TOKEN:-}" ]]; then
  export UV_INDEX_FLASHINFER_INTERNAL_PYPI_USERNAME="${UV_INDEX_FLASHINFER_INTERNAL_PYPI_USERNAME:-__token__}"
  export UV_INDEX_FLASHINFER_INTERNAL_PYPI_PASSWORD="${UV_INDEX_FLASHINFER_INTERNAL_PYPI_PASSWORD:-${GITLAB_FLASHINFER_TOKEN}}"
fi

NUM_NODES="${NUM_NODES:-8}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
JOB_NAME="${JOB_NAME:-grpo-nanov3omni-gym-super-branch-v6-w-gui-test}"
SEED="${SEED:-$(echo -n "train:${JOB_NAME}" | openssl dgst -md5 -binary | od -An -tu4 -N4 | xargs)}"

MODEL_NAME="${MODEL_NAME:-/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_sft/users/pjin/checkpoints/nano-v3-vl-mpo_sft_mmlongbench_txt_0403_2200-iter-200-rl-20260407-step-50-hf}"
POLICY_CHAT_TEMPLATE="${POLICY_CHAT_TEMPLATE:-/lustre/fsw/portfolios/llmservice/users/smohsenitahe/checkpoint/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/chat_template.jinja}"
VLLM_CHAT_TEMPLATE="${VLLM_CHAT_TEMPLATE:-${POLICY_CHAT_TEMPLATE}}"
DATA_PATH="${DATA_PATH:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/eagle-next/image_data/rl_data/random_blend_v6_w_gui_mmlongbench_gym.jsonl}"

NUM_PROMPTS_PER_STEP="${NUM_PROMPTS_PER_STEP:-32}"
NUM_GENERATIONS_PER_PROMPT="${NUM_GENERATIONS_PER_PROMPT:-16}"
TRAIN_GLOBAL_BATCH_SIZE="${TRAIN_GLOBAL_BATCH_SIZE:-$((NUM_PROMPTS_PER_STEP * NUM_GENERATIONS_PER_PROMPT))}"
GRPO_MAX_NUM_STEPS="${GRPO_MAX_NUM_STEPS:-10}"

POLICY_TP="${POLICY_TP:-8}"
POLICY_EP="${POLICY_EP:-8}"
POLICY_CP="${POLICY_CP:-1}"
VLLM_TP="${VLLM_TP:-2}"
POLICY_VLLM_LOAD_FORMAT="${POLICY_VLLM_LOAD_FORMAT:-}"
if [[ -z "${SEQUENCE_PACKING_ENABLED+x}" && "${POLICY_CP}" -gt 1 ]]; then
  SEQUENCE_PACKING_ENABLED=true
fi
SEQUENCE_PACKING_ENABLED="${SEQUENCE_PACKING_ENABLED:-false}"

RESULTS_DIR="${RESULTS_DIR:-results/${JOB_NAME}}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-grpo-nanov3omni-gym-super-branch-v6-w-gui-test}"
NEMO_GYM_LOG_RESPONSES="${NEMO_GYM_LOG_RESPONSES:-true}"

export NUM_NODES GPUS_PER_NODE
export CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/smohsenitahe/sqsh/super-omni-vllm20-super-vlm2-20260507-0905b74.sqsh}"
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-false}"
export NRL_VENVS_TRUST_EXISTING="${NRL_VENVS_TRUST_EXISTING:-1}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR:-/opt/ray_venvs}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NVTE_FWD_LAYERNORM_SM_MARGIN="${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}"
export NVTE_BWD_LAYERNORM_SM_MARGIN="${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}"
export NEMO_RL_LOG_GPU_MEMORY="${NEMO_RL_LOG_GPU_MEMORY:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NRL_IGNORE_VERSION_MISMATCH="${NRL_IGNORE_VERSION_MISMATCH:-true}"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

cat > "${NEMORL}/.rayignore" <<'RAYIGNORE'
3rdparty/
.git/
.venv/
debug_results/
__pycache__/
*.egg-info/
*_logs/
logs/
RAYIGNORE

read -r -d '' SETUP_COMMAND <<SETUPEOF || true
set -euo pipefail
cd ${NEMORL}

rsync -a \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='venv/' \
  --exclude='build/' \
  --exclude='dist/' \
  --exclude='__pycache__/' \
  --exclude='.cache/' \
  --exclude='.mypy_cache/' \
  --exclude='.pytest_cache/' \
  --exclude='.ruff_cache/' \
  --include='*/' \
  --include='*.py' \
  --include='pyproject.toml' \
  --include='setup.py' \
  --include='setup.cfg' \
  --include='*.toml' \
  --include='*.yaml' \
  --include='*.jsonl' \
  --include='*.cfg' \
  --include='*.txt' \
  --include='*.egg-info/**' \
  --exclude='*' \
  ${NEMORL}/3rdparty/ \
  /opt/nemo-rl/3rdparty/

export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NRL_FORCE_REBUILD_VENVS="\${NRL_FORCE_REBUILD_VENVS:-false}"

mkdir -p /opt/nemo_rl_venv/lib/python3.12/site-packages
printf '%s\n' /opt/nemo-rl/3rdparty/Gym-workspace/Gym \
  > /opt/nemo_rl_venv/lib/python3.12/site-packages/nemo_gym_source.pth
/opt/nemo_rl_venv/bin/python -c 'import orjson, devtools, yappi, jsonlines, gprof2dot, pydot' \
  || uv pip install --python /opt/nemo_rl_venv/bin/python \
    orjson devtools yappi jsonlines gprof2dot pydot

GYM_VENV_STAMP="\$(date +%s)"
prepare_gym_venv_link() {
  local venv_path="\$1"
  mkdir -p "\$(dirname "\$venv_path")"
  if [[ -e "\$venv_path" && ! -L "\$venv_path" ]]; then
    mv "\$venv_path" "\${venv_path}.stale.\${GYM_VENV_STAMP}"
  fi
  ln -sfn /opt/nemo_rl_venv "\$venv_path"
}

prepare_gym_venv_link /opt/gym_venvs/responses_api_models/vllm_model/.venv
prepare_gym_venv_link /opt/gym_venvs/resources_servers/math_with_judge/.venv
prepare_gym_venv_link /opt/gym_venvs/resources_servers/mcqa/.venv
prepare_gym_venv_link /opt/gym_venvs/resources_servers/gui_coordinate/.venv
prepare_gym_venv_link /opt/gym_venvs/resources_servers/string_match/.venv
prepare_gym_venv_link /opt/gym_venvs/responses_api_agents/simple_agent/.venv
SETUPEOF
export SETUP_COMMAND

export COMMAND="\
set -euo pipefail
cd ${NEMORL}
export PYTHONPATH=${HOME}/.cache/huggingface/modules:${NEMORL}:${NEMORL}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${NEMORL}/3rdparty/Megatron-LM-workspace/Megatron-LM\${PYTHONPATH:+:\$PYTHONPATH}
uv run --no-sync examples/nemo_gym/run_grpo_nemo_gym.py --config examples/nemo_gym/grpo_nanov3omni.yaml \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.model_name=${MODEL_NAME} \
policy.megatron_cfg.tensor_model_parallel_size=${POLICY_TP} \
policy.tokenizer.chat_template=${POLICY_CHAT_TEMPLATE} \
policy.generation.vllm_cfg.http_server_serving_chat_kwargs.chat_template=${VLLM_CHAT_TEMPLATE} \
policy.megatron_cfg.expert_model_parallel_size=${POLICY_EP} \
policy.megatron_cfg.context_parallel_size=${POLICY_CP} \
policy.generation.vllm_cfg.tensor_parallel_size=${VLLM_TP} \
+policy.generation.vllm_cfg.logprobs_mode=raw_logprobs \
${POLICY_VLLM_LOAD_FORMAT:+policy.generation.vllm_cfg.load_format=${POLICY_VLLM_LOAD_FORMAT}} \
+policy.generation.vllm_kwargs.mm_processor_cache_gb=0 \
policy.sequence_packing.enabled=${SEQUENCE_PACKING_ENABLED} \
grpo.seed=${SEED} \
grpo.num_prompts_per_step=${NUM_PROMPTS_PER_STEP} \
grpo.num_generations_per_prompt=${NUM_GENERATIONS_PER_PROMPT} \
grpo.max_num_steps=${GRPO_MAX_NUM_STEPS} \
policy.megatron_cfg.scheduler.lr_warmup_iters=1 \
policy.train_global_batch_size=${TRAIN_GLOBAL_BATCH_SIZE} \
checkpointing.checkpoint_dir='${RESULTS_DIR}' \
logger.log_dir='${RESULTS_DIR}' \
logger.wandb_enabled=${WANDB_ENABLED} \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${JOB_NAME}' \
policy.max_total_sequence_length=16384 \
data.train.data_path=${DATA_PATH} \
data.validation.data_path=${DATA_PATH} \
+env.nemo_gym.skip_venv_if_present=true \
env.should_log_nemo_gym_responses=${NEMO_GYM_LOG_RESPONSES}"

echo "Submitting super_gym NanoV3 Gym GRPO job"
echo "  repo=${NEMORL}"
echo "  job_name=${JOB_NAME}"
echo "  nodes=${NUM_NODES}"
echo "  gpus_per_node=${GPUS_PER_NODE}"
echo "  model=${MODEL_NAME}"
echo "  data=${DATA_PATH}"
echo "  chat_template=${POLICY_CHAT_TEMPLATE}"
echo "  vllm_chat_template=${VLLM_CHAT_TEMPLATE}"
echo "  policy_tp=${POLICY_TP} policy_ep=${POLICY_EP} policy_cp=${POLICY_CP} vllm_tp=${VLLM_TP}"
echo "  prompts=${NUM_PROMPTS_PER_STEP} generations=${NUM_GENERATIONS_PER_PROMPT} train_gbs=${TRAIN_GLOBAL_BATCH_SIZE}"
echo "  vllm_load_format=${POLICY_VLLM_LOAD_FORMAT:-yaml default}"
echo "  max_steps=${GRPO_MAX_NUM_STEPS}"
echo "  sequence_packing=${SEQUENCE_PACKING_ENABLED}"

MOUNTS="/lustre:/lustre,${NEMORL}:/opt/nemo-rl" \
sbatch \
    --nodes="${NUM_NODES}" \
    --account="${SBATCH_ACCOUNT:-llmservice_fm_vision}" \
    --job-name="nemo-rl-${JOB_NAME}" \
    --partition="${SBATCH_PARTITION:-batch_block1}" \
    --dependency="${SBATCH_DEPENDENCY:-singleton}" \
    --time="${SBATCH_TIME:-4:00:00}" \
    --gres="gpu:${GPUS_PER_NODE}" \
    "ray.sub"
