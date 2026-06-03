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

declare -A CALLER_ENV_VARS=()
while IFS='=' read -r name value; do
  if [[ "${name}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
    CALLER_ENV_VARS["${name}"]="${value}"
  fi
done < <(env)

set -a
if [[ -f /lustre/fsw/portfolios/llmservice/users/smohsenitahe/.env ]]; then
  # shellcheck disable=SC1091
  source /lustre/fsw/portfolios/llmservice/users/smohsenitahe/.env
elif [[ -f "${NEMORL}/.env" ]]; then
  # shellcheck disable=SC1091
  source "${NEMORL}/.env"
fi
set +a

for name in "${!CALLER_ENV_VARS[@]}"; do
  export "${name}=${CALLER_ENV_VARS[${name}]}"
done
unset CALLER_ENV_VARS
set -u

if [[ -n "${GITLAB_FLASHINFER_TOKEN:-}" ]]; then
  export UV_INDEX_FLASHINFER_INTERNAL_PYPI_USERNAME="${UV_INDEX_FLASHINFER_INTERNAL_PYPI_USERNAME:-__token__}"
  export UV_INDEX_FLASHINFER_INTERNAL_PYPI_PASSWORD="${UV_INDEX_FLASHINFER_INTERNAL_PYPI_PASSWORD:-${GITLAB_FLASHINFER_TOKEN}}"
fi

NUM_NODES="${NUM_NODES:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NUM_GEN_NODES="${NUM_GEN_NODES:-1}"
JOB_NAME="${JOB_NAME:-async-grpo-vlm}"
SEED="${SEED:-42}"

MODEL_NAME="${MODEL_NAME:-${IMAGE_GRPO_MODEL_NAME:-/lustre/fs1/portfolios/coreai/users/aroshanghias/checkpoints/sft_omni_300k_128_nodes_rebalanced_no_long_omni_0301_tp_1_hf}}"
POLICY_CHAT_TEMPLATE="${POLICY_CHAT_TEMPLATE:-/lustre/fs1/portfolios/coreai/users/aroshanghias/checkpoints/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/chat_template.jinja}"
VLLM_CHAT_TEMPLATE="${VLLM_CHAT_TEMPLATE:-${POLICY_CHAT_TEMPLATE}}"
DATA_PATH="${DATA_PATH:-/lustre/fs1/portfolios/coreai/users/aroshanghias/data/nemo_gym_dfw/mix_text_vision_2k_dfw_image_only_filtered.jsonl}"

NUM_PROMPTS_PER_STEP="${NUM_PROMPTS_PER_STEP:-15}"
NUM_GENERATIONS_PER_PROMPT="${NUM_GENERATIONS_PER_PROMPT:-16}"
TRAIN_GLOBAL_BATCH_SIZE="${TRAIN_GLOBAL_BATCH_SIZE:-240}"
GRPO_MAX_NUM_STEPS="${GRPO_MAX_NUM_STEPS:-100000}"
GRPO_MAX_NUM_EPOCHS="${GRPO_MAX_NUM_EPOCHS:-1000}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-10}"
LR_DECAY_ITERS="${LR_DECAY_ITERS:-${GRPO_MAX_NUM_STEPS}}"

POLICY_TP="${POLICY_TP:-8}"
POLICY_EP="${POLICY_EP:-8}"
POLICY_CP="${POLICY_CP:-1}"
VLLM_TP="${VLLM_TP:-2}"
POLICY_VLLM_LOAD_FORMAT="${POLICY_VLLM_LOAD_FORMAT:-}"
SEQUENCE_PACKING_ENABLED="${SEQUENCE_PACKING_ENABLED:-false}"

RESULTS_DIR="${RESULTS_DIR:-results/${JOB_NAME}}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-async-grpo-compare}"
WANDB_ENTITY="${WANDB_ENTITY:-nvidia}"
NEMO_GYM_LOG_RESPONSES="${NEMO_GYM_LOG_RESPONSES:-false}"
DEDUPLICATE_MULTIMODAL_DATA="${DEDUPLICATE_MULTIMODAL_DATA:-false}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

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
export HF_HOME="${HF_HOME:-/lustre/fs1/portfolios/coreai/users/aroshanghias/.cache/huggingface}"

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
export NRL_PATCH_EXISTING_VENVS="\${NRL_PATCH_EXISTING_VENVS:-true}"

if [[ "\${NRL_FORCE_REBUILD_VENVS}" == "true" && "\${NRL_PATCH_EXISTING_VENVS}" != "true" ]]; then
  rm -rf /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker
  rm -rf /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
  rm -rf /opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym
  rm -rf /opt/ray_venvs/nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector
  rm -rf /opt/ray_venvs/nemo_rl.algorithms.async_utils.ReplayBuffer
fi

patch_python_file_into_venv() {
  local venv_path="\$1"
  local rel_path="\$2"
  if [[ ! -d "\${venv_path}" ]]; then
    return
  fi
  local site_packages
  site_packages="\$(find "\${venv_path}/lib" -maxdepth 2 -type d -name site-packages | head -n 1)"
  if [[ -z "\${site_packages}" ]]; then
    return
  fi
  if [[ -f "\${NEMORL}/\${rel_path}" && -f "\${site_packages}/\${rel_path}" ]]; then
    cp "\${NEMORL}/\${rel_path}" "\${site_packages}/\${rel_path}"
  fi
}

if [[ "\${NRL_PATCH_EXISTING_VENVS}" == "true" ]]; then
  patch_python_file_into_venv \
    /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker \
    nemo_rl/models/generation/vllm/vllm_worker_async.py
  patch_python_file_into_venv \
    /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker \
    nemo_rl/models/generation/vllm/vllm_backend.py
  patch_python_file_into_venv \
    /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker \
    nemo_rl/utils/packed_tensor.py
  patch_python_file_into_venv \
    /opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym \
    nemo_rl/environments/nemo_gym.py
  if [[ -f /opt/nemo-rl/3rdparty/Gym-workspace/Gym/nemo_gym/server_utils.py ]]; then
    cp \
      "\${NEMORL}/3rdparty/Gym-workspace/Gym/nemo_gym/server_utils.py" \
      /opt/nemo-rl/3rdparty/Gym-workspace/Gym/nemo_gym/server_utils.py
  fi
fi

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

HF_MODULES_PATH="${HF_MODULES_PATH:-${HF_HOME}/modules}"

export COMMAND="\
set -euo pipefail
cd ${NEMORL}
export HF_HOME=${HF_HOME}
export PYTHONPATH=${HF_MODULES_PATH}:${NEMORL}:${NEMORL}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${NEMORL}/3rdparty/Megatron-LM-workspace/Megatron-LM\${PYTHONPATH:+:\$PYTHONPATH}
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
policy.generation.vllm_cfg.async_engine=true \
policy.generation.colocated.enabled=false \
policy.generation.colocated.resources.num_nodes=${NUM_GEN_NODES} \
policy.generation.colocated.resources.gpus_per_node=${GPUS_PER_NODE} \
policy.sequence_packing.enabled=${SEQUENCE_PACKING_ENABLED} \
grpo.async_grpo.enabled=true \
grpo.async_grpo.max_trajectory_age_steps=2 \
+grpo.async_grpo.in_flight_weight_updates=true \
+grpo.async_grpo.recompute_kv_cache_after_weight_updates=false \
grpo.deduplicate_multimodal_data=${DEDUPLICATE_MULTIMODAL_DATA} \
+grpo.debug_payload_metrics=true \
loss_fn.use_importance_sampling_correction=true \
grpo.seed=${SEED} \
grpo.num_prompts_per_step=${NUM_PROMPTS_PER_STEP} \
grpo.num_generations_per_prompt=${NUM_GENERATIONS_PER_PROMPT} \
grpo.max_num_steps=${GRPO_MAX_NUM_STEPS} \
grpo.max_num_epochs=${GRPO_MAX_NUM_EPOCHS} \
grpo.val_period=100 \
policy.megatron_cfg.scheduler.lr_warmup_iters=${LR_WARMUP_ITERS} \
policy.megatron_cfg.scheduler.lr_decay_iters=${LR_DECAY_ITERS} \
policy.train_global_batch_size=${TRAIN_GLOBAL_BATCH_SIZE} \
checkpointing.enabled=false \
checkpointing.checkpoint_dir='${RESULTS_DIR}' \
logger.log_dir='${RESULTS_DIR}' \
logger.wandb_enabled=${WANDB_ENABLED} \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${JOB_NAME}' \
+logger.wandb.entity='${WANDB_ENTITY}' \
logger.monitor_gpus=true \
policy.max_total_sequence_length=16384 \
data.train.data_path=${DATA_PATH} \
data.validation.data_path=${DATA_PATH} \
+env.nemo_gym.skip_venv_if_present=true \
env.should_log_nemo_gym_responses=${NEMO_GYM_LOG_RESPONSES} \
${EXTRA_OVERRIDES}"

echo "Submitting async NanoV3 Gym GRPO job"
echo "  repo=${NEMORL}"
echo "  job_name=${JOB_NAME}"
echo "  nodes=${NUM_NODES} (gen=${NUM_GEN_NODES}, train=$((NUM_NODES - NUM_GEN_NODES)))"
echo "  gpus_per_node=${GPUS_PER_NODE}"
echo "  prompts=${NUM_PROMPTS_PER_STEP} generations=${NUM_GENERATIONS_PER_PROMPT} train_gbs=${TRAIN_GLOBAL_BATCH_SIZE}"
echo "  max_steps=${GRPO_MAX_NUM_STEPS} max_epochs=${GRPO_MAX_NUM_EPOCHS}"
echo "  in_flight_weight_updates=true max_trajectory_age=2"
echo "  wandb=${WANDB_ENABLED} entity=${WANDB_ENTITY} project=${WANDB_PROJECT}"
echo "  data=${DATA_PATH}"

MOUNTS="${MOUNTS:-/lustre:/lustre},${NEMORL}:/opt/nemo-rl" \
sbatch \
    --nodes="${NUM_NODES}" \
    --account="${SBATCH_ACCOUNT:-coreai_dlalgo_nemorl}" \
    --job-name="nemo-rl-${JOB_NAME}" \
    --partition="${SBATCH_PARTITION:-batch}" \
    --dependency="${SBATCH_DEPENDENCY:-singleton}" \
    --time="${SBATCH_TIME:-4:00:00}" \
    --gres="gpu:${GPUS_PER_NODE}" \
    "ray.sub"
