#!/usr/bin/env bash

set -euo pipefail

ACTION=${ACTION:-test-only}
MODEL=${MODEL:-qwen30}
ARM=${ARM:-mxfp8}
BRANCH=${BRANCH:-sna/exp-current-bf16-mxfp8-superset-20260816}
GIT_REMOTE=${GIT_REMOTE:-origin}
EXPECTED_HEAD=${EXPECTED_HEAD:-}
BASE=${BASE:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna}
REPO=${REPO:-${BASE}/RL-exp-current-bf16-mxfp8-superset-20260816}
CONTAINER=${CONTAINER:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/mkar/containers/nemo-rl-nightly-ngc-20260815_212622.sqsh}
RUNTIME_SITE_PACKAGES=${RUNTIME_SITE_PACKAGES:-}
HF_HOME=${HF_HOME:-${BASE}/hf_home}
RUN_SUFFIX=${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-${BASE}/experiments/current-mxfp8-moe-rollout-ab/${MODEL}/${ARM}/${RUN_SUFFIX}}
CACHE_ROOT=${CACHE_ROOT:-${BASE}/.cache/current-mxfp8-moe-rollout-ab/${MODEL}/${ARM}/${RUN_SUFFIX}}
MOUNT_UV_CACHE=${MOUNT_UV_CACHE:-true}
USE_CONTAINER_PYTHON=${USE_CONTAINER_PYTHON:-false}
WORKER_VENV_ROOT=${WORKER_VENV_ROOT:-/tmp/nemo_rl_worker_venvs/${MODEL}/${ARM}/${RUN_SUFFIX}}
WANDB_PROJECT=${WANDB_PROJECT:-sna-current-mxfp8-moe-rollout-ab}
WANDB_NAME=${WANDB_NAME:-${MODEL}-${ARM}-20step-${RUN_SUFFIX}}
WANDB_ENABLED=${WANDB_ENABLED:-true}
LOGPROB_MODE=${LOGPROB_MODE:-recipe}
ACCOUNT=${SLURM_ACCOUNT:-nemotron_sw_post}
PARTITION=${PARTITION:-batch}
CLUSTER_LABEL=${CLUSTER_LABEL:-oci-hsg-cs-001}
TOTAL_NODES=${TOTAL_NODES:-}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
SLURM_USE_GRES=${SLURM_USE_GRES:-auto}
SLURM_USE_SEGMENT=${SLURM_USE_SEGMENT:-true}
GEN_NODES=${GEN_NODES:-}
SEGMENT_SIZE=${SEGMENT_SIZE:-}
VLLM_TP=${VLLM_TP:-}
VLLM_PP=${VLLM_PP:-1}
MAX_STEPS=${MAX_STEPS:-20}
JOB_NAME=${JOB_NAME:-${ACCOUNT}-mxfp8-ab.${MODEL}-${ARM}-${MAX_STEPS}s}
WALLTIME=${WALLTIME:-04:00:00}
SLURM_DEPENDENCY=${SLURM_DEPENDENCY:-}
IDLE_REAPER_EXEMPT_MINS=${IDLE_REAPER_EXEMPT_MINS:-120}
IDLE_REAPER_EXEMPT_REASON=${IDLE_REAPER_EXEMPT_REASON:-model_loading}
IDLE_REAPER_EXEMPT_DESCRIPTION=${IDLE_REAPER_EXEMPT_DESCRIPTION:-NeMo-RL environment build, model load, FlashInfer autotuning, and CUDA Graph capture}

case "${MODEL}:${ARM}" in
  qwen30:bf16)
    CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml
    MODEL_LABEL=Qwen3-30B-A3B
    TRAIN_GLOBAL_BATCH_SIZE=2048
    ROLLOUT_PRECISION=bf16
    QUANTIZATION_SCOPE=none
    TOTAL_NODES=${TOTAL_NODES:-8}
    GEN_NODES=${GEN_NODES:-4}
    SEGMENT_SIZE=${SEGMENT_SIZE:-4}
    VLLM_TP=${VLLM_TP:-1}
    ;;
  qwen30:mxfp8)
    CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml
    MODEL_LABEL=Qwen3-30B-A3B
    TRAIN_GLOBAL_BATCH_SIZE=2048
    ROLLOUT_PRECISION=mxfp8
    QUANTIZATION_SCOPE=routed_expert_fc1_fc2_only
    TOTAL_NODES=${TOTAL_NODES:-8}
    GEN_NODES=${GEN_NODES:-4}
    SEGMENT_SIZE=${SEGMENT_SIZE:-4}
    VLLM_TP=${VLLM_TP:-1}
    ;;
  qwen235:mxfp8)
    CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g-mxfp8-rollout.yaml
    MODEL_LABEL=Qwen3-235B-A22B
    TRAIN_GLOBAL_BATCH_SIZE=512
    ROLLOUT_PRECISION=mxfp8
    QUANTIZATION_SCOPE=routed_expert_fc1_fc2_only
    TOTAL_NODES=${TOTAL_NODES:-32}
    GEN_NODES=${GEN_NODES:-16}
    SEGMENT_SIZE=${SEGMENT_SIZE:-4}
    VLLM_TP=${VLLM_TP:-4}
    ;;
  nano:bf16)
    CONFIG=examples/configs/recipes/llm/performance/grpo-nanov3-30ba3b-8n4g-bf16-rollout-nccl.yaml
    MODEL_LABEL=Nemotron3-Nano-30B-A3B
    TRAIN_GLOBAL_BATCH_SIZE=16
    ROLLOUT_PRECISION=bf16
    QUANTIZATION_SCOPE=none
    TOTAL_NODES=${TOTAL_NODES:-8}
    GEN_NODES=${GEN_NODES:-4}
    SEGMENT_SIZE=${SEGMENT_SIZE:-4}
    VLLM_TP=${VLLM_TP:-1}
    ;;
  nano:mxfp8)
    CONFIG=examples/configs/recipes/llm/performance/grpo-nanov3-30ba3b-8n4g-mxfp8-rollout-nccl.yaml
    MODEL_LABEL=Nemotron3-Nano-30B-A3B
    TRAIN_GLOBAL_BATCH_SIZE=16
    ROLLOUT_PRECISION=mxfp8
    QUANTIZATION_SCOPE=routed_expert_fc1_fc2_only
    TOTAL_NODES=${TOTAL_NODES:-8}
    GEN_NODES=${GEN_NODES:-4}
    SEGMENT_SIZE=${SEGMENT_SIZE:-4}
    VLLM_TP=${VLLM_TP:-1}
    ;;
  *)
    echo "Supported MODEL:ARM pairs are qwen30:bf16, qwen30:mxfp8, qwen235:mxfp8, nano:bf16, and nano:mxfp8" >&2
    exit 2
    ;;
esac

case "${LOGPROB_MODE}" in
  recipe)
    LOGPROB_OVERRIDES="grpo.seed=42"
    ;;
  full)
    LOGPROB_OVERRIDES="grpo.seed=42 loss_fn.force_on_policy_ratio=false loss_fn.use_importance_sampling_correction=true"
    ;;
  *)
    echo "LOGPROB_MODE must be recipe or full" >&2
    exit 2
    ;;
esac

case "${ACTION}" in
  submit) SBATCH_ACTION=() ;;
  test-only) SBATCH_ACTION=(--test-only) ;;
  *) echo "ACTION must be submit or test-only" >&2; exit 2 ;;
esac

git -C "${REPO}" fetch "${GIT_REMOTE}" "${BRANCH}"
git -C "${REPO}" pull --ff-only "${GIT_REMOTE}" "${BRANCH}"
LOCAL_HEAD=$(git -C "${REPO}" rev-parse HEAD)
REMOTE_HEAD=$(git -C "${REPO}" rev-parse "${GIT_REMOTE}/${BRANCH}")
if [[ "${LOCAL_HEAD}" != "${REMOTE_HEAD}" ]]; then
  echo "Local and remote heads differ: local=${LOCAL_HEAD}, remote=${REMOTE_HEAD}" >&2
  exit 2
fi
if [[ -n "${EXPECTED_HEAD}" && "${LOCAL_HEAD}" != "${EXPECTED_HEAD}" ]]; then
  echo "Expected HEAD ${EXPECTED_HEAD}; got ${LOCAL_HEAD}" >&2
  exit 2
fi
test -z "$(git -C "${REPO}" status --porcelain --untracked-files=no)"
if git -C "${REPO}" submodule status --recursive | grep -q '^-'; then
  echo "All pinned submodules must be initialized" >&2
  exit 2
fi

for path in "${REPO}/${CONFIG}" "${REPO}/ray.sub" "${CONTAINER}" "${HF_HOME}"; do
  test -e "${path}"
done
if [[ -n "${RUNTIME_SITE_PACKAGES}" ]]; then
  test -f "${RUNTIME_SITE_PACKAGES}/urllib3/exceptions.py"
  test -f "${RUNTIME_SITE_PACKAGES}/nvidia/cudnn/lib/libcudnn.so.9"
fi

mkdir -p "${EXPERIMENT_ROOT}" "${CACHE_ROOT}"
WANDB_KEY_FILE=${CACHE_ROOT}/.wandb_key
if [[ "${WANDB_ENABLED}" == true && -f "${HOME}/.netrc" ]]; then
  umask 077
  awk '
    {
      for (i = 1; i <= NF; i++) {
        if ($i == "machine" && $(i + 1) == "api.wandb.ai") found = 1
        if (found && $i == "password") { print $(i + 1); exit }
      }
    }
  ' "${HOME}/.netrc" >"${WANDB_KEY_FILE}"
elif [[ "${WANDB_ENABLED}" == true && -n "${WANDB_API_KEY:-}" ]]; then
  umask 077
  printf '%s\n' "${WANDB_API_KEY}" >"${WANDB_KEY_FILE}"
fi
if [[ "${WANDB_ENABLED}" == true ]]; then
  test -s "${WANDB_KEY_FILE}"
  WANDB_KEY_SETUP="export WANDB_API_KEY=\"\$(cat ${WANDB_KEY_FILE})\""
elif [[ "${WANDB_ENABLED}" == false ]]; then
  WANDB_KEY_SETUP=:
else
  echo "WANDB_ENABLED must be true or false" >&2
  exit 2
fi
if [[ -n "${RUNTIME_SITE_PACKAGES}" ]]; then
  PYTHON_RUNNER=/opt/nemo_rl_venv/bin/python
  UV_DRIVER_SETUP="unset UV_PROJECT_ENVIRONMENT UV_PYTHON_INSTALL_DIR
export UV_PYTHON=/opt/nemo_rl_venv/bin/python
export UV_NO_MANAGED_PYTHON=1"
else
  PYTHON_RUNNER="uv run --frozen"
  UV_DRIVER_SETUP="export UV_PROJECT_ENVIRONMENT=${CACHE_ROOT}/driver-venv
export UV_PYTHON_INSTALL_DIR=${CACHE_ROOT}/uv-python"
  if [[ "${USE_CONTAINER_PYTHON}" == true ]]; then
    UV_DRIVER_SETUP+="
export UV_PYTHON=/opt/nemo_rl_venv/bin/python
export UV_NO_MANAGED_PYTHON=1"
  elif [[ "${USE_CONTAINER_PYTHON}" != false ]]; then
    echo "USE_CONTAINER_PYTHON must be true or false" >&2
    exit 2
  fi
fi

cat >"${EXPERIMENT_ROOT}/metadata.env" <<EOF
repo=${REPO}
repo_sha=${LOCAL_HEAD}
branch=${BRANCH}
config=${CONFIG}
container=${CONTAINER}
runtime_site_packages=${RUNTIME_SITE_PACKAGES:-container_default}
python_runner=${PYTHON_RUNNER}
mount_uv_cache=${MOUNT_UV_CACHE}
use_container_python=${USE_CONTAINER_PYTHON}
worker_venv_root=${WORKER_VENV_ROOT}
python_version=runtime_interpreter
gym_source_commit=5a6fc589c0196f73a5931781b06da61f668a80d7
cluster=${CLUSTER_LABEL}
hardware=GB200
model=${MODEL_LABEL}
arm=${ARM}
total_nodes=${TOTAL_NODES}
gpus_per_node=${GPUS_PER_NODE}
slurm_use_gres=${SLURM_USE_GRES}
trainer_nodes=$((TOTAL_NODES - GEN_NODES))
generation_nodes=${GEN_NODES}
training_precision=bf16
rollout_precision=${ROLLOUT_PRECISION}
quantization_scope=${QUANTIZATION_SCOPE}
moe_backend=flashinfer_trtllm
refit_transport=nccl_reshard
cuda_graphs=enabled
logprob_mode=${LOGPROB_MODE}
max_steps=${MAX_STEPS}
slurm_dependency=${SLURM_DEPENDENCY:-none}
idle_reaper_exempt_minutes=${IDLE_REAPER_EXEMPT_MINS}
seed=42
wandb_project=${WANDB_PROJECT}
wandb_name=${WANDB_NAME}
wandb_enabled=${WANDB_ENABLED}
EOF

COMMAND=$(cat <<EOF
set -euo pipefail
cd ${REPO}
export HF_HOME=${HF_HOME}
export HF_DATASETS_CACHE=${HF_HOME}/cache
export NCCL_NVLS_ENABLE=0
export RAY_CGRAPH_get_timeout=2400
export NRL_FORCE_REBUILD_VENVS=true
export NEMO_RL_VENV_DIR=${WORKER_VENV_ROOT}
export NVTE_CUDA_ARCHS=100
export PYTHONPATH=${REPO}
export TORCH_CUDA_ARCH_LIST=10.0
export UV_CACHE_DIR=/root/.cache/uv
${UV_DRIVER_SETUP}
export UV_LOCK_TIMEOUT=7200
${WANDB_KEY_SETUP}
printf 'NEMO_RL_SOURCE_COMMIT=%s\n' "\$(git rev-parse HEAD)"
${PYTHON_RUNNER} examples/run_grpo.py \
  --config ${CONFIG} \
  cluster.num_nodes=${TOTAL_NODES} \
  cluster.gpus_per_node=${GPUS_PER_NODE} \
  cluster.segment_size=${SEGMENT_SIZE} \
  policy.generation.colocated.enabled=false \
  policy.generation.colocated.resources.num_nodes=${GEN_NODES} \
  policy.generation.colocated.resources.gpus_per_node=${GPUS_PER_NODE} \
  policy.generation.refit_transport=nccl_reshard \
  policy.megatron_cfg.expert_tensor_parallel_size=1 \
  policy.generation.vllm_cfg.tensor_parallel_size=${VLLM_TP} \
  policy.generation.vllm_cfg.pipeline_parallel_size=${VLLM_PP} \
  policy.generation.vllm_cfg.expert_parallel_size=1 \
  policy.generation.vllm_cfg.async_engine=false \
  policy.generation.vllm_cfg.enforce_eager=false \
  policy.generation.vllm_cfg.use_tqdm=false \
  policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm \
  ++policy.generation.vllm_kwargs.distributed_timeout_seconds=2400 \
  policy.train_global_batch_size=${TRAIN_GLOBAL_BATCH_SIZE} \
  grpo.max_num_steps=${MAX_STEPS} \
  ${LOGPROB_OVERRIDES} \
  grpo.val_at_start=false \
  ++grpo.val_at_end=false \
  checkpointing.enabled=false \
  logger.log_dir=${EXPERIMENT_ROOT}/logs \
  logger.wandb_enabled=${WANDB_ENABLED} \
  logger.tensorboard_enabled=true \
  logger.monitor_gpus=true \
  ++logger.wandb.entity=nvidia \
  logger.wandb.project=${WANDB_PROJECT} \
  logger.wandb.name=${WANDB_NAME}
${PYTHON_RUNNER} tests/json_dump_tb_logs.py ${EXPERIMENT_ROOT}/logs \
  --output_path ${EXPERIMENT_ROOT}/metrics.json || true
EOF
)

export BASE_LOG_DIR=${EXPERIMENT_ROOT}
export COMMAND
export CONTAINER
export CONTAINER_REMAP_ROOT=1
export GPUS_PER_NODE
export MOUNTS=/lustre:/lustre
if [[ -n "${RUNTIME_SITE_PACKAGES}" ]]; then
  MOUNTS+=",${RUNTIME_SITE_PACKAGES}:/opt/nemo_rl_venv/lib/python3.13/site-packages"
fi
case "${MOUNT_UV_CACHE}" in
  true) export UV_CACHE_DIR_OVERRIDE=${CACHE_ROOT}/uv-cache ;;
  false) unset UV_CACHE_DIR_OVERRIDE ;;
  *) echo "MOUNT_UV_CACHE must be true or false" >&2; exit 2 ;;
esac

SBATCH_ARGS=(
  --nodes="${TOTAL_NODES}"
  --exclusive
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --time="${WALLTIME}"
  --job-name="${JOB_NAME}"
  --output="${EXPERIMENT_ROOT}/slurm-%j.out"
  --comment="{\"OccupiedIdleGPUsJobReaper\":{\"exemptIdleTimeMins\":\"${IDLE_REAPER_EXEMPT_MINS}\",\"reason\":\"${IDLE_REAPER_EXEMPT_REASON}\",\"description\":\"${IDLE_REAPER_EXEMPT_DESCRIPTION}\"}}"
)
case "${SLURM_USE_SEGMENT}" in
  true) SBATCH_ARGS+=(--segment="${SEGMENT_SIZE}") ;;
  false) ;;
  *) echo "SLURM_USE_SEGMENT must be true or false" >&2; exit 2 ;;
esac
case "${SLURM_USE_GRES}" in
  true) SBATCH_ARGS+=(--gres="gpu:${GPUS_PER_NODE}") ;;
  false) ;;
  auto)
    if sinfo -p "${PARTITION}" -h -o '%G' | grep -q 'gpu:'; then
      SBATCH_ARGS+=(--gres="gpu:${GPUS_PER_NODE}")
    fi
    ;;
  *) echo "SLURM_USE_GRES must be auto, true, or false" >&2; exit 2 ;;
esac
if [[ -n "${SLURM_DEPENDENCY}" ]]; then
  SBATCH_ARGS+=(--dependency="${SLURM_DEPENDENCY}")
fi

printf 'action=%s\nmodel=%s\narm=%s\nsha=%s\nresult=%s\n' \
  "${ACTION}" "${MODEL}" "${ARM}" "${LOCAL_HEAD}" "${EXPERIMENT_ROOT}"
exec sbatch "${SBATCH_ACTION[@]}" "${SBATCH_ARGS[@]}" "${REPO}/ray.sub"
