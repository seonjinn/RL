#!/usr/bin/env bash

set -euo pipefail

: "${ARM:?ARM is required: baseline or optimized}"
: "${MODE:?MODE is required: bf16, blockwise-fp8, mxfp8-rollout, or mxfp8-probe}"
: "${REPO:?REPO is required}"
: "${CONTAINER:?CONTAINER is required}"
: "${TOTAL_NODES:?TOTAL_NODES is required}"
: "${GPUS_PER_NODE:?GPUS_PER_NODE is required}"
: "${GEN_NODES:?GEN_NODES is required}"
: "${SEGMENT_SIZE:?SEGMENT_SIZE is required}"
: "${MAX_STEPS:?MAX_STEPS is required}"
: "${RUN_NAME:?RUN_NAME is required}"
: "${EXPERIMENT_ROOT:?EXPERIMENT_ROOT is required}"

RAY_SUB_PATH=${RAY_SUB_PATH:-${REPO}/ray.sub}
CONFIG_BF16=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml
CONFIG_FP8=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml
REFIT_TRANSPORT=${REFIT_TRANSPORT:-}

case "${ARM}" in
  baseline)
    export NRL_MXFP8_BATCHED_SHUFFLE=0
    export NRL_REFIT_CACHED_LOADERS=0
    ;;
  optimized)
    export NRL_MXFP8_BATCHED_SHUFFLE=1
    export NRL_REFIT_CACHED_LOADERS=1
    ;;
  *)
    echo "ARM must be baseline or optimized" >&2
    exit 2
    ;;
esac

case "${MODE}" in
  bf16)
    CONFIG=${CONFIG_BF16}
    PRECISION_OVERRIDES="policy.megatron_cfg.fp8_cfg.enabled=false policy.megatron_cfg.fp8_cfg.fp8_param=false ++policy.generation.vllm_cfg.precision=bfloat16 ++policy.generation.vllm_cfg.is_mx=false"
    ;;
  blockwise-fp8)
    CONFIG=${CONFIG_FP8}
    PRECISION_OVERRIDES="policy.megatron_cfg.fp8_cfg.enabled=true policy.megatron_cfg.fp8_cfg.fp8=e4m3 policy.megatron_cfg.fp8_cfg.fp8_recipe=blockwise policy.megatron_cfg.fp8_cfg.fp8_param=true policy.generation.vllm_cfg.precision=fp8 policy.generation.vllm_cfg.is_mx=false"
    ;;
  mxfp8-rollout)
    CONFIG=${CONFIG_FP8}
    REFIT_TRANSPORT=${REFIT_TRANSPORT:-null}
    PRECISION_OVERRIDES="policy.megatron_cfg.fp8_cfg.enabled=false policy.megatron_cfg.fp8_cfg.fp8_param=false policy.generation.vllm_cfg.precision=fp8 policy.generation.vllm_cfg.is_mx=true"
    ;;
  mxfp8-probe)
    CONFIG=${CONFIG_FP8}
    PRECISION_OVERRIDES="policy.megatron_cfg.fp8_cfg.enabled=true policy.megatron_cfg.fp8_cfg.fp8=e4m3 policy.megatron_cfg.fp8_cfg.fp8_recipe=blockwise policy.megatron_cfg.fp8_cfg.fp8_param=true policy.generation.vllm_cfg.precision=fp8 policy.generation.vllm_cfg.is_mx=true"
    ;;
  *)
    echo "MODE must be bf16, blockwise-fp8, mxfp8-rollout, or mxfp8-probe" >&2
    exit 2
    ;;
esac
REFIT_TRANSPORT=${REFIT_TRANSPORT:-nccl_reshard}

git -C "${REPO}" rev-parse --git-dir >/dev/null
test -f "${REPO}/${CONFIG}"
test -f "${CONTAINER}"
test -f "${RAY_SUB_PATH}"
mkdir -p "${EXPERIMENT_ROOT}" "${EXPERIMENT_ROOT}/logs"

WORK_ROOT=${WORK_ROOT:-/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna}
HF_HOME=${HF_HOME:-${WORK_ROOT}/.cache/huggingface}
CACHE_ROOT=${CACHE_ROOT:-${WORK_ROOT}/mopd_nano_fast/.cache/nccl-reshard-pr3294/${ARM}-${MODE}}
SHARED_UV_CACHE=${SHARED_UV_CACHE:-${CACHE_ROOT}/uv}
WANDB_PROJECT=${WANDB_PROJECT:-sna-pr3294-nccl-reshard}
WANDB_KEY_FILE=${WANDB_KEY_FILE:-${CACHE_ROOT}/.wandb_key}
NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS:-false}
RAY_BOOTSTRAP_VENV=${RAY_BOOTSTRAP_VENV:-}
RAY_BOOTSTRAP_ARCHIVE=${RAY_BOOTSTRAP_ARCHIVE:-}
RAY_BOOTSTRAP_LOCAL_ROOT=${RAY_BOOTSTRAP_LOCAL_ROOT:-}
USE_CONTAINER_VENVS=${USE_CONTAINER_VENVS:-false}
CONTAINER_DRIVER_VENV=${CONTAINER_DRIVER_VENV:-/opt/nemo_rl_venv}
CONTAINER_ACTOR_VENV_ROOT=${CONTAINER_ACTOR_VENV_ROOT:-/opt/ray_venvs}

if [[ -n "${RAY_BOOTSTRAP_ARCHIVE}" ]]; then
  test -f "${RAY_BOOTSTRAP_ARCHIVE}"
  RAY_BOOTSTRAP_LOCAL_ROOT=${RAY_BOOTSTRAP_LOCAL_ROOT:-/tmp/nrl-ray-bootstrap-${SLURM_JOB_ID}}
  RAY_BOOTSTRAP_VENV=${RAY_BOOTSTRAP_LOCAL_ROOT}
  export SETUP_COMMAND="
set -euo pipefail
if [[ ! -x '${RAY_BOOTSTRAP_LOCAL_ROOT}/bin/ray' ]]; then
  rm -rf '${RAY_BOOTSTRAP_LOCAL_ROOT}' '${RAY_BOOTSTRAP_LOCAL_ROOT}.tmp'
  mkdir -p '${RAY_BOOTSTRAP_LOCAL_ROOT}.tmp'
  tar -xzf '${RAY_BOOTSTRAP_ARCHIVE}' \
    -C '${RAY_BOOTSTRAP_LOCAL_ROOT}.tmp' --strip-components=1
  mv '${RAY_BOOTSTRAP_LOCAL_ROOT}.tmp' '${RAY_BOOTSTRAP_LOCAL_ROOT}'
fi
test -x '${RAY_BOOTSTRAP_LOCAL_ROOT}/bin/ray'
test -x '${RAY_BOOTSTRAP_LOCAL_ROOT}/bin/uv'
"
fi

mkdir -p "${HF_HOME}"
if [[ "${USE_CONTAINER_VENVS}" != true ]]; then
  mkdir -p "${SHARED_UV_CACHE}" "${CACHE_ROOT}/venvs"
fi
if [[ ! -s "${WANDB_KEY_FILE}" && -f "${HOME}/.netrc" ]]; then
  mkdir -p "$(dirname "${WANDB_KEY_FILE}")"
  (umask 077; awk '/machine api.wandb.ai/{f=1} f&&/password/{print $2; exit}' \
    "${HOME}/.netrc" >"${WANDB_KEY_FILE}")
fi

cat >"${EXPERIMENT_ROOT}/metadata.env" <<EOF
arm=${ARM}
mode=${MODE}
repo=${REPO}
repo_sha=$(git -C "${REPO}" rev-parse HEAD)
container=${CONTAINER}
total_nodes=${TOTAL_NODES}
gpus_per_node=${GPUS_PER_NODE}
generation_nodes=${GEN_NODES}
max_steps=${MAX_STEPS}
refit_transport=${REFIT_TRANSPORT}
ray_sub_path=${RAY_SUB_PATH}
ray_bootstrap_venv=${RAY_BOOTSTRAP_VENV}
ray_bootstrap_archive=${RAY_BOOTSTRAP_ARCHIVE}
shared_uv_cache=${SHARED_UV_CACHE}
force_rebuild_venvs=${NRL_FORCE_REBUILD_VENVS}
use_container_venvs=${USE_CONTAINER_VENVS}
container_driver_venv=${CONTAINER_DRIVER_VENV}
container_actor_venv_root=${CONTAINER_ACTOR_VENV_ROOT}
NRL_MXFP8_BATCHED_SHUFFLE=${NRL_MXFP8_BATCHED_SHUFFLE}
NRL_REFIT_CACHED_LOADERS=${NRL_REFIT_CACHED_LOADERS}
EOF

export NRL_REFIT_NUM_STREAMS=${NRL_REFIT_NUM_STREAMS:-2}
export CONTAINER
export MOUNTS=/lustre:/lustre
export BASE_LOG_DIR="${EXPERIMENT_ROOT}"
if [[ -n "${RAY_BOOTSTRAP_VENV}" ]]; then
  if [[ -z "${RAY_BOOTSTRAP_ARCHIVE}" ]]; then
    test -x "${RAY_BOOTSTRAP_VENV}/bin/ray"
  fi
  export PATH="${RAY_BOOTSTRAP_VENV}/bin:${PATH}"
  export PYTHONPATH="${RAY_BOOTSTRAP_VENV}/lib/python3.13/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
  UV_PYTHON_COMMAND="export UV_PYTHON='${RAY_BOOTSTRAP_VENV}/bin/python3.13'"
else
  UV_PYTHON_COMMAND=
fi

case "${USE_CONTAINER_VENVS}" in
  true)
    unset UV_CACHE_DIR_OVERRIDE
    VLLM_ACTOR_PYTHON="${CONTAINER_ACTOR_VENV_ROOT}/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python"
    MCORE_ACTOR_PYTHON="${CONTAINER_ACTOR_VENV_ROOT}/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python"
    export SETUP_COMMAND="${SETUP_COMMAND:-}
set -euo pipefail
test -x '${CONTAINER_DRIVER_VENV}/bin/python'
test -x '${VLLM_ACTOR_PYTHON}'
test -x '${MCORE_ACTOR_PYTHON}'
"
    if [[ -n "${RAY_BOOTSTRAP_VENV}" ]]; then
      DRIVER_PYTHONPATH="${REPO}:${RAY_BOOTSTRAP_VENV}/lib/python3.13/site-packages"
    else
      DRIVER_PYTHONPATH="${REPO}"
    fi
    DRIVER_ENV_COMMANDS="
export NEMO_RL_VENV_DIR='${CONTAINER_ACTOR_VENV_ROOT}'
export UV_PROJECT_ENVIRONMENT='${CONTAINER_DRIVER_VENV}'
unset UV_CACHE_DIR
"
    DRIVER_LAUNCH="'${CONTAINER_DRIVER_VENV}/bin/python' examples/run_grpo.py"
    ;;
  false)
    export UV_CACHE_DIR_OVERRIDE="${SHARED_UV_CACHE}"
    DRIVER_PYTHONPATH="${REPO}"
    DRIVER_ENV_COMMANDS="
export NEMO_RL_VENV_DIR='${CACHE_ROOT}/venvs'
export UV_CACHE_DIR='${SHARED_UV_CACHE}'
export UV_PROJECT_ENVIRONMENT='${CACHE_ROOT}/driver-venv'
${UV_PYTHON_COMMAND}
"
    DRIVER_LAUNCH="uv run --frozen examples/run_grpo.py"
    ;;
  *)
    echo "USE_CONTAINER_VENVS must be true or false" >&2
    exit 2
    ;;
esac

export COMMAND="
set -euo pipefail
cd '${REPO}'
export PYTHONPATH='${DRIVER_PYTHONPATH}'
export HF_HOME='${HF_HOME}'
export NRL_FORCE_REBUILD_VENVS='${NRL_FORCE_REBUILD_VENVS}'
${DRIVER_ENV_COMMANDS}
export UV_LOCK_TIMEOUT=7200
export NVTE_CUDA_ARCHS='${NVTE_CUDA_ARCHS_OVERRIDE:-100}'
export TORCH_CUDA_ARCH_LIST='${TORCH_CUDA_ARCH_LIST_OVERRIDE:-10.0}'
export NRL_REFIT_NUM_STREAMS='${NRL_REFIT_NUM_STREAMS}'
export NRL_MXFP8_BATCHED_SHUFFLE='${NRL_MXFP8_BATCHED_SHUFFLE}'
export NRL_REFIT_CACHED_LOADERS='${NRL_REFIT_CACHED_LOADERS}'
if [[ -s '${WANDB_KEY_FILE}' ]]; then
  export WANDB_API_KEY=\$(cat '${WANDB_KEY_FILE}')
fi
${DRIVER_LAUNCH} \
  --config '${CONFIG}' \
  cluster.num_nodes='${TOTAL_NODES}' \
  cluster.gpus_per_node='${GPUS_PER_NODE}' \
  cluster.segment_size='${SEGMENT_SIZE}' \
  policy.generation.colocated.enabled=false \
  policy.generation.colocated.resources.num_nodes='${GEN_NODES}' \
  policy.generation.colocated.resources.gpus_per_node='${GPUS_PER_NODE}' \
  policy.generation.refit_transport='${REFIT_TRANSPORT}' \
  policy.megatron_cfg.tensor_model_parallel_size=1 \
  policy.megatron_cfg.pipeline_model_parallel_size=1 \
  policy.megatron_cfg.expert_model_parallel_size=16 \
  policy.megatron_cfg.expert_tensor_parallel_size=1 \
  policy.generation.vllm_cfg.tensor_parallel_size=1 \
  ++policy.generation.vllm_cfg.pipeline_parallel_size=1 \
  ++policy.generation.vllm_cfg.expert_parallel_size=1 \
  policy.train_global_batch_size=2048 \
  loss_fn.force_on_policy_ratio=true \
  grpo.max_num_steps='${MAX_STEPS}' \
  grpo.val_at_start=false \
  ++grpo.val_at_end=false \
  grpo.val_period=1000000 \
  checkpointing.enabled=false \
  logger.log_dir='${EXPERIMENT_ROOT}/logs' \
  logger.wandb_enabled=true \
  logger.wandb.project='${WANDB_PROJECT}' \
  logger.wandb.name='${RUN_NAME}' \
  ${PRECISION_OVERRIDES}
"

exec bash "${RAY_SUB_PATH}"
