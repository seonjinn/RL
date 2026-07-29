#!/usr/bin/env bash

set -euo pipefail

: "${ARM:?ARM is required: baseline or optimized}"
: "${MODE:?MODE is required: bf16, blockwise-fp8, or mxfp8-probe}"
: "${REPO:?REPO is required}"
: "${CONTAINER:?CONTAINER is required}"
: "${TOTAL_NODES:?TOTAL_NODES is required}"
: "${GPUS_PER_NODE:?GPUS_PER_NODE is required}"
: "${GEN_NODES:?GEN_NODES is required}"
: "${SEGMENT_SIZE:?SEGMENT_SIZE is required}"
: "${MAX_STEPS:?MAX_STEPS is required}"
: "${RUN_NAME:?RUN_NAME is required}"
: "${EXPERIMENT_ROOT:?EXPERIMENT_ROOT is required}"

CONFIG_BF16=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml
CONFIG_FP8=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml

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
    PRECISION_OVERRIDES="
      policy.megatron_cfg.fp8_cfg.enabled=false
      policy.megatron_cfg.fp8_cfg.fp8_param=false
      ++policy.generation.vllm_cfg.precision=bfloat16
      ++policy.generation.vllm_cfg.is_mx=false
    "
    ;;
  blockwise-fp8)
    CONFIG=${CONFIG_FP8}
    PRECISION_OVERRIDES="
      policy.megatron_cfg.fp8_cfg.enabled=true
      policy.megatron_cfg.fp8_cfg.fp8=e4m3
      policy.megatron_cfg.fp8_cfg.fp8_recipe=blockwise
      policy.megatron_cfg.fp8_cfg.fp8_param=true
      policy.generation.vllm_cfg.precision=fp8
      policy.generation.vllm_cfg.is_mx=false
    "
    ;;
  mxfp8-probe)
    CONFIG=${CONFIG_FP8}
    PRECISION_OVERRIDES="
      policy.megatron_cfg.fp8_cfg.enabled=true
      policy.megatron_cfg.fp8_cfg.fp8=e4m3
      policy.megatron_cfg.fp8_cfg.fp8_recipe=blockwise
      policy.megatron_cfg.fp8_cfg.fp8_param=true
      policy.generation.vllm_cfg.precision=fp8
      policy.generation.vllm_cfg.is_mx=true
    "
    ;;
  *)
    echo "MODE must be bf16, blockwise-fp8, or mxfp8-probe" >&2
    exit 2
    ;;
esac

test -d "${REPO}/.git"
test -f "${REPO}/${CONFIG}"
test -f "${CONTAINER}"
mkdir -p "${EXPERIMENT_ROOT}" "${EXPERIMENT_ROOT}/logs"

WORK_ROOT=${WORK_ROOT:-/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna}
HF_HOME=${HF_HOME:-${WORK_ROOT}/.cache/huggingface}
CACHE_ROOT=${CACHE_ROOT:-${WORK_ROOT}/mopd_nano_fast/.cache/nccl-reshard-pr3294/${ARM}-${MODE}}
WANDB_PROJECT=${WANDB_PROJECT:-sna-pr3294-nccl-reshard}
WANDB_KEY_FILE=${WANDB_KEY_FILE:-${CACHE_ROOT}/.wandb_key}
PYTHON_BIN=${PYTHON_BIN:-}

mkdir -p "${HF_HOME}" "${CACHE_ROOT}/uv" "${CACHE_ROOT}/venvs"
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
NRL_MXFP8_BATCHED_SHUFFLE=${NRL_MXFP8_BATCHED_SHUFFLE}
NRL_REFIT_CACHED_LOADERS=${NRL_REFIT_CACHED_LOADERS}
EOF

export NRL_REFIT_NUM_STREAMS=${NRL_REFIT_NUM_STREAMS:-2}
export CONTAINER
export MOUNTS=/lustre:/lustre
export BASE_LOG_DIR="${EXPERIMENT_ROOT}"
export UV_CACHE_DIR_OVERRIDE="${CACHE_ROOT}/uv"

export COMMAND="
set -euo pipefail
cd '${REPO}'
export PYTHONPATH='${REPO}'
export HF_HOME='${HF_HOME}'
export NRL_FORCE_REBUILD_VENVS=false
export NEMO_RL_VENV_DIR='${CACHE_ROOT}/venvs'
export UV_CACHE_DIR='${CACHE_ROOT}/uv'
export UV_LOCK_TIMEOUT=7200
export NVTE_CUDA_ARCHS=100
export TORCH_CUDA_ARCH_LIST=10.0
export NRL_REFIT_NUM_STREAMS='${NRL_REFIT_NUM_STREAMS}'
export NRL_MXFP8_BATCHED_SHUFFLE='${NRL_MXFP8_BATCHED_SHUFFLE}'
export NRL_REFIT_CACHED_LOADERS='${NRL_REFIT_CACHED_LOADERS}'
if [[ -s '${WANDB_KEY_FILE}' ]]; then
  export WANDB_API_KEY=\$(cat '${WANDB_KEY_FILE}')
fi
PYTHON_BIN='${PYTHON_BIN}'
if [[ -z \"\${PYTHON_BIN}\" ]]; then
  PYTHON_BIN=/opt/nemo_rl_venv/bin/python
fi
if [[ ! -x \"\${PYTHON_BIN}\" ]]; then
  PYTHON_BIN=\$(find /root/.local/share/uv/python \
    -type f -path '*/bin/python3.13' -perm -u+x -print -quit)
fi
test -n \"\${PYTHON_BIN}\" && test -x \"\${PYTHON_BIN}\"
\"\${PYTHON_BIN}\" examples/run_grpo.py \
  --config '${CONFIG}' \
  cluster.num_nodes='${TOTAL_NODES}' \
  cluster.gpus_per_node='${GPUS_PER_NODE}' \
  cluster.segment_size='${SEGMENT_SIZE}' \
  policy.generation.colocated.enabled=false \
  policy.generation.colocated.resources.num_nodes='${GEN_NODES}' \
  policy.generation.colocated.resources.gpus_per_node='${GPUS_PER_NODE}' \
  policy.generation.refit_transport=nccl_reshard \
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

exec bash "${REPO}/ray.sub"
