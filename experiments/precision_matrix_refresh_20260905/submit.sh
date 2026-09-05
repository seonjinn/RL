#!/usr/bin/env bash

set -euo pipefail

ACTION=${ACTION:-render}
CLUSTER=${CLUSTER:-oci}
MODEL=${MODEL:-qwen30}
MODE=${MODE:-async}
ARM=${ARM:-bf16-bf16}
MAX_STEPS=${MAX_STEPS:-20}
RUN_GROUP=${RUN_GROUP:-$(date +%Y%m%d-%H%M%S)}
WALLTIME=${WALLTIME:-04:00:00}
PARTITION=${PARTITION:-batch}
EXPERIMENT=experiments/precision_matrix_refresh_20260905

case "${ACTION}" in
  render|test-only|submit) ;;
  *) echo "ACTION must be render, test-only, or submit" >&2; exit 2 ;;
esac
case "${MODEL}" in
  qwen30|qwen235|lightning|qwen35) ;;
  *) echo "MODEL must be qwen30, qwen235, lightning, or qwen35" >&2; exit 2 ;;
esac
case "${MODE}" in
  sync|async) ;;
  *) echo "MODE must be sync or async" >&2; exit 2 ;;
esac
case "${ARM}" in
  bf16-bf16|bf16-mxfp8|mxfp8-mxfp8) ;;
  *) echo "ARM must be bf16-bf16, bf16-mxfp8, or mxfp8-mxfp8" >&2; exit 2 ;;
esac

case "${CLUSTER}" in
  oci)
    REPO=${REPO:-/home/${USER}/RL-precision-matrix-refresh-20260905}
    CONTAINER=${CONTAINER:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/${USER}/containers/nemo_rl_nightly.sqsh}
    HF_HOME_SOURCE=${HF_HOME_SOURCE:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/${USER}/hf_home}
    RESULT_ROOT=${RESULT_ROOT:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/${USER}/precision-matrix-refresh-20260905}
    LOCAL_ROOT=${LOCAL_ROOT:-/raid/scratch/${USER}/precision-matrix-refresh-20260905}
    GPU_REQUEST=(--gres=gpu:4)
    ;;
  ptyche)
    REPO=${REPO:-/home/${USER}/RL-precision-matrix-refresh-20260905}
    CONTAINER=${CONTAINER:-/lustre/fsw/coreai_dlalgo_llm/users/${USER}/containers/nemo_rl_nightly.sqsh}
    HF_HOME_SOURCE=${HF_HOME_SOURCE:-/lustre/fsw/coreai_dlalgo_llm/users/${USER}/hf_home}
    RESULT_ROOT=${RESULT_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/${USER}/precision-matrix-refresh-20260905}
    LOCAL_ROOT=${LOCAL_ROOT:-/tmp/${USER}/precision-matrix-refresh-20260905}
    GPU_REQUEST=()
    ;;
  lyris)
    REPO=${REPO:-/home/${USER}/RL-precision-matrix-refresh-20260905}
    CONTAINER=${CONTAINER:-/lustre/fsw/coreai_dlalgo_llm/users/${USER}/containers/nemo_rl_nightly.sqsh}
    HF_HOME_SOURCE=${HF_HOME_SOURCE:-/lustre/fsw/coreai_dlalgo_llm/users/${USER}/hf_home}
    RESULT_ROOT=${RESULT_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/${USER}/precision-matrix-refresh-20260905}
    LOCAL_ROOT=${LOCAL_ROOT:-/raid/scratch/${USER}/precision-matrix-refresh-20260905}
    GPU_REQUEST=()
    ;;
  *) echo "CLUSTER must be oci, ptyche, or lyris" >&2; exit 2 ;;
esac

: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT after checking FairShare}"
: "${WANDB_HOME:=/home/${USER}}"

case "${MODEL}:${MODE}" in
  qwen30:sync)
    CONFIG=${EXPERIMENT}/qwen30-sync.yaml
    NUM_NODES=4
    SEGMENT_SIZE=4
    MODEL_CACHE=models--Qwen--Qwen3-30B-A3B
    FIRST_BF16=0
    LAST_BF16=0
    ;;
  qwen30:async)
    CONFIG=${EXPERIMENT}/qwen30-async.yaml
    NUM_NODES=4
    SEGMENT_SIZE=2
    MODEL_CACHE=models--Qwen--Qwen3-30B-A3B
    FIRST_BF16=0
    LAST_BF16=0
    ;;
  qwen235:sync)
    CONFIG=${EXPERIMENT}/qwen235-sync.yaml
    NUM_NODES=16
    SEGMENT_SIZE=16
    MODEL_CACHE=models--Qwen--Qwen3-235B-A22B
    FIRST_BF16=0
    LAST_BF16=0
    ;;
  qwen235:async)
    CONFIG=${EXPERIMENT}/qwen235-async.yaml
    NUM_NODES=32
    SEGMENT_SIZE=16
    MODEL_CACHE=models--Qwen--Qwen3-235B-A22B
    FIRST_BF16=0
    LAST_BF16=0
    ;;
  lightning:sync)
    CONFIG=${EXPERIMENT}/lightning-sync.yaml
    NUM_NODES=4
    SEGMENT_SIZE=4
    MODEL_CACHE=models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
    FIRST_BF16=2
    LAST_BF16=6
    ;;
  lightning:async)
    CONFIG=${EXPERIMENT}/lightning-async.yaml
    NUM_NODES=8
    SEGMENT_SIZE=4
    MODEL_CACHE=models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
    FIRST_BF16=2
    LAST_BF16=6
    ;;
  qwen35:sync)
    CONFIG=${EXPERIMENT}/qwen35-sync.yaml
    NUM_NODES=4
    SEGMENT_SIZE=4
    MODEL_CACHE=models--Qwen--Qwen3.5-35B-A3B-Base
    FIRST_BF16=2
    LAST_BF16=6
    ;;
  qwen35:async)
    CONFIG=${EXPERIMENT}/qwen35-async.yaml
    NUM_NODES=6
    SEGMENT_SIZE=2
    MODEL_CACHE=models--Qwen--Qwen3.5-35B-A3B-Base
    FIRST_BF16=2
    LAST_BF16=6
    ;;
esac

SOURCE_SHA=$(git -C "${REPO}" rev-parse HEAD 2>/dev/null || printf unknown)
RUN_NAME="pmx-${CLUSTER}-${MODEL}-${MODE}-${ARM}-${RUN_GROUP}"
RUN_ROOT="${RESULT_ROOT}/${RUN_NAME}"
LOCAL_JOB_ROOT="${LOCAL_ROOT}/${RUN_NAME}"

COMMON_OVERRIDES=(
  "grpo.max_num_steps=${MAX_STEPS}"
  "grpo.val_at_start=false"
  "++grpo.val_at_end=false"
  "++policy.generation.refit_timeout_s=300.0"
  "checkpointing.enabled=false"
  "policy.generation.vllm_cfg.use_tqdm=false"
  "policy.generation.vllm_cfg.refit_cache_loader_routes=true"
  "policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm"
  "policy.generation.vllm_kwargs.expert_placement_strategy=linear"
  "logger.log_dir=${RUN_ROOT}/logs"
  "logger.wandb_enabled=true"
  "logger.wandb.project=nemo-rl-mxfp8-training"
  "logger.wandb.name=${RUN_NAME}"
  "logger.tensorboard_enabled=true"
  "logger.monitor_gpus=true"
)

case "${ARM}" in
  bf16-bf16)
    PRECISION_OVERRIDES=(
      "policy.megatron_cfg.fp8_cfg.enabled=false"
      "policy.megatron_cfg.fp8_cfg.fp8_param=false"
      "policy.generation.vllm_cfg.precision=bfloat16"
      "++policy.generation.vllm_cfg.is_mx=false"
      "policy.generation.vllm_cfg.refit_prequantize=false"
      "policy.generation.vllm_cfg.num_first_layers_in_bf16=0"
      "policy.generation.vllm_cfg.num_last_layers_in_bf16=0"
    )
    ;;
  bf16-mxfp8)
    PRECISION_OVERRIDES=(
      "policy.megatron_cfg.fp8_cfg.enabled=false"
      "policy.megatron_cfg.fp8_cfg.fp8_param=false"
      "policy.generation.vllm_cfg.precision=fp8"
      "++policy.generation.vllm_cfg.is_mx=true"
      "policy.generation.vllm_cfg.refit_prequantize=$([[ ${MODE} == sync ]] && printf true || printf false)"
      "policy.generation.vllm_cfg.num_first_layers_in_bf16=${FIRST_BF16}"
      "policy.generation.vllm_cfg.num_last_layers_in_bf16=${LAST_BF16}"
    )
    ;;
  mxfp8-mxfp8)
    PRECISION_OVERRIDES=(
      "policy.megatron_cfg.fp8_cfg.enabled=true"
      "policy.megatron_cfg.fp8_cfg.fp8=e4m3"
      "policy.megatron_cfg.fp8_cfg.fp8_recipe=mxfp8"
      "policy.megatron_cfg.fp8_cfg.fp8_param=true"
      "++policy.megatron_cfg.moe_router_dtype=fp32"
      "++policy.megatron_cfg.te_precision_config_file=${EXPERIMENT}/te_routed_fp8param.yaml"
      "++policy.megatron_cfg.first_last_layers_bf16=true"
      "++policy.megatron_cfg.num_layers_at_start_in_bf16=${FIRST_BF16}"
      "++policy.megatron_cfg.num_layers_at_end_in_bf16=${LAST_BF16}"
      "policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=true"
      "policy.megatron_cfg.distributed_data_parallel_config.overlap_grad_reduce=true"
      "policy.generation.vllm_cfg.precision=fp8"
      "++policy.generation.vllm_cfg.is_mx=true"
      "policy.generation.vllm_cfg.refit_prequantize=false"
      "policy.generation.vllm_cfg.num_first_layers_in_bf16=${FIRST_BF16}"
      "policy.generation.vllm_cfg.num_last_layers_in_bf16=${LAST_BF16}"
    )
    ;;
esac

printf 'cluster=%s\nmodel=%s\nmode=%s\narm=%s\nconfig=%s\nnodes=%s\nsegment=%s\nsteps=%s\nsha=%s\nrun=%s\n' \
  "${CLUSTER}" "${MODEL}" "${MODE}" "${ARM}" "${CONFIG}" "${NUM_NODES}" \
  "${SEGMENT_SIZE}" "${MAX_STEPS}" "${SOURCE_SHA}" "${RUN_NAME}"
printf 'overrides:'
printf ' %q' "${COMMON_OVERRIDES[@]}" "${PRECISION_OVERRIDES[@]}"
printf '\n'

if [[ "${ACTION}" == render ]]; then
  exit 0
fi

for path in "${REPO}/${CONFIG}" "${REPO}/ray.sub" "${CONTAINER}" \
  "${HF_HOME_SOURCE}/hub/${MODEL_CACHE}" "${WANDB_HOME}/.netrc"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 2
  fi
done

if [[ "${ACTION}" == submit ]]; then
  git -C "${REPO}" pull --ff-only
  git -C "${REPO}" submodule update --init --recursive --checkout
  if [[ -n "$(git -C "${REPO}" status --porcelain --untracked-files=no --ignore-submodules=none)" ]]; then
    echo "Repository and pinned submodules must be clean before submission" >&2
    exit 2
  fi
fi

SOURCE_SHA=$(git -C "${REPO}" rev-parse HEAD)
mkdir -p "${RUN_ROOT}/logs"

COMMAND=$(printf '%q ' /opt/nemo_rl_venv/bin/python examples/run_grpo.py \
  --config "${CONFIG}" "${COMMON_OVERRIDES[@]}" "${PRECISION_OVERRIDES[@]}")
COMMAND="set -euo pipefail; cd ${REPO}; \
export HOME=/root; \
export HF_HOME=${LOCAL_JOB_ROOT}/hf; \
export HF_DATASETS_CACHE=${LOCAL_JOB_ROOT}/hf/datasets; \
export HUGGINGFACE_HUB_CACHE=${LOCAL_JOB_ROOT}/hf/hub; \
export NEMO_RL_VENV_DIR=${LOCAL_JOB_ROOT}/venv; \
export VLLM_CACHE_ROOT=${LOCAL_JOB_ROOT}/vllm; \
export TORCHINDUCTOR_CACHE_DIR=${LOCAL_JOB_ROOT}/inductor; \
export TRITON_CACHE_DIR=${LOCAL_JOB_ROOT}/triton; \
export UV_CACHE_DIR=${LOCAL_JOB_ROOT}/uv; \
export RAY_TMPDIR=${LOCAL_JOB_ROOT}/ray; \
export PYTHONPATH=${REPO}:${REPO}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${REPO}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM; \
export FLA_TILELANG=0; \
${COMMAND}"

SETUP_COMMAND="set -euo pipefail; \
mkdir -p ${LOCAL_JOB_ROOT}/hf/hub ${LOCAL_JOB_ROOT}/hf/datasets ${LOCAL_JOB_ROOT}/vllm ${LOCAL_JOB_ROOT}/inductor ${LOCAL_JOB_ROOT}/triton ${LOCAL_JOB_ROOT}/uv ${LOCAL_JOB_ROOT}/ray; \
rsync -a --ignore-existing ${HF_HOME_SOURCE}/hub/${MODEL_CACHE}/ ${LOCAL_JOB_ROOT}/hf/hub/${MODEL_CACHE}/; \
if [ -d ${HF_HOME_SOURCE}/datasets ]; then rsync -a --ignore-existing ${HF_HOME_SOURCE}/datasets/ ${LOCAL_JOB_ROOT}/hf/datasets/; fi"

export CONTAINER
export MOUNTS="/lustre:/lustre,/home:/home,${WANDB_HOME}/.netrc:/root/.netrc"
if [[ "${CLUSTER}" == oci ]]; then
  MOUNTS="${MOUNTS},/raid/scratch:/raid/scratch"
fi
export CONTAINER_REMAP_ROOT=1
export COMMAND
export SETUP_COMMAND
export GPUS_PER_NODE=4
export CPUS_PER_WORKER=${CPUS_PER_WORKER:-144}
export BASE_LOG_DIR="${RUN_ROOT}"
export RAY_TMPDIR_ROOT="${LOCAL_JOB_ROOT}/ray"

SBATCH_MODE=()
if [[ "${ACTION}" == test-only ]]; then
  SBATCH_MODE=(--test-only)
fi

exec sbatch "${SBATCH_MODE[@]}" \
  --nodes="${NUM_NODES}" \
  "${GPU_REQUEST[@]}" \
  --exclusive \
  --account="${SLURM_ACCOUNT}" \
  --partition="${PARTITION}" \
  --time="${WALLTIME}" \
  --segment="${SEGMENT_SIZE}" \
  --job-name="${SLURM_ACCOUNT}.${RUN_NAME}" \
  --output="${RUN_ROOT}/slurm-%j.out" \
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"model_loading","description":"precision matrix startup"}}' \
  "${REPO}/ray.sub"
