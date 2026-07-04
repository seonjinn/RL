#!/bin/bash

set -euo pipefail

# 64-node Nemotron 3 Super / Super V3 NeMo-RL run on latest-main-based code.
# This uses main's native pretrained_checkpoint.format=megatron_lm path instead
# of the older NRL_MLM_CHECKPOINT_DIR hack from ashors/debug/load-mlm-checkpoint.

NUM_ACTOR_NODES=${NUM_ACTOR_NODES:-64}
STEPS=${STEPS:-20}
TRAIN_GLOBAL_BATCH_SIZE=${TRAIN_GLOBAL_BATCH_SIZE:-992}
RUN_KIND=${RUN_KIND:-superv3_nemorl_main}
RUN_ID=${RUN_ID:-$(date +%H%M%S)}

WORKDIR=${WORKDIR:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/RL_main_prepacked_latest_48b2cd2}
WORKDIR=$(realpath "${WORKDIR}")
DATA_DIR=${DATA_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/pthombre/LongSFTRun}
ORIG_MLM_CKPT=${ORIG_MLM_CKPT:-${DATA_DIR}/reinit_output/checkpoints/pthombre-reinit-embeddings}
PRETRAINED_DIR=${PRETRAINED_DIR:-${DATA_DIR}/pretrainedSuperModelHF}
TOKENIZER_DIR=${TOKENIZER_DIR:-/lustre/fsw/portfolios/llmservice/users/jjennings/workspace/megatron-sft-right-truncation/tokenizers}

SHARED_CACHE_ROOT=${SHARED_CACHE_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/.cache/nemo_rl_superv3_main}
HF_HOME=${SUPER_HF_HOME:-${SHARED_CACHE_ROOT}/hf_home}

TRAIN_FILE=${TRAIN_FILE:-final_shuffled_25pct_materialized_filtered.jsonl}
VAL_FILE=${VAL_FILE:-${TRAIN_FILE}}
CONFIG_PATH=${CONFIG_PATH:-examples/configs/sft_superv3.yaml}

CONTAINER=${CONTAINER:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo-rl-nightly-20260522/nemo_rl_nightly.sqsh}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-coreai_dlalgo_nemorl}
SLURM_PARTITION=${SLURM_PARTITION:-batch}
SLURM_TIME=${SLURM_TIME:-4:00:0}

WANDB_PROJECT=${WANDB_PROJECT:-pthombre-nemotron-sft}
WANDB_NAME=${WANDB_NAME:-superv3_nemorl_main_${RUN_KIND}_${STEPS}step_${RUN_ID}}
WANDB_ENABLED=${WANDB_ENABLED:-false}
CHECKPOINTING_ENABLED=${CHECKPOINTING_ENABLED:-false}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-results/${WANDB_NAME}}

LOG_DIR=${LOG_DIR:-${WORKDIR}/logs/$(date +%Y%m%d)}
mkdir -p "${LOG_DIR}" "${HF_HOME}" "${SHARED_CACHE_ROOT}"

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

require_dir "${WORKDIR}" WORKDIR
require_file "${WORKDIR}/ray.sub" ray.sub
require_file "${WORKDIR}/${CONFIG_PATH}" "${CONFIG_PATH}"
require_dir "${DATA_DIR}" DATA_DIR
require_file "${DATA_DIR}/${TRAIN_FILE}" train_data
require_file "${DATA_DIR}/${VAL_FILE}" val_data
require_dir "${ORIG_MLM_CKPT}" ORIG_MLM_CKPT
require_dir "${PRETRAINED_DIR}" PRETRAINED_DIR
require_dir "${TOKENIZER_DIR}" TOKENIZER_DIR
require_file "${CONTAINER}" CONTAINER

cd "${WORKDIR}"

SNAPSHOT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo main)
EXP_SUFFIX="${SNAPSHOT_COMMIT}-$(date +%m%d)-${RUN_ID}"
NAME="sft_superv3_${RUN_KIND}_gbs${TRAIN_GLOBAL_BATCH_SIZE}_${NUM_ACTOR_NODES}n_${STEPS}step@${EXP_SUFFIX}"

rm -f "${LOG_DIR}/${NAME}.log"

export CONTAINER
export BASE_LOG_DIR="${LOG_DIR}"
export MOUNTS="${WORKDIR}:${WORKDIR},${WORKDIR}:/opt/nemo-rl,${SHARED_CACHE_ROOT}:/mnt/nemo_cache,${PRETRAINED_DIR}:/mnt/superv3_pretrained:ro,${ORIG_MLM_CKPT}:/mnt/superv3_pretrained_mlm:ro,${DATA_DIR}:/mnt/data:ro,${TOKENIZER_DIR}:/mnt/tokenizer:ro,/lustre:/lustre"

PYTHONPATH_PREFIX="${WORKDIR}:${WORKDIR}/examples:${WORKDIR}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${WORKDIR}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:${WORKDIR}/3rdparty/Automodel-workspace/Automodel:\${PYTHONPATH:-}"
EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-}

export COMMAND="set -euo pipefail; \
export NUM_ACTOR_NODES=${NUM_ACTOR_NODES}; \
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-/opt/nemo_rl_venv}; \
UV_HTTP_TIMEOUT=600 \
HF_HOME=/mnt/nemo_cache/hf_home \
HF_MODULES_CACHE=/mnt/nemo_cache/hf_modules \
NEMO_RL_VENV_DIR=/mnt/nemo_cache/venvs/${NAME} \
NRL_MEGATRON_CHECKPOINT_DIR=/mnt/nemo_cache/nemo_rl \
PYTHONPATH=${PYTHONPATH_PREFIX} \
VLLM_PRECOMPILED_WHEEL_LOCATION=https://github.com/vllm-project/vllm/releases/download/v0.11.2/vllm-0.11.2+cu129-cp38-abi3-manylinux1_x86_64.whl \
NRL_WG_USE_RAY_REF=1 \
NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS:-true} \
NRL_IGNORE_VERSION_MISMATCH=${NRL_IGNORE_VERSION_MISMATCH:-1} \
NRL_RAY_DISABLE_TASK_EVENTS_FOR_TRAIN=${NRL_RAY_DISABLE_TASK_EVENTS_FOR_TRAIN:-0} \
NRL_RAY_DISCARD_NON_RETURN_TRAIN_RESULTS=${NRL_RAY_DISCARD_NON_RETURN_TRAIN_RESULTS:-0} \
NRL_RAY_ASYNC_DRAIN_NON_RETURN_TRAIN_RESULTS=${NRL_RAY_ASYNC_DRAIN_NON_RETURN_TRAIN_RESULTS:-0} \
NRL_SKIP_MOE_METRICS=${NRL_SKIP_MOE_METRICS:-0} \
NRL_CHECK_FOR_NAN_IN_GRAD=${NRL_CHECK_FOR_NAN_IN_GRAD:-1} \
NRL_LIGHTWEIGHT_STEP_BREAKDOWN=${NRL_LIGHTWEIGHT_STEP_BREAKDOWN:-0} \
WANDB_MODE=${WANDB_MODE:-offline} \
uv run --locked --extra mcore examples/run_sft.py \
  --config ${CONFIG_PATH} \
  logger.wandb.name=${WANDB_NAME} \
  logger.wandb.project=${WANDB_PROJECT} \
  logger.wandb_enabled=${WANDB_ENABLED} \
  policy.model_name=/mnt/superv3_pretrained \
  policy.tokenizer.name=/mnt/tokenizer \
  ++policy.pretrained_checkpoint.format=megatron_lm \
  ++policy.pretrained_checkpoint.path=/mnt/superv3_pretrained_mlm \
  policy.train_global_batch_size=${TRAIN_GLOBAL_BATCH_SIZE} \
  policy.sequence_packing.enabled=True \
  data.train.data_path=/mnt/data/${TRAIN_FILE} \
  data.validation.data_path=/mnt/data/${VAL_FILE} \
  checkpointing.enabled=${CHECKPOINTING_ENABLED} \
  checkpointing.save_period=100000 \
  checkpointing.checkpoint_dir=${CHECKPOINT_DIR} \
  cluster.num_nodes=${NUM_ACTOR_NODES} \
  sft.max_num_steps=${STEPS} \
  sft.val_period=100000 \
  sft.val_at_start=false \
  policy.megatron_cfg.scheduler.lr_warmup_iters=469 \
  policy.megatron_cfg.sequence_parallel=True \
  policy.megatron_cfg.bias_activation_fusion=False \
  policy.megatron_cfg.moe_permute_fusion=True \
  policy.megatron_cfg.mtp_loss_scaling_factor=0.3 \
  policy.megatron_cfg.mtp_num_layers=2 \
  policy.megatron_cfg.mtp_use_repeated_layer=True \
  policy.megatron_cfg.mtp_detach_heads=False \
  policy.megatron_cfg.distributed_data_parallel_config.overlap_grad_reduce=True \
  policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=True \
  policy.megatron_cfg.distributed_data_parallel_config.data_parallel_sharding_strategy=no_shard \
  ${EXTRA_OVERRIDES} \
  2>&1 | tee ${LOG_DIR}/${NAME}.log"

echo "WORKDIR=${WORKDIR}"
echo "DATA_DIR=${DATA_DIR}"
echo "PRETRAINED_DIR=${PRETRAINED_DIR}"
echo "ORIG_MLM_CKPT=${ORIG_MLM_CKPT}"
echo "TOKENIZER_DIR=${TOKENIZER_DIR}"
echo "CONTAINER=${CONTAINER}"
echo "SHARED_CACHE_ROOT=${SHARED_CACHE_ROOT}"
echo "NUM_ACTOR_NODES=${NUM_ACTOR_NODES}"
echo "NAME=${NAME}"

SBATCH_ARGS=(
  --nodes="${NUM_ACTOR_NODES}"
  --account="${SLURM_ACCOUNT}"
  --job-name="${NAME}"
  --partition="${SLURM_PARTITION}"
  --time="${SLURM_TIME}"
  --gres=gpu:8
)

if [[ -n "${SBATCH_COMMENT:-}" ]]; then
  SBATCH_ARGS+=(--comment="${SBATCH_COMMENT}")
fi

if [[ -n "${DEPENDENCY:-}" ]]; then
  SBATCH_ARGS+=(--dependency="${DEPENDENCY}")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1"
  echo "COMMAND=${COMMAND}"
  echo "MOUNTS=${MOUNTS}"
  printf 'SBATCH_ARGS=%q ' "${SBATCH_ARGS[@]}"
  echo "ray.sub"
  exit 0
fi

sbatch "${SBATCH_ARGS[@]}" ray.sub
