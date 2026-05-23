#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_NEMORL="${NEMORL:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SOURCE_NEMORL="$(cd "${SOURCE_NEMORL}" && pwd)"
NEMORL="${SOURCE_NEMORL}"

if [[ -f "${SOURCE_NEMORL}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${SOURCE_NEMORL}/.env"
  set +a
fi

CONFIG_PATH="${CONFIG_PATH:-examples/configs/vlmConv3d_grpo_mix_omnirlSDG-videorlSDG-videor1Comm-2minVidFilter-imageCommRB5-aud_nomni_32f_dedup_draco_super.yaml}"
EXP_NAME="${EXP_NAME:-nomni_comboGRPO_v3_GA-SFT-MPO-TRL1-visG125-ckpt_GA-IRL-Filtered-AllImg3_nrt_fix0417}"
RUN_ID="${RUN_ID:-20260418}"
NUM_NODES="${NUM_NODES:-32}"
SNAPSHOT_CODE="${SNAPSHOT_CODE:-1}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
MICRO_BS="${MICRO_BS:-1}"
LOGPROB_BS="${LOGPROB_BS:-1}"
JOB_CYCLES="${JOB_CYCLES:-8}"
NUM_FRAMES="${NUM_FRAMES:-32}"
export NRL_VIDEO_SFT_MIN_FRAMES="${NUM_FRAMES}"
export NRL_VIDEO_SFT_MAX_FRAMES="${NUM_FRAMES}"
GLOBAL_TRAIN_BATCH_SIZE="${GLOBAL_TRAIN_BATCH_SIZE:-$((NUM_NODES * GRADIENT_ACCUMULATION_STEPS * MICRO_BS * GPUS_PER_NODE))}"
JOB_NAME="${JOB_NAME:-${EXP_NAME}_n${NUM_NODES}_bs${GLOBAL_TRAIN_BATCH_SIZE}_ga${GRADIENT_ACCUMULATION_STEPS}_f${NUM_FRAMES}_j${RUN_ID}}"
JOB_HASH="${JOB_HASH:-$(printf '%s' "${JOB_NAME}" | openssl dgst -sha1 -binary | od -An -tx1 | tr -d ' \n' | cut -c1-12)}"

MODEL_NAME="/lustre/fsw/portfolios/llmservice/users/smohsenitahe/checkpoint/grpo_vision_mpo_sft_iter_2200_mpo_200_rl_50_20260407_blend_v6/iter_125"
TRAIN_DATA_PATH="/lustre/fsw/portfolios/llmservice/users/hanrongy/dataset/nemotron_omni_data/rl/comboGRPO_v3_gaIRL-Filtered/comboV3_gaFiltered-allImgV8_wOCR_wUnans_NRT_20260419_084620.jsonl"

RESULTS_ROOT="${RESULTS_ROOT:-${SOURCE_NEMORL}/../jobs}"
RESULTS_DIR="${RESULTS_DIR:-${RESULTS_ROOT}/${JOB_NAME}}"
LOGS_DIR="${LOGS_DIR:-${RESULTS_DIR}/logs}"
mkdir -p "${LOGS_DIR}" "${RESULTS_DIR}"
export BASE_LOG_DIR="${BASE_LOG_DIR:-${LOGS_DIR}}"
export OBJECT_STORE_MEMORY="${OBJECT_STORE_MEMORY:-300000000000}"

if [[ ! -f "${SOURCE_NEMORL}/${CONFIG_PATH}" ]]; then
  echo "Config not found: ${SOURCE_NEMORL}/${CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -f "${SOURCE_NEMORL}/ray.sub" ]]; then
  echo "ray.sub not found under NEMORL=${SOURCE_NEMORL}" >&2
  exit 1
fi

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-nemotron_omni_vision}"
SBATCH_TIME="${SBATCH_TIME:-04:00:00}"
if [[ -z "${SBATCH_PARTITION:-}" ]]; then
  if [[ "$(hostname)" == *"draco-oci"* ]]; then
    SBATCH_PARTITION="batch_block1,batch_block3,batch_block4,backfill_block1,backfill_block2,backfill_block3,backfill_block4"
  elif [[ "$(hostname)" == *"cw-dfw"* ]]; then
    SBATCH_PARTITION="batch,backfill,batch_short"
  elif [[ "$(hostname)" == *"cs-oci-ord"* ]]; then
    SBATCH_PARTITION="backfill_block1,grizzly,polar,polar3,polar4"
  elif [[ "$(hostname)" == *"oci-nrt"* ]]; then
    SBATCH_PARTITION="batch_block1"
  else
    SBATCH_PARTITION="batch,batch_large,batch_large_long,batch_long"
  fi
fi

CONTAINER_ROOT="${CONTAINER_ROOT:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/hanrongy/project/nemotron_omni/rl/images}"
export CONTAINER="${CONTAINER:-${CONTAINER_ROOT}/super-omni-vllm20-super-vlm2-20260507-0905b74.sqsh}"
export MOUNTS="${MOUNTS:-/lustre:/lustre,/home}"
export NUM_NODES

export CACHE_ROOT="${CACHE_ROOT:-${SOURCE_NEMORL}/.cache}"
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-false}"
export NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR:-/opt/ray_venvs}"
export NRL_VENVS_TRUST_EXISTING="${NRL_VENVS_TRUST_EXISTING:-1}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
KNOWN_GOOD_VLLM_NEMORL="${KNOWN_GOOD_VLLM_NEMORL:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/hanrongy/project/nemotron_omni/rl/nemo-rl_super/jobs/nomni_comboGRPO_v3_GA-SFT-MPO-TRL1-visG125-ckpt_GA-IRL-Filtered-AllImg3_nrt_fix0417_n32_bs4096_ga16_jvllmfix-20260520-135224/code}"
VLLM_NEMORL="${VLLM_NEMORL:-${KNOWN_GOOD_VLLM_NEMORL}}"
VLLM_NEMORL="$(cd "${VLLM_NEMORL}" && pwd)"

export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-${HF_HOME}/nemo_rl}"
export TMPDIR="${TMPDIR:-/tmp/nrl-${JOB_HASH}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR}/triton}"
mkdir -p "${HF_HOME}" "${HF_MODULES_CACHE}" "${NRL_MEGATRON_CHECKPOINT_DIR}" "${TMPDIR}" "${TRITON_CACHE_DIR}"

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NVTE_FWD_LAYERNORM_SM_MARGIN="${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}"
export NVTE_BWD_LAYERNORM_SM_MARGIN="${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}"
export NEMO_RL_LOG_GPU_MEMORY="${NEMO_RL_LOG_GPU_MEMORY:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NRL_IGNORE_VERSION_MISMATCH="${NRL_IGNORE_VERSION_MISMATCH:-true}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800000}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export TORCH_FR_BUFFER_SIZE="${TORCH_FR_BUFFER_SIZE:-1000}"
export NRL_DEBUG="${NRL_DEBUG:-0}"
export USE_REPO_VLLM="${USE_REPO_VLLM:-1}"
export NRL_VIDEO_PROMPT_STYLE="${NRL_VIDEO_PROMPT_STYLE:-sft_v2_grouped}"
export NRL_VIDEO_SAMPLING_STYLE="${NRL_VIDEO_SAMPLING_STYLE:-sft_v2_duration}"
export NRL_VIDEO_SFT_MIN_FRAMES="${NUM_FRAMES}"
export NRL_VIDEO_SFT_MAX_FRAMES="${NUM_FRAMES}"
export NRL_VIDEO_SFT_DEFAULT_FPS="${NRL_VIDEO_SFT_DEFAULT_FPS:-2}"

SEED="${SEED:-$(printf '%s' "train:${JOB_NAME}" | openssl dgst -md5 -binary | od -An -tu4 -N4 | xargs)}"
WANDB_PROJECT="${WANDB_PROJECT:-Nemotron-omni-RL}"
MAX_TOTAL_SEQUENCE_LENGTH="${MAX_TOTAL_SEQUENCE_LENGTH:-24576}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.6}"
VLLM_LOAD_FORMAT="${VLLM_LOAD_FORMAT:-auto}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-false}"
OFFLOAD_OPTIMIZER_FOR_LOGPROB="${OFFLOAD_OPTIMIZER_FOR_LOGPROB:-false}"
SEQUENCE_PACKING_ENABLED="${SEQUENCE_PACKING_ENABLED:-false}"

EXTRA_OVERRIDES=""
if [[ -n "${WANDB_RUN_ID:-}" ]]; then
  EXTRA_OVERRIDES+=" +logger.wandb.id=${WANDB_RUN_ID} +logger.wandb.resume=${WANDB_RESUME:-must}"
fi
if [[ -n "${EXTRA_HYDRA_OVERRIDES:-}" ]]; then
  EXTRA_OVERRIDES+=" ${EXTRA_HYDRA_OVERRIDES}"
fi

SNAPSHOT_CODE_LOWER="${SNAPSHOT_CODE,,}"
if [[ "${SNAPSHOT_CODE_LOWER}" == "1" || "${SNAPSHOT_CODE_LOWER}" == "true" || "${SNAPSHOT_CODE_LOWER}" == "yes" ]]; then
  SNAPSHOT_NEMORL="${SNAPSHOT_NEMORL:-${RESULTS_DIR}/code}"
  mkdir -p "${SNAPSHOT_NEMORL}"
  SNAPSHOT_NEMORL="$(cd "${SNAPSHOT_NEMORL}" && pwd)"
  if [[ "${SNAPSHOT_NEMORL}" == "${SOURCE_NEMORL}" || "${SNAPSHOT_NEMORL}/" == "${SOURCE_NEMORL}/"* ]]; then
    echo "[ERROR] SNAPSHOT_NEMORL must be outside SOURCE_NEMORL to avoid recursive rsync: ${SNAPSHOT_NEMORL}" >&2
    exit 1
  fi

  echo "Snapshotting code from ${SOURCE_NEMORL} to ${SNAPSHOT_NEMORL}"
  RSYNC_EXCLUDES=(
    --exclude='.git/'
    --exclude='.env'
    --exclude='.venv/'
    --exclude='.cache/'
    --exclude='.tmp/'
    --exclude='.pytest_cache/'
    --exclude='.mypy_cache/'
    --exclude='.ruff_cache/'
    --exclude='__pycache__/'
    --exclude='*.pyc'
    --exclude='*.pyo'
    --exclude='*.out'
    --exclude='slurm-*.out'
    --exclude='wandb/'
    --exclude='logs/'
    --exclude='results/'
    --exclude='jobs/'
    --exclude='checkpoints/'
    --exclude='build/'
    --exclude='*.o'
    --exclude='*.a'
    --exclude='*.egg-info/'
    --exclude='scripts/omnirl_scripts/tmp_docs/'
  )
  rsync -a --delete "${RSYNC_EXCLUDES[@]}" "${SOURCE_NEMORL}/" "${SNAPSHOT_NEMORL}/"
  {
    echo "source_nemorl=${SOURCE_NEMORL}"
    echo "snapshot_nemorl=${SNAPSHOT_NEMORL}"
    echo "job_name=${JOB_NAME}"
    echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${SNAPSHOT_NEMORL}/.nemo_rl_snapshot_info"
  NEMORL="${SNAPSHOT_NEMORL}"
else
  NEMORL="${SOURCE_NEMORL}"
fi
export NEMORL

if [[ "${USE_REPO_VLLM}" != "1" ]]; then
  echo "[ERROR] USE_REPO_VLLM=0 is no longer supported by this launcher; container vLLM patching has been removed." >&2
  echo "[ERROR] Run tools/build-custom-vllm.sh and use the repo vLLM checkout at 3rdparty/vllm." >&2
  exit 1
fi
if [[ ! -f "${VLLM_NEMORL}/3rdparty/vllm/vllm/__init__.py" ]]; then
  echo "[ERROR] repo vLLM checkout missing at ${VLLM_NEMORL}/3rdparty/vllm." >&2
  echo "[ERROR] Run tools/build-custom-vllm.sh before launching this job." >&2
  exit 1
fi
if [[ -f "${VLLM_NEMORL}/3rdparty/vllm/nemo-rl.env" ]]; then
  # shellcheck disable=SC1091
  source "${VLLM_NEMORL}/3rdparty/vllm/nemo-rl.env"
fi
# The precompiled wheel location is build-time metadata from build-custom-vllm.sh.
# Do not leak it into vLLM20 runtime, where it is reported as an unknown env var.
unset VLLM_PRECOMPILED_WHEEL_LOCATION
export SETUP_COMMAND

PYTHONPATH_ROOTS="${VLLM_NEMORL}/3rdparty/vllm:${NEMORL}:${NEMORL}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${NEMORL}/3rdparty/Megatron-LM-workspace/Megatron-LM"

export COMMAND="\
mkdir -p '${HF_HOME}' '${HF_MODULES_CACHE}' '${NRL_MEGATRON_CHECKPOINT_DIR}' '${TRITON_CACHE_DIR}' '${TMPDIR}' '${RESULTS_DIR}' && \
export PYTHONPATH=${PYTHONPATH_ROOTS}\${PYTHONPATH:+:\$PYTHONPATH} && \
uv run --no-sync examples/run_vlm_grpo.py --config '${CONFIG_PATH}' \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
grpo.seed=${SEED} \
grpo.num_prompts_per_step=256 \
grpo.seq_logprob_error_threshold=50 \
grpo.zero_variance_prompt_filtering=true \
grpo.val_at_end=false \
policy.train_global_batch_size=${GLOBAL_TRAIN_BATCH_SIZE} \
policy.train_micro_batch_size=${MICRO_BS} \
policy.logprob_batch_size=${LOGPROB_BS} \
policy.offload_optimizer_for_logprob=${OFFLOAD_OPTIMIZER_FOR_LOGPROB} \
policy.sequence_packing.enabled=${SEQUENCE_PACKING_ENABLED} \
	policy.max_total_sequence_length=${MAX_TOTAL_SEQUENCE_LENGTH} \
	policy.model_name='${MODEL_NAME}' \
policy.megatron_cfg.freeze_vision_model=true \
policy.megatron_cfg.freeze_vision_projection=true \
policy.megatron_cfg.freeze_sound_encoder=true \
	policy.megatron_cfg.freeze_sound_projection=true \
	policy.megatron_cfg.scheduler.lr_warmup_iters=3 \
		policy.generation.vllm_cfg.gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION} \
		+policy.generation.vllm_cfg.load_format=${VLLM_LOAD_FORMAT} \
		+policy.generation.vllm_cfg.enable_prefix_caching=${VLLM_ENABLE_PREFIX_CACHING} \
		data.train.train_data_path='${TRAIN_DATA_PATH}' \
	data.default.num_frames=${NUM_FRAMES} \
	data.default.max_images_per_prompt=${NUM_FRAMES} \
	policy.generation.vllm_kwargs.limit_mm_per_prompt.image=${NUM_FRAMES} \
	checkpointing.checkpoint_dir='${RESULTS_DIR}/checkpoints' \
checkpointing.save_period=5 \
logger.log_dir='${RESULTS_DIR}/nemorl_logs' \
logger.wandb_enabled=true \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${JOB_NAME}' \
+policy.megatron_cfg.checkpoint.async_save=false \
+policy.megatron_cfg.freeze_embedding=false\
${EXTRA_OVERRIDES}"

cd "${NEMORL}"

SBATCH_ARRAY_ARGS=()
if [[ "${JOB_CYCLES}" -gt 0 ]]; then
  SBATCH_ARRAY_ARGS+=(--array="0-${JOB_CYCLES}%1" --dependency=singleton)
fi
SBATCH_MEMORY_ARGS=()
if [[ -n "${SBATCH_MEM:-}" ]]; then
  SBATCH_MEMORY_ARGS+=(--mem="${SBATCH_MEM}")
fi
if [[ -n "${SBATCH_MEM_PER_GPU:-}" ]]; then
  SBATCH_MEMORY_ARGS+=(--mem-per-gpu="${SBATCH_MEM_PER_GPU}")
fi
SBATCH_EXTRA_ARGS_ARRAY=()
if [[ -n "${SBATCH_EXTRA_ARGS:-}" ]]; then
  read -r -a SBATCH_EXTRA_ARGS_ARRAY <<< "${SBATCH_EXTRA_ARGS}"
fi

echo "JOB_NAME=${JOB_NAME}"
echo "NUM_NODES=${NUM_NODES}"
echo "NUM_FRAMES=${NUM_FRAMES}"
echo "NRL_VIDEO_SFT_MIN_FRAMES=${NRL_VIDEO_SFT_MIN_FRAMES}"
echo "NRL_VIDEO_SFT_MAX_FRAMES=${NRL_VIDEO_SFT_MAX_FRAMES}"
echo "GLOBAL_TRAIN_BATCH_SIZE=${GLOBAL_TRAIN_BATCH_SIZE}"
echo "SBATCH_ACCOUNT=${SBATCH_ACCOUNT}"
echo "SBATCH_PARTITION=${SBATCH_PARTITION}"
echo "SBATCH_MEM=${SBATCH_MEM:-}"
echo "SBATCH_MEM_PER_GPU=${SBATCH_MEM_PER_GPU:-}"
echo "SBATCH_EXTRA_ARGS=${SBATCH_EXTRA_ARGS:-}"
echo "CONTAINER=${CONTAINER}"
echo "RESULTS_DIR=${RESULTS_DIR}"
echo "SOURCE_NEMORL=${SOURCE_NEMORL}"
echo "NEMORL=${NEMORL}"
echo "VLLM_NEMORL=${VLLM_NEMORL}"
echo "SNAPSHOT_CODE=${SNAPSHOT_CODE}"

sbatch \
  --nodes="${NUM_NODES}" \
  --account="${SBATCH_ACCOUNT}" \
  --job-name="${JOB_NAME}" \
  --partition="${SBATCH_PARTITION}" \
  --time="${SBATCH_TIME}" \
  --gres="gpu:${GPUS_PER_NODE}" \
  --output="${LOGS_DIR}/%x_%A_%a.log" \
  "${SBATCH_MEMORY_ARGS[@]}" \
  "${SBATCH_EXTRA_ARGS_ARRAY[@]}" \
  "${SBATCH_ARRAY_ARGS[@]}" \
  ray.sub
