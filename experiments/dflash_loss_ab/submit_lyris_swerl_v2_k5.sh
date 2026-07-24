#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-dry-run}"
VARIANT="${VARIANT:-dflash_v2_k5}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"

SOURCE_ROOT="${SOURCE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-wt-nemogym-dynsd-lyris}"
LAUNCHER="${LAUNCHER:-/lustre/fsw/coreai_dlalgo_llm/users/sna/swerl_fullgrpo_launchers/20260721_swerl_235b_dflash_v1_smoke/run_swerl_235b_dflash_smoke.sh}"
EXPECTED_SOURCE_HEAD="${EXPECTED_SOURCE_HEAD:-b9c29565bde277e997eb969af9cc47da55ef4d16}"
EXPECTED_LAUNCHER_SHA256="${EXPECTED_LAUNCHER_SHA256:-9231e24d1065eec18746db726612acf483e44e900c0c525306feeaa66605671a}"

USER_ROOT="/lustre/fsw/coreai_dlalgo_llm/users/sna"
MODEL_PATH="${USER_ROOT}/hf_home/hub/models--Qwen--Qwen3-235B-A22B-Thinking-2507/snapshots/6cbffae6d8e28b986a6b17bd36f42f9fa0f1f0a5"
CONFIG_FILE="${SOURCE_ROOT}/examples/nemo_gym/grpo_qwen3_235b_thinking_swe2_smoke.yaml"
TRAIN_DATA_PATH="${SOURCE_ROOT}/data/swe2/train-pool224.jsonl"
VAL_DATA_PATH="${SOURCE_ROOT}/data/swe2/val-mini3.jsonl"
CONTAINER="${USER_ROOT}/containers/nemo_rl_nightly_20260715.sqsh"
VENV_ROOT="${USER_ROOT}/nrl_venvs_dynsd025"
PERSISTENT_CACHE="${USER_ROOT}/.cache/swerl_dflash_v2"
RUN_ROOT="${USER_ROOT}/swerl_fullgrpo_launchers/20260723_swerl_235b_dflash_v2_k5"

case "${VARIANT}" in
  baseline)
    specdec_args=""
    ;;
  dflash_v2_k5)
    specdec_args="++policy.generation.vllm_kwargs.speculative_config.method=dflash \
++policy.generation.vllm_kwargs.speculative_config.model=/home/sna/drafters/dflash_235bthink_v2 \
++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1 \
++policy.generation.vllm_kwargs.speculative_config.max_model_len=4096 \
++policy.generation.vllm_kwargs.speculative_config.attention_backend=FLASH_ATTN \
++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=5 \
++policy.generation.vllm_kwargs.kernel_config.enable_flashinfer_autotune=false \
++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=FULL \
++policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes=[6,12,24,48,96]"
    ;;
  *)
    echo "VARIANT must be baseline or dflash_v2_k5; got ${VARIANT}" >&2
    exit 2
    ;;
esac

actual_head="$(git -C "${SOURCE_ROOT}" rev-parse HEAD)"
if [[ "${actual_head}" != "${EXPECTED_SOURCE_HEAD}" ]]; then
  echo "Unexpected source HEAD: ${actual_head}; expected ${EXPECTED_SOURCE_HEAD}" >&2
  exit 2
fi

actual_launcher_sha256="$(sha256sum "${LAUNCHER}" | awk '{print $1}')"
if [[ "${actual_launcher_sha256}" != "${EXPECTED_LAUNCHER_SHA256}" ]]; then
  echo "Unexpected launcher SHA256: ${actual_launcher_sha256}" >&2
  exit 2
fi

for required_path in \
  "${CONFIG_FILE}" \
  "${TRAIN_DATA_PATH}" \
  "${VAL_DATA_PATH}" \
  "${CONTAINER}" \
  "${MODEL_PATH}" \
  "${VENV_ROOT}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Required path is missing: ${required_path}" >&2
    exit 2
  fi
done

if [[ "${MODE}" == "test-only" ]]; then
  sbatch \
    --test-only \
    --nodes=16 \
    --account=coreai_dlalgo_llm \
    --job-name="coreai_dlalgo_llm-swerl.${VARIANT}" \
    --partition=gb200 \
    --time=02:30:00 \
    --segment=8 \
    --exclude=lyris0264 \
    --exclusive \
    "${SOURCE_ROOT}/ray.sub"
  exit 0
fi

if [[ "${MODE}" != "dry-run" && "${MODE}" != "submit" ]]; then
  echo "MODE must be dry-run, test-only, or submit; got ${MODE}" >&2
  exit 2
fi

dry_run=1
if [[ "${MODE}" == "submit" ]]; then
  dry_run=0
fi

common_args="logger.wandb_enabled=False \
checkpointing.enabled=false \
env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.concurrency=8 \
env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.concurrency=4"
extra_args="${common_args} ${specdec_args}"
exp_suffix="dflashv2-swerl1-${VARIANT}-${RUN_TAG}"
base_log_dir="${RUN_ROOT}/${exp_suffix}/logs"

mkdir -p "${base_log_dir}"

env \
  REPO_ROOT="${SOURCE_ROOT}" \
  CONFIG_FILE="${CONFIG_FILE}" \
  TRAIN_DATA_PATH="${TRAIN_DATA_PATH}" \
  VAL_DATA_PATH="${VAL_DATA_PATH}" \
  MODEL_PATH="${MODEL_PATH}" \
  CONTAINER="${CONTAINER}" \
  NEMO_RL_VENV_DIR="${VENV_ROOT}" \
  NRL_FORCE_REBUILD_VENVS=false \
  HYBRIDEP=0 \
  NUM_TRAIN_REPLICAS=1 \
  GEN_TRAIN_RATIO=1 \
  VLLM_TP=8 \
  GPP=4 \
  SAMPLES_PER_NODE=1 \
  CPUS_PER_WORKER=64 \
  MAX_NUM_STEPS=1 \
  SBATCH_TIME=02:30:00 \
  SBATCH_SEGMENT=8 \
  EXTRA_CONTAINER_MOUNTS="/home/sna:/home/sna,/dev/fuse:/dev/fuse" \
  PERSISTENT_CACHE="${PERSISTENT_CACHE}" \
  UV_CACHE_SCOPE=exp_suffix \
  EXP_SUFFIX="${exp_suffix}" \
  WANDB_GROUP="dflash-v2-swerl-full-step" \
  BASE_LOG_DIR="${base_log_dir}" \
  EXTRA_ARGS="${extra_args}" \
  DRY_RUN="${dry_run}" \
  bash "${LAUNCHER}"
