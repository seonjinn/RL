#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUSTRE_ROOT="${LUSTRE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}"
HF_HOME="${HF_HOME:-${LUSTRE_ROOT}/hf_home}"
MODEL="${MODEL:-${HF_HOME}/hub/models--nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16/snapshots/624ba927cfbef0427354998700de3d51173c8c04}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${LUSTRE_ROOT}/containers/vllm-openai-v0.24.0-aarch64-ubuntu2404.sqsh}"
RAY_SITE="${RAY_SITE:-${LUSTRE_ROOT}/vllm024-dynamicsd/python-sites/ray-2.55.1-py312}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_ultra_bf16_mtp}"
RESULT_ROOT="${RESULT_ROOT:-${LUSTRE_ROOT}/vllm024-dynamicsd/nemotron-ultra-bf16/${RUN_ID}}"
STATIC_K_VALUES="${STATIC_K_VALUES:-1 3 5 6 7}"
DYNAMIC_SCHEDULE="${DYNAMIC_SCHEDULE:-1:1:6,2:2:5,3:4:3,5:8:2,9:32:1}"
TEMPERATURES="${TEMPERATURES:-0 1}"
SMOKE="${SMOKE:-true}"
DRY_RUN="${DRY_RUN:-false}"
TEST_ONLY="${TEST_ONLY:-false}"
REQUIRE_GIT_PULL="${REQUIRE_GIT_PULL:-true}"

if [[ "${SMOKE}" == "true" ]]; then
  ISL="${ISL:-1024}"
  OSL="${OSL:-128}"
  BATCH_SIZES="${BATCH_SIZES:-1 2}"
  MEASURE_REPEATS="${MEASURE_REPEATS:-1}"
  TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
else
  ISL="${ISL:-10240}"
  OSL="${OSL:-16384}"
  BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16 32}"
  MEASURE_REPEATS="${MEASURE_REPEATS:-3}"
  TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
fi
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((ISL + OSL + 256))}"

if [[ "${DRY_RUN}" != "true" && "${REQUIRE_GIT_PULL}" == "true" ]]; then
  git -C "${SCRIPT_DIR}" pull --ff-only
fi

run_variant() {
  local temperature="$1"
  local top_p="$2"
  local variant="$3"
  local label="$4"
  local static_k="$5"

  env \
    CLUSTER="${CLUSTER:-auto}" \
    ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}" \
    PARTITION="${PARTITION:-batch}" \
    LUSTRE_ROOT="${LUSTRE_ROOT}" \
    HF_HOME="${HF_HOME}" \
    MODEL="${MODEL}" \
    DRAFT_MODEL="" \
    CONTAINER_IMAGE="${CONTAINER_IMAGE}" \
    RAY_SITE="${RAY_SITE}" \
    RESULT_ROOT="${RESULT_ROOT}" \
    RUN_ID="${label}" \
    JOB_LABEL="ultra-bf16-${label//\//-}" \
    VARIANTS="${variant}" \
    TEMPERATURES="${temperature}" \
    TOP_P="${top_p}" \
    STATIC_K="${static_k}" \
    DYNAMIC_SCHEDULE="${DYNAMIC_SCHEDULE}" \
    TP=8 \
    PP=1 \
    NODES=2 \
    SEGMENT=2 \
    DISTRIBUTED_EXECUTOR_BACKEND=ray \
    DIST_TIMEOUT_SECONDS=3600 \
    ENABLE_EXPERT_PARALLEL=true \
    MODEL_LOADER_NUM_THREADS=96 \
    DISABLE_FUSE_ALLREDUCE_RMS=true \
    MAMBA_SSM_CACHE_DTYPE=float16 \
    MAMBA_BACKEND=flashinfer \
    ENABLE_MAMBA_CACHE_STOCHASTIC_ROUNDING=true \
    MAMBA_CACHE_PHILOX_ROUNDS=5 \
    THROUGHPUT_GPU_COUNT=8 \
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}" \
    KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}" \
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}" \
    MOE_BACKEND="${MOE_BACKEND:-flashinfer_trtllm}" \
    CUDAGRAPH_MODE=PIECEWISE \
    ENFORCE_EAGER=false \
    ISL="${ISL}" \
    OSL="${OSL}" \
    BATCH_SIZES="${BATCH_SIZES}" \
    MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    WARMUP_REPEATS="${WARMUP_REPEATS:-1}" \
    MEASURE_REPEATS="${MEASURE_REPEATS}" \
    TIME_LIMIT="${TIME_LIMIT}" \
    SMOKE="${SMOKE}" \
    DRY_RUN="${DRY_RUN}" \
    TEST_ONLY="${TEST_ONLY}" \
    REQUIRE_GIT_PULL=false \
    "${SCRIPT_DIR}/submit_matrix.sh"
}

for temperature in ${TEMPERATURES}; do
  case "${temperature}" in
    0|0.0) top_p=1.0 ;;
    1|1.0) top_p=0.95 ;;
    *)
      echo "Unsupported temperature profile: ${temperature}" >&2
      exit 2
      ;;
  esac
  temp_label="t$(printf '%s' "${temperature}" | tr '.' 'p')"
  run_variant "${temperature}" "${top_p}" baseline "${temp_label}/baseline" 1
  for static_k in ${STATIC_K_VALUES}; do
    run_variant \
      "${temperature}" "${top_p}" mtp_static \
      "${temp_label}/mtp_k${static_k}" "${static_k}"
  done
  run_variant \
    "${temperature}" "${top_p}" mtp_dynamic \
    "${temp_label}/mtp_dynamic" 1
done
