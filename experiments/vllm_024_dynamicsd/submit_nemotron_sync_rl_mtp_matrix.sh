#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUSTRE_ROOT="${LUSTRE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}"
HF_HOME="${HF_HOME:-${LUSTRE_ROOT}/hf_home}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${LUSTRE_ROOT}/containers/vllm-openai-v0.24.0-aarch64-ubuntu2404.sqsh}"
RAY_SITE="${RAY_SITE:-${LUSTRE_ROOT}/vllm024-dynamicsd/python-sites/ray-2.55.1-py312}"
MODELS="${MODELS:-ultra super}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_nemotron_sync_rl_mtp}"
RESULT_ROOT="${RESULT_ROOT:-${LUSTRE_ROOT}/vllm024-dynamicsd/nemotron-sync-rl/${RUN_ID}}"
VARIANTS="${VARIANTS:-baseline mtp_static mtp_dynamic}"
SMOKE="${SMOKE:-true}"
PROMPT_JSONL="${PROMPT_JSONL:-}"
DRY_RUN="${DRY_RUN:-false}"
TEST_ONLY="${TEST_ONLY:-false}"
REQUIRE_GIT_PULL="${REQUIRE_GIT_PULL:-true}"

if [[ "${SMOKE}" == "true" ]]; then
  NUM_PROMPTS="${NUM_PROMPTS:-4}"
  SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-4}"
  ROLLOUT_BATCHES="${ROLLOUT_BATCHES:-2}"
  MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1024}"
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
  ENGINE_MAX_NUM_SEQS="${ENGINE_MAX_NUM_SEQS:-16}"
  TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
else
  NUM_PROMPTS="${NUM_PROMPTS:-16}"
  SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-16}"
  ROLLOUT_BATCHES="${ROLLOUT_BATCHES:-3}"
  MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-4096}"
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
  ENGINE_MAX_NUM_SEQS="${ENGINE_MAX_NUM_SEQS:-64}"
  TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
  if [[ -z "${PROMPT_JSONL}" ]]; then
    echo "SMOKE=false requires PROMPT_JSONL from a pinned RL math dataset" >&2
    exit 2
  fi
fi
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_TOKENS + MAX_NEW_TOKENS + 256))}"

if [[ "${DRY_RUN}" != "true" && "${REQUIRE_GIT_PULL}" == "true" ]]; then
  git -C "${SCRIPT_DIR}" pull --ff-only
fi

for model_key in ${MODELS}; do
  case "${model_key}" in
    ultra)
      model="${HF_HOME}/hub/models--nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16/snapshots/624ba927cfbef0427354998700de3d51173c8c04"
      static_k=5
      dynamic_schedule="1:4:5,5:8:3,9:16:2,17:256:1"
      tp=8
      nodes=2
      segment=2
      distributed_backend=ray
      enable_expert_parallel=true
      mamba_ssm_cache_dtype=float16
      enable_stochastic_rounding=true
      mamba_philox_rounds=5
      model_loader_threads=96
      disable_fuse_allreduce_rms=true
      ;;
    super)
      model="${HF_HOME}/hub/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16/snapshots/d51eab0d1f979ebc26b546e634a04f450d99158e"
      static_k=3
      dynamic_schedule="1:4:3,5:8:2,9:256:1"
      tp=2
      nodes=1
      segment=1
      distributed_backend=""
      enable_expert_parallel=false
      mamba_ssm_cache_dtype=float32
      enable_stochastic_rounding=false
      mamba_philox_rounds=""
      model_loader_threads=48
      disable_fuse_allreduce_rms=false
      ;;
    *)
      echo "Unsupported model: ${model_key}" >&2
      exit 2
      ;;
  esac

  env \
    CLUSTER="${CLUSTER:-auto}" \
    ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}" \
    PARTITION="${PARTITION:-batch}" \
    LUSTRE_ROOT="${LUSTRE_ROOT}" \
    HF_HOME="${HF_HOME}" \
    MODEL="${model}" \
    DRAFT_MODEL="" \
    CONTAINER_IMAGE="${CONTAINER_IMAGE}" \
    RAY_SITE="${RAY_SITE}" \
    RESULT_ROOT="${RESULT_ROOT}" \
    RUN_ID="${model_key}" \
    JOB_LABEL="sync-${model_key}-bf16" \
    VARIANTS="${VARIANTS}" \
    STATIC_K="${static_k}" \
    DYNAMIC_SCHEDULE="${dynamic_schedule}" \
    TP="${tp}" \
    PP=1 \
    NODES="${nodes}" \
    SEGMENT="${segment}" \
    DISTRIBUTED_EXECUTOR_BACKEND="${distributed_backend}" \
    DIST_TIMEOUT_SECONDS=3600 \
    ENABLE_EXPERT_PARALLEL="${enable_expert_parallel}" \
    MODEL_LOADER_NUM_THREADS="${model_loader_threads}" \
    DISABLE_FUSE_ALLREDUCE_RMS="${disable_fuse_allreduce_rms}" \
    MAMBA_SSM_CACHE_DTYPE="${mamba_ssm_cache_dtype}" \
    MAMBA_BACKEND=flashinfer \
    ENABLE_MAMBA_CACHE_STOCHASTIC_ROUNDING="${enable_stochastic_rounding}" \
    MAMBA_CACHE_PHILOX_ROUNDS="${mamba_philox_rounds}" \
    KV_CACHE_DTYPE=fp8 \
    MOE_BACKEND="${MOE_BACKEND:-flashinfer_trtllm}" \
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}" \
    CUDAGRAPH_MODE=PIECEWISE \
    ENGINE_MAX_NUM_SEQS="${ENGINE_MAX_NUM_SEQS}" \
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}" \
    TEMPERATURE=1.0 \
    TOP_P=0.95 \
    NUM_PROMPTS="${NUM_PROMPTS}" \
    SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT}" \
    ROLLOUT_BATCHES="${ROLLOUT_BATCHES}" \
    MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS}" \
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
    MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    PROMPT_JSONL="${PROMPT_JSONL}" \
    SOURCE_RECIPE="sync-rl-math-rollout" \
    SMOKE="${SMOKE}" \
    TIME_LIMIT="${TIME_LIMIT}" \
    DRY_RUN="${DRY_RUN}" \
    TEST_ONLY="${TEST_ONLY}" \
    REQUIRE_GIT_PULL=false \
    "${SCRIPT_DIR}/submit_sync_rollout.sh"
done
