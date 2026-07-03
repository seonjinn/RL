#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUSTRE_ROOT="${LUSTRE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}"
HF_HOME="${HF_HOME:-${LUSTRE_ROOT}/hf_home}"
ASSET_ROOT="${ASSET_ROOT:-${LUSTRE_ROOT}/vllm024-dynamicsd/assets}"
MODEL="${MODEL:-${HF_HOME}/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
PARD_MODEL="${PARD_MODEL:-${HF_HOME}/hub/models--amd--PARD-Qwen3-0.6B/snapshots/f9f650fbab180c26498817718f0db5cae8f25136}"
PARD2_MODEL="${PARD2_MODEL:-${HF_HOME}/hub/models--amd--PARD2-Qwen3-8B/snapshots/67a1516c8f6fc145cda99916799a0cbb3a4af135}"
DFLASH_MODEL="${DFLASH_MODEL:-${HF_HOME}/hub/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/9b41424b7109f9c5413454f481b09a82b85333f4}"
COMMON_SITE="${COMMON_SITE:-${ASSET_ROOT}/python/common}"
PARD2_OVERLAY="${PARD2_OVERLAY:-${ASSET_ROOT}/python/pard2_overlay}"
METHODS="${METHODS:-baseline suffix pard pard2 dflash}"
DOMAINS="${DOMAINS:-Math SWE}"
TEMPERATURES="${TEMPERATURES:-0.0 1.0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_qwen8_extended}"
RESULT_ROOT="${RESULT_ROOT:-${LUSTRE_ROOT}/vllm024-dynamicsd/extended-methods}"
SMOKE="${SMOKE:-true}"
DRY_RUN="${DRY_RUN:-false}"
TEST_ONLY="${TEST_ONLY:-false}"
REQUIRE_GIT_PULL="${REQUIRE_GIT_PULL:-true}"

if [[ "${SMOKE}" == "true" ]]; then
  BATCH_SIZES="${BATCH_SIZES:-4}"
  OSL="${OSL:-256}"
  TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
else
  BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16 32}"
  OSL="${OSL:-32768}"
  TIME_LIMIT="${TIME_LIMIT:-05:00:00}"
fi

if [[ "${DRY_RUN}" != "true" && "${REQUIRE_GIT_PULL}" == "true" ]]; then
  git -C "${SCRIPT_DIR}" pull --ff-only
fi

for domain in ${DOMAINS}; do
  case "${domain}" in
    Math)
      prompt_jsonl="${MATH_PROMPT_JSONL:-${LUSTRE_ROOT}/vllm024-dynamicsd/datasets/math_500_data_prompts_qmath_20260617.jsonl}"
      max_num_batched_tokens=131072
      domain_label=math
      ;;
    SWE)
      prompt_jsonl="${SWE_PROMPT_JSONL:-${LUSTRE_ROOT}/vllm-benchmark/data/swebench_verified_prompts_all.jsonl}"
      max_num_batched_tokens=65536
      domain_label=swe
      ;;
    *)
      echo "Unsupported domain: ${domain}" >&2
      exit 2
      ;;
  esac

  for method in ${METHODS}; do
    case "${method}" in
      baseline)
        draft_model=""
        static_k=1
        extra_pythonpath="${COMMON_SITE}"
        ;;
      suffix)
        draft_model=""
        static_k=32
        extra_pythonpath="${COMMON_SITE}"
        ;;
      pard)
        draft_model="${PARD_MODEL}"
        static_k=12
        extra_pythonpath="${COMMON_SITE}"
        ;;
      pard2)
        draft_model="${PARD2_MODEL}"
        static_k=15
        extra_pythonpath="${PARD2_OVERLAY}:${COMMON_SITE}"
        ;;
      dflash)
        draft_model="${DFLASH_MODEL}"
        static_k=15
        extra_pythonpath="${COMMON_SITE}"
        ;;
      *)
        echo "Unsupported method: ${method}" >&2
        exit 2
        ;;
    esac

    echo "domain=${domain}"
    echo "method=${method}"
    echo "isl=4096"
    echo "osl=${OSL}"
    echo "batch_sizes=${BATCH_SIZES}"

    env \
      CLUSTER="${CLUSTER:-auto}" \
      ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}" \
      PARTITION="${PARTITION:-gb200}" \
      LUSTRE_ROOT="${LUSTRE_ROOT}" \
      HF_HOME="${HF_HOME}" \
      MODEL="${MODEL}" \
      DRAFT_MODEL="${draft_model}" \
      RESULT_ROOT="${RESULT_ROOT}/${RUN_ID}/${domain_label}/${method}" \
      RUN_ID=matrix \
      JOB_LABEL="q8-${domain_label}-${method}" \
      VARIANTS="${method}" \
      TEMPERATURES="${TEMPERATURES}" \
      STATIC_K="${static_k}" \
      DRAFT_TP=1 \
      TP=1 \
      PP=1 \
      ISL=4096 \
      OSL="${OSL}" \
      BATCH_SIZES="${BATCH_SIZES}" \
      TOP_P=1.0 \
      SMOKE=false \
      GPU_MEMORY_UTILIZATION=0.90 \
      KV_CACHE_DTYPE=auto \
      MAX_MODEL_LEN=40960 \
      MAX_NUM_BATCHED_TOKENS="${max_num_batched_tokens}" \
      CUDAGRAPH_MODE=NONE \
      ATTENTION_BACKEND=TRITON_ATTN \
      ENFORCE_EAGER=true \
      DISABLE_CUSTOM_ALL_REDUCE=true \
      THROUGHPUT_GPU_COUNT=4 \
      PROMPT_JSONL="${prompt_jsonl}" \
      PROMPT_OFFSET=0 \
      WARMUP_REPEATS=1 \
      MEASURE_REPEATS=1 \
      SUFFIX_MAX_CACHED_REQUESTS=10000 \
      SUFFIX_MAX_SPEC_FACTOR=1.0 \
      SUFFIX_MIN_TOKEN_PROB=0.1 \
      EXTRA_PYTHONPATH="${extra_pythonpath}" \
      TIME_LIMIT="${TIME_LIMIT}" \
      DEPENDENCY="${DEPENDENCY:-}" \
      DRY_RUN="${DRY_RUN}" \
      TEST_ONLY="${TEST_ONLY}" \
      REQUIRE_GIT_PULL=false \
      "${SCRIPT_DIR}/submit_matrix.sh"
  done
done
