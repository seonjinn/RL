#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUSTRE_ROOT="${LUSTRE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}"
HF_HOME="${HF_HOME:-${LUSTRE_ROOT}/hf_home}"
MODELS="${MODELS:-qwen30ba3b qwen32 qwen235b}"
DOMAINS="${DOMAINS:-Math SWE}"
VARIANTS="${VARIANTS:-baseline static dynamic}"
TEMPERATURES="${TEMPERATURES:-0.0 1.0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_legacy_0619_replay}"
RESULT_ROOT="${RESULT_ROOT:-${LUSTRE_ROOT}/vllm024-dynamicsd/legacy-0619-replay}"
SMOKE="${SMOKE:-true}"
DRY_RUN="${DRY_RUN:-false}"
TEST_ONLY="${TEST_ONLY:-false}"
REQUIRE_GIT_PULL="${REQUIRE_GIT_PULL:-true}"

if [[ "${SMOKE}" == "true" ]]; then
  BATCH_SIZES="${BATCH_SIZES:-4}"
  OSL="${OSL:-256}"
else
  BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16 32}"
  OSL="${OSL:-32768}"
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

  for model_key in ${MODELS}; do
    case "${model_key}" in
      qwen30ba3b)
        model="${HF_HOME}/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
        draft_model="${HF_HOME}/hub/models--RedHatAI--Qwen3-30B-A3B-Thinking-2507-speculator.eagle3/snapshots/a7ec796dd65236f1ecd4ed2958a7f0689e5da5cf"
        tp=1
        kv_cache_dtype=auto
        gpu_memory_utilization=0.86
        ;;
      qwen32)
        model="${HF_HOME}/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137"
        draft_model="${HF_HOME}/hub/models--RedHatAI--Qwen3-32B-speculator.eagle3/snapshots/dc84fe7ff1db31efa824776f49c141fc8195eb47"
        tp=2
        kv_cache_dtype=auto
        if [[ "${domain}" == "SWE" ]]; then
          gpu_memory_utilization=0.90
        else
          gpu_memory_utilization=0.86
        fi
        ;;
      qwen235b)
        model="${HF_HOME}/hub/models--Qwen--Qwen3-235B-A22B/snapshots/8efa61729e24bd65b1d152b5ab5409052aa80e65"
        draft_model="${HF_HOME}/hub/models--nvidia--Qwen3-235B-A22B-Eagle3/snapshots/33f3c01ce807376d1171301b9a148b1b28f239ba"
        tp=4
        kv_cache_dtype=fp8
        gpu_memory_utilization=0.94
        ;;
      *)
        echo "Unsupported model: ${model_key}" >&2
        exit 2
        ;;
    esac

    for batch_size in ${BATCH_SIZES}; do
      echo "domain=${domain}"
      echo "model=${model_key}"
      echo "batch_size=${batch_size}"
      echo "isl=4096"
      echo "osl=${OSL}"
      echo "target_tp=${tp}"
      echo "throughput_gpu_count=4"
      echo "enforce_eager=true"

      env \
        CLUSTER="${CLUSTER:-auto}" \
        ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}" \
        PARTITION="${PARTITION:-gb200}" \
        LUSTRE_ROOT="${LUSTRE_ROOT}" \
        HF_HOME="${HF_HOME}" \
        MODEL="${model}" \
        DRAFT_MODEL="${draft_model}" \
        RESULT_ROOT="${RESULT_ROOT}/${RUN_ID}/${domain_label}/${model_key}/bs${batch_size}" \
        RUN_ID=matrix \
        JOB_LABEL="0619-${domain_label}-${model_key}-b${batch_size}" \
        VARIANTS="${VARIANTS}" \
        TEMPERATURES="${TEMPERATURES}" \
        STATIC_K=3 \
        DYNAMIC_SCHEDULE="1:16:5,17:32:4,33:64:3,65:128:1,129:512:0" \
        TP="${tp}" \
        PP=1 \
        ISL=4096 \
        OSL="${OSL}" \
        BATCH_SIZES="${batch_size}" \
        TOP_P=1.0 \
        SMOKE=false \
        GPU_MEMORY_UTILIZATION="${gpu_memory_utilization}" \
        KV_CACHE_DTYPE="${kv_cache_dtype}" \
        MAX_MODEL_LEN=40960 \
        MAX_NUM_BATCHED_TOKENS="${max_num_batched_tokens}" \
        CUDAGRAPH_MODE=NONE \
        ATTENTION_BACKEND=TRITON_ATTN \
        MOE_BACKEND=triton \
        ENFORCE_EAGER=true \
        DISABLE_CUSTOM_ALL_REDUCE=true \
        THROUGHPUT_GPU_COUNT=4 \
        PROMPT_JSONL="${prompt_jsonl}" \
        PROMPT_OFFSET=0 \
        WARMUP_REPEATS=1 \
        MEASURE_REPEATS=1 \
        TIME_LIMIT="${TIME_LIMIT:-05:00:00}" \
        DRY_RUN="${DRY_RUN}" \
        TEST_ONLY="${TEST_ONLY}" \
        REQUIRE_GIT_PULL=false \
        "${SCRIPT_DIR}/submit_matrix.sh"
    done
  done
done
