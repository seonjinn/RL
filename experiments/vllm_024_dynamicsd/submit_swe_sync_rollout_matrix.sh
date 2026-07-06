#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUSTRE_ROOT="${LUSTRE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}"
HF_HOME="${HF_HOME:-${LUSTRE_ROOT}/hf_home}"
MODELS="${MODELS:-qwen32}"
REQUEST_PROFILES="${REQUEST_PROFILES:-32k}"
TEMPERATURES="${TEMPERATURES:-0.0 1.0}"
VARIANTS="${VARIANTS:-baseline static dynamic}"
RUN_ID_BASE="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_swe_sync}"
RESULT_ROOT="${RESULT_ROOT:-${LUSTRE_ROOT}/vllm024-dynamicsd/swe-sync-rollout}"
LONG_CONTEXT_VIEW_ROOT="${LONG_CONTEXT_VIEW_ROOT:-${LUSTRE_ROOT}/vllm024-dynamicsd/long-context-models/yarn4}"
SWE_PROMPT_JSONL="${SWE_PROMPT_JSONL:-${LUSTRE_ROOT}/vllm-benchmark/data/swebench_verified_prompts_all.jsonl}"
SMOKE="${SMOKE:-true}"
FULL_CONTRACT="${FULL_CONTRACT:-false}"
ROLLOUT_BATCHES="${ROLLOUT_BATCHES:-}"
DRY_RUN="${DRY_RUN:-false}"
TEST_ONLY="${TEST_ONLY:-false}"
REQUIRE_GIT_PULL="${REQUIRE_GIT_PULL:-true}"

render_command() {
  printf "%q " "$@"
  printf "\n"
}

materialize_long_context_views() {
  local target_view_name="$1"
  local target_source="$2"
  local draft_view_name="$3"
  local draft_source="$4"
  local command=(
    python3
    "${SCRIPT_DIR}/materialize_long_context_model_views.py"
    --view-root
    "${LONG_CONTEXT_VIEW_ROOT}"
    --max-position-embeddings
    131072
    --rope-factor
    4.0
    --model-view
    "${target_view_name}=${target_source}"
    --model-view
    "${draft_view_name}=${draft_source}"
  )
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf "[DRY-RUN] "
    render_command "${command[@]}"
    return
  fi
  "${command[@]}"
}

if [[ "${DRY_RUN}" != "true" && "${REQUIRE_GIT_PULL}" == "true" ]]; then
  git -C "${SCRIPT_DIR}" pull --ff-only
fi

for model_key in ${MODELS}; do
  case "${model_key}" in
    qwen30ba3b)
      model_source="${HF_HOME}/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
      draft_model_source="${HF_HOME}/hub/models--RedHatAI--Qwen3-30B-A3B-Thinking-2507-speculator.eagle3/snapshots/a7ec796dd65236f1ecd4ed2958a7f0689e5da5cf"
      target_tp=1
      gpu_memory_utilization=0.86
      ;;
    qwen32)
      model_source="${HF_HOME}/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137"
      draft_model_source="${HF_HOME}/hub/models--RedHatAI--Qwen3-32B-speculator.eagle3/snapshots/dc84fe7ff1db31efa824776f49c141fc8195eb47"
      target_tp=2
      gpu_memory_utilization=0.90
      ;;
    qwen235b)
      model_source="${HF_HOME}/hub/models--Qwen--Qwen3-235B-A22B/snapshots/8efa61729e24bd65b1d152b5ab5409052aa80e65"
      draft_model_source="${HF_HOME}/hub/models--nvidia--Qwen3-235B-A22B-Eagle3/snapshots/33f3c01ce807376d1171301b9a148b1b28f239ba"
      target_tp=8
      gpu_memory_utilization=0.94
      ;;
    *)
      echo "Unsupported model: ${model_key}" >&2
      exit 2
      ;;
  esac

  benchmark_nodes=$(((target_tp + 3) / 4))
  distributed_backend=""
  if (( benchmark_nodes > 1 )); then
    distributed_backend=ray
  fi

  for profile in ${REQUEST_PROFILES}; do
    case "${profile}" in
      32k)
        request_plan_host="${SCRIPT_DIR}/profiles/swe_sync_32k.json"
        request_plan_container="/workspace/experiment/profiles/swe_sync_32k.json"
        max_model_len=36864
        max_new_tokens=32768
        model="${model_source}"
        draft_model="${draft_model_source}"
        ;;
      64k)
        request_plan_host="${SCRIPT_DIR}/profiles/swe_sync_64k.json"
        request_plan_container="/workspace/experiment/profiles/swe_sync_64k.json"
        max_model_len=69632
        max_new_tokens=65536
        target_view_name="${model_key}-target"
        draft_view_name="${model_key}-eagle3-draft"
        materialize_long_context_views \
          "${target_view_name}" \
          "${model_source}" \
          "${draft_view_name}" \
          "${draft_model_source}"
        model="${LONG_CONTEXT_VIEW_ROOT}/${target_view_name}"
        draft_model="${LONG_CONTEXT_VIEW_ROOT}/${draft_view_name}"
        ;;
      *)
        echo "Unsupported REQUEST_PROFILES entry: ${profile}" >&2
        exit 2
        ;;
    esac

    request_plan_hash="$(
      PYTHONPATH="${SCRIPT_DIR}" python3 -c \
        'import sys; from pathlib import Path; from sync_rollout_core import load_request_plan; print(load_request_plan(Path(sys.argv[1])).plan_hash)' \
        "${request_plan_host}" 2>/dev/null || printf unknown
    )"

    if [[ "${SMOKE}" == "true" ]]; then
      num_prompts="${NUM_PROMPTS:-16}"
      samples_per_prompt="${SAMPLES_PER_PROMPT:-1}"
      effective_rollout_batches="${ROLLOUT_BATCHES:-1}"
      time_limit="${TIME_LIMIT:-02:00:00}"
    else
      num_prompts="${NUM_PROMPTS:-16}"
      if [[ "${FULL_CONTRACT}" == "true" ]]; then
        samples_per_prompt="${SAMPLES_PER_PROMPT:-16}"
      else
        samples_per_prompt="${SAMPLES_PER_PROMPT:-4}"
      fi
      effective_rollout_batches="${ROLLOUT_BATCHES:-3}"
      time_limit="${TIME_LIMIT:-06:00:00}"
    fi

    for temperature in ${TEMPERATURES}; do
      temperature_slug="$(printf '%s' "${temperature}" | tr '.' 'p')"
      echo "swe_sync_model=${model_key}"
      echo "request_profile=${profile}"
      echo "request_plan_hash=${request_plan_hash}"
      echo "temperature=${temperature}"
      echo "target_tp=${target_tp}"
      echo "max_model_len=${max_model_len}"
      echo "full_contract=${FULL_CONTRACT}"

      env \
        CLUSTER="${CLUSTER:-auto}" \
        ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}" \
        PARTITION="${PARTITION:-gb200}" \
        LUSTRE_ROOT="${LUSTRE_ROOT}" \
        HF_HOME="${HF_HOME}" \
        MODEL="${model}" \
        DRAFT_MODEL="${draft_model}" \
        RESULT_ROOT="${RESULT_ROOT}/${RUN_ID_BASE}/${model_key}/${profile}/t${temperature_slug}" \
        RUN_ID="matrix" \
        JOB_LABEL="swe-${model_key}-${profile}-t${temperature_slug}" \
        VARIANTS="${VARIANTS}" \
        STATIC_K="${STATIC_K:-5}" \
        DYNAMIC_SCHEDULE="${DYNAMIC_SCHEDULE:-1:16:5,17:32:4,33:64:3,65:128:1,129:512:0}" \
        TP="${target_tp}" \
        PP=1 \
        NODES="${benchmark_nodes}" \
        SEGMENT="${benchmark_nodes}" \
        DISTRIBUTED_EXECUTOR_BACKEND="${distributed_backend}" \
        TEMPERATURE="${temperature}" \
        TOP_P="${TOP_P:-0.95}" \
        GPU_MEMORY_UTILIZATION="${gpu_memory_utilization}" \
        CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-PIECEWISE}" \
        ENGINE_MAX_NUM_SEQS="${ENGINE_MAX_NUM_SEQS:-64}" \
        PROMPT_JSONL="${SWE_PROMPT_JSONL}" \
        PROMPT_OFFSET="${PROMPT_OFFSET:-0}" \
        REQUEST_PLAN="${request_plan_host}" \
        REQUEST_PLAN_IN_CONTAINER="${request_plan_container}" \
        RESOLVED_REQUEST_PLAN_OUTPUT=auto \
        RESPONSE_OUTPUT=auto \
        SOURCE_RECIPE="swe_sync_${profile}" \
        SMOKE="${SMOKE}" \
        NUM_PROMPTS="${num_prompts}" \
        SAMPLES_PER_PROMPT="${samples_per_prompt}" \
        ROLLOUT_BATCHES="${effective_rollout_batches}" \
        MAX_PROMPT_TOKENS=4096 \
        MAX_NEW_TOKENS="${max_new_tokens}" \
        MAX_MODEL_LEN="${max_model_len}" \
        MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}" \
        TIME_LIMIT="${time_limit}" \
        DRY_RUN="${DRY_RUN}" \
        TEST_ONLY="${TEST_ONLY}" \
        REQUIRE_GIT_PULL=false \
        "${SCRIPT_DIR}/submit_sync_rollout.sh"
    done
  done
done
