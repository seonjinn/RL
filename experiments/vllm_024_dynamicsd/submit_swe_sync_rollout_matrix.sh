#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATRIX_FILE="${MATRIX_FILE:-${SCRIPT_DIR}/model_method_matrix.json}"
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

MANIFEST_ROOT="${RESULT_ROOT}/${RUN_ID_BASE}"
MANIFEST="${MANIFEST_ROOT}/jobs.tsv"
TEMP_MANIFEST_ROOT=""

render_command() {
  printf "%q " "$@"
  printf "\n"
}

variant_requested() {
  [[ " ${VARIANTS} " == *" $1 "* ]]
}

record_manifest_row() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" >>"${MANIFEST}"
}

cleanup_temp_manifest() {
  if [[ -n "${TEMP_MANIFEST_ROOT}" ]]; then
    if [[ -f "${MANIFEST}" ]]; then
      cat "${MANIFEST}"
    fi
    rm -rf "${TEMP_MANIFEST_ROOT}"
  fi
}

clear_qwen_matrix_state() {
  unset \
    model_source \
    draft_model_source \
    target_tp \
    benchmark_nodes \
    distributed_backend \
    gpu_memory_utilization \
    request_plan_host \
    request_plan_container \
    profile_context_policy \
    max_model_len \
    max_new_tokens \
    max_position_embeddings \
    rope_factor \
    target_view_name \
    draft_view_name \
    method_baseline_status \
    method_baseline_reason_code \
    method_baseline_reason \
    method_baseline_variants \
    method_eagle3_status \
    method_eagle3_reason_code \
    method_eagle3_reason \
    method_eagle3_variants \
    method_pard_status \
    method_pard_reason_code \
    method_pard_reason \
    method_pard_variants \
    method_pard2_status \
    method_pard2_reason_code \
    method_pard2_reason \
    method_pard2_variants \
    method_dflash_status \
    method_dflash_reason_code \
    method_dflash_reason \
    method_dflash_variants \
    method_dflare_status \
    method_dflare_reason_code \
    method_dflare_reason \
    method_dflare_variants \
    method_mtp_static_status \
    method_mtp_static_reason_code \
    method_mtp_static_reason \
    method_mtp_static_variants \
    method_mtp_dynamic_status \
    method_mtp_dynamic_reason_code \
    method_mtp_dynamic_reason \
    method_mtp_dynamic_variants
}

load_qwen_matrix_state() {
  local model_key="$1"
  local profile_key="$2"
  local loader_output=""

  clear_qwen_matrix_state
  if ! loader_output="$(load_qwen_matrix_entry "${model_key}" "${profile_key}")"; then
    clear_qwen_matrix_state
    printf 'Failed to load SWE matrix entry for model=%s profile=%s\n' "${model_key}" "${profile_key}" >&2
    return 1
  fi
  if [[ -z "${loader_output}" ]]; then
    clear_qwen_matrix_state
    printf 'Empty SWE matrix entry for model=%s profile=%s\n' "${model_key}" "${profile_key}" >&2
    return 1
  fi
  if ! eval "${loader_output}"; then
    clear_qwen_matrix_state
    printf 'Failed to evaluate SWE matrix entry for model=%s profile=%s\n' "${model_key}" "${profile_key}" >&2
    return 1
  fi
}

materialize_long_context_views() {
  local target_view_name="$1"
  local target_source="$2"
  local draft_view_name="$3"
  local draft_source="$4"
  local max_position_embeddings="$5"
  local rope_factor="$6"
  local command=(
    python3
    "${SCRIPT_DIR}/materialize_long_context_model_views.py"
    --view-root
    "${LONG_CONTEXT_VIEW_ROOT}"
    --max-position-embeddings
    "${max_position_embeddings}"
    --rope-factor
    "${rope_factor}"
    --model-view
    "${target_view_name}=${target_source}"
    --model-view
    "${draft_view_name}=${draft_source}"
  )
  if [[ "${DRY_RUN}" == "true" || "${TEST_ONLY}" == "true" ]]; then
    if [[ "${DRY_RUN}" == "true" ]]; then
      printf "[DRY-RUN] "
    else
      printf "[TEST-ONLY] "
    fi
    render_command "${command[@]}"
    return
  fi
  "${command[@]}"
}

load_qwen_matrix_entry() {
  local model_key="$1"
  local profile_key="$2"
  python3 - "$MATRIX_FILE" "$SCRIPT_DIR" "$model_key" "$profile_key" <<'PY'
import json
import shlex
import sys
from pathlib import Path

matrix_path = Path(sys.argv[1])
script_dir = sys.argv[2]
model_key = sys.argv[3]
profile_key = sys.argv[4]
matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
model = next(
    item
    for item in matrix["models"]
    if item["launcher"] == "swe_sync_rollout" and item["key"] == model_key
)
profile = next(item for item in model["profiles"] if item["key"] == profile_key)


def emit(name: str, value: object) -> None:
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif value is None:
        text = ""
    else:
        text = str(value)
    print(f"{name}={shlex.quote(text)}")


emit("model_source", model["target_checkpoint"])
emit("draft_model_source", model["methods"]["eagle3"].get("draft_checkpoint", ""))
emit("target_tp", model["topology"]["target_tp"])
emit("benchmark_nodes", model["topology"]["benchmark_nodes"])
emit(
    "distributed_backend",
    model["topology"].get("distributed_executor_backend", ""),
)
emit("gpu_memory_utilization", model["topology"]["gpu_memory_utilization"])
emit(
    "request_plan_host",
    profile["request_plan_host"].replace("__SCRIPT_DIR__", script_dir),
)
emit("request_plan_container", profile["request_plan_container"])
emit("profile_context_policy", profile["context_policy"])
emit("max_model_len", profile["max_model_len"])
emit("max_new_tokens", profile["max_new_tokens"])
emit("max_position_embeddings", profile.get("max_position_embeddings", ""))
emit("rope_factor", profile.get("rope_factor", ""))
emit("target_view_name", profile.get("target_view_name", ""))
emit("draft_view_name", profile.get("draft_view_name", ""))
for method in matrix["method_order"]:
    entry = model["methods"][method]
    emit(f"method_{method}_status", entry["status"])
    emit(f"method_{method}_reason_code", entry.get("reason_code", ""))
    emit(f"method_{method}_reason", entry.get("reason", ""))
    emit(f"method_{method}_variants", " ".join(entry.get("variants", [])))
PY
}

if [[ "${DRY_RUN}" != "true" && "${TEST_ONLY}" != "true" && "${REQUIRE_GIT_PULL}" == "true" ]]; then
  git -C "${SCRIPT_DIR}" pull --ff-only
fi

if [[ "${DRY_RUN}" == "true" || "${TEST_ONLY}" == "true" ]]; then
  TEMP_MANIFEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/vllm024-swe-matrix.XXXXXX")"
  MANIFEST_ROOT="${TEMP_MANIFEST_ROOT}"
  MANIFEST="${MANIFEST_ROOT}/jobs.tsv"
  trap cleanup_temp_manifest EXIT
else
  if ! mkdir -p "${MANIFEST_ROOT}" 2>/dev/null; then
    mkdir -p "${MANIFEST_ROOT}"
  fi
fi
printf 'status\tmodel_key\tprofile_key\tmethod\tvariant\ttemperature\trun_dir\treason_code\treason\n' >"${MANIFEST}"

for model_key in ${MODELS}; do
  for profile in ${REQUEST_PROFILES}; do
    load_qwen_matrix_state "${model_key}" "${profile}"

    supported_variants=()
    for method in pard pard2 dflash dflare mtp_static mtp_dynamic; do
      status_var="method_${method}_status"
      reason_code_var="method_${method}_reason_code"
      reason_var="method_${method}_reason"
      status="${!status_var}"
      reason_code="${!reason_code_var}"
      reason="${!reason_var}"
      case "${status}" in
        integration)
          for temperature in ${TEMPERATURES}; do
            record_manifest_row \
              "INTEGRATION" \
              "${model_key}" \
              "${profile}" \
              "${method}" \
              "-" \
              "${temperature}" \
              "-" \
              "${reason_code}" \
              "${reason}"
          done
          ;;
        unsupported)
          for temperature in ${TEMPERATURES}; do
            record_manifest_row \
              "UNSUPPORTED" \
              "${model_key}" \
              "${profile}" \
              "${method}" \
              "-" \
              "${temperature}" \
              "-" \
              "${reason_code}" \
              "${reason}"
          done
          ;;
      esac
    done

    if variant_requested baseline; then
      supported_variants+=(baseline)
    fi
    if [[ "${method_eagle3_status}" == "supported" ]]; then
      if variant_requested static; then
        supported_variants+=(static)
      fi
      if variant_requested dynamic; then
        supported_variants+=(dynamic)
      fi
    fi

    for temperature in ${TEMPERATURES}; do
      temperature_slug="$(printf '%s' "${temperature}" | tr '.' 'p')"
      base_run_root="${RESULT_ROOT}/${RUN_ID_BASE}/${model_key}/${profile}/t${temperature_slug}"
      for variant in "${supported_variants[@]}"; do
        method_key="baseline"
        if [[ "${variant}" == "static" || "${variant}" == "dynamic" ]]; then
          method_key="eagle3"
        fi
        record_manifest_row \
          "SUPPORTED" \
          "${model_key}" \
          "${profile}" \
          "${method_key}" \
          "${variant}" \
          "${temperature}" \
          "${base_run_root}/matrix/${variant}" \
          "-" \
          "-"
      done
    done

    if ((${#supported_variants[@]} == 0)); then
      continue
    fi

    model="${model_source}"
    draft_model="${draft_model_source}"
    if [[ "${profile}" == "64k" ]]; then
      materialize_long_context_views \
        "${target_view_name}" \
        "${model_source}" \
        "${draft_view_name}" \
        "${draft_model_source}" \
        "${max_position_embeddings}" \
        "${rope_factor}"
      model="${LONG_CONTEXT_VIEW_ROOT}/${target_view_name}"
      draft_model="${LONG_CONTEXT_VIEW_ROOT}/${draft_view_name}"
    fi

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
      echo "context_policy=${profile_context_policy}"
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
        VARIANTS="${supported_variants[*]}" \
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

echo "manifest=${MANIFEST}"
