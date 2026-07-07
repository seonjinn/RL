#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATRIX_FILE="${MATRIX_FILE:-${SCRIPT_DIR}/model_method_matrix.json}"
LUSTRE_ROOT="${LUSTRE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}"
HF_HOME="${HF_HOME:-${LUSTRE_ROOT}/hf_home}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${LUSTRE_ROOT}/containers/vllm-openai-v0.24.0-aarch64-ubuntu2404.sqsh}"
RAY_SITE="${RAY_SITE:-${LUSTRE_ROOT}/vllm024-dynamicsd/python-sites/ray-2.55.1-py312}"
MODELS="${MODELS:-ultra super}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_nemotron_sync_rl_mtp}"
RESULT_ROOT="${RESULT_ROOT:-${LUSTRE_ROOT}/vllm024-dynamicsd/nemotron-sync-rl/${RUN_ID}}"
VARIANTS="${VARIANTS:-baseline mtp_static}"
SMOKE="${SMOKE:-true}"
PROMPT_JSONL="${PROMPT_JSONL:-}"
CALIBRATION_ARTIFACT_ROOT="${CALIBRATION_ARTIFACT_ROOT:-}"
CALIBRATION_REQUEST_PLAN="${CALIBRATION_REQUEST_PLAN:-${SCRIPT_DIR}/profiles/swe_sync_32k.json}"
CALIBRATION_DATASET_CONFIG="${CALIBRATION_DATASET_CONFIG:-throughput_1k}"
RUNTIME_IMAGE_SHA256="${RUNTIME_IMAGE_SHA256:-}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
SEED="${SEED:-1234}"
DRY_RUN="${DRY_RUN:-false}"
TEST_ONLY="${TEST_ONLY:-false}"
REQUIRE_GIT_PULL="${REQUIRE_GIT_PULL:-true}"

MANIFEST="${RESULT_ROOT}/jobs.tsv"
TEMP_MANIFEST_ROOT=""
NUM_PROMPTS_OVERRIDE_SET=false
SAMPLES_PER_PROMPT_OVERRIDE_SET=false
ROLLOUT_BATCHES_OVERRIDE_SET=false
MAX_PROMPT_TOKENS_OVERRIDE_SET=false
MAX_NEW_TOKENS_OVERRIDE_SET=false
ENGINE_MAX_NUM_SEQS_OVERRIDE_SET=false
TIME_LIMIT_OVERRIDE_SET=false
MAX_MODEL_LEN_OVERRIDE_SET=false

if [[ ${NUM_PROMPTS+x} ]]; then
  NUM_PROMPTS_OVERRIDE_SET=true
  NUM_PROMPTS_OVERRIDE="${NUM_PROMPTS}"
fi
if [[ ${SAMPLES_PER_PROMPT+x} ]]; then
  SAMPLES_PER_PROMPT_OVERRIDE_SET=true
  SAMPLES_PER_PROMPT_OVERRIDE="${SAMPLES_PER_PROMPT}"
fi
if [[ ${ROLLOUT_BATCHES+x} ]]; then
  ROLLOUT_BATCHES_OVERRIDE_SET=true
  ROLLOUT_BATCHES_OVERRIDE="${ROLLOUT_BATCHES}"
fi
if [[ ${MAX_PROMPT_TOKENS+x} ]]; then
  MAX_PROMPT_TOKENS_OVERRIDE_SET=true
  MAX_PROMPT_TOKENS_OVERRIDE="${MAX_PROMPT_TOKENS}"
fi
if [[ ${MAX_NEW_TOKENS+x} ]]; then
  MAX_NEW_TOKENS_OVERRIDE_SET=true
  MAX_NEW_TOKENS_OVERRIDE="${MAX_NEW_TOKENS}"
fi
if [[ ${ENGINE_MAX_NUM_SEQS+x} ]]; then
  ENGINE_MAX_NUM_SEQS_OVERRIDE_SET=true
  ENGINE_MAX_NUM_SEQS_OVERRIDE="${ENGINE_MAX_NUM_SEQS}"
fi
if [[ ${TIME_LIMIT+x} ]]; then
  TIME_LIMIT_OVERRIDE_SET=true
  TIME_LIMIT_OVERRIDE="${TIME_LIMIT}"
fi
if [[ ${MAX_MODEL_LEN+x} ]]; then
  MAX_MODEL_LEN_OVERRIDE_SET=true
  MAX_MODEL_LEN_OVERRIDE="${MAX_MODEL_LEN}"
fi

variant_requested() {
  [[ " ${VARIANTS} " == *" $1 "* ]]
}

record_manifest_row() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" >>"${MANIFEST}"
}

cleanup_temp_manifest() {
  if [[ -n "${TEMP_MANIFEST_ROOT}" ]]; then
    if [[ -f "${MANIFEST}" ]]; then
      cat "${MANIFEST}"
    fi
    rm -rf "${TEMP_MANIFEST_ROOT}"
  fi
}

request_plan_hash() {
  python3 - "$1" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
buckets = [
    {
        "ignore_eos": bucket.get("ignore_eos", False),
        "max_tokens": bucket["max_tokens"],
        "min_tokens": bucket.get("min_tokens", bucket["max_tokens"]),
        "weight": bucket["weight"],
    }
    for bucket in payload["buckets"]
]
buckets.sort(
    key=lambda bucket: (
        bucket["max_tokens"],
        bucket["min_tokens"],
        bucket["weight"],
        bucket["ignore_eos"],
    )
)
canonical = {
    "buckets": buckets,
    "max_model_len": payload["max_model_len"],
    "name": payload["name"],
}
encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
print(hashlib.sha256(encoded).hexdigest())
PY
}

model_config_hash() {
  python3 - "$1" <<'PY'
import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "config.json"
if not path.is_file():
    raise SystemExit(f"missing model config for calibration gate: {path}")
print(hashlib.sha256(path.read_bytes()).hexdigest())
PY
}

runtime_digest_for_gate() {
  if [[ -n "${RUNTIME_IMAGE_SHA256}" ]]; then
    printf '%s\n' "${RUNTIME_IMAGE_SHA256}"
  elif [[ -s "${CONTAINER_IMAGE}.sha256" ]]; then
    awk '{print $1; exit}' "${CONTAINER_IMAGE}.sha256"
  elif [[ -s "${CONTAINER_IMAGE}" ]]; then
    sha256sum "${CONTAINER_IMAGE}" | awk '{print $1; exit}'
  else
    echo "runtime image SHA is required to validate a dynamic calibration artifact" >&2
    return 1
  fi
}

validated_dynamic_schedule() {
  local model_key="$1"
  local model="$2"
  local plan_hash="$3"
  if [[ -z "${CALIBRATION_ARTIFACT_ROOT}" ]]; then
    echo "CALIBRATION_ARTIFACT_ROOT is required for DynamicMTP calibration artifact validation" >&2
    return 1
  fi
  local artifact="${CALIBRATION_ARTIFACT_ROOT}/${model_key}/${profile_key}/mtp/schedule.json"
  if [[ ! -s "${artifact}" ]]; then
    echo "calibration artifact is required for ${model_key}/${profile_key}/mtp: ${artifact}" >&2
    return 1
  fi
  local model_hash runtime_hash
  model_hash="$(model_config_hash "${model}")"
  runtime_hash="$(runtime_digest_for_gate)"
  python3 "${SCRIPT_DIR}/summarize_speedbench_k_calibration.py" validate \
    --artifact "${artifact}" \
    --model "${model}" \
    --model-config-hash "${model_hash}" \
    --context-profile "${context_policy}" \
    --request-plan-hash "${plan_hash}" \
    --runtime-image-sha256 "${runtime_hash}" \
    --method mtp \
    --dataset-config "${CALIBRATION_DATASET_CONFIG}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --seed "${SEED}"
}

clear_nemotron_matrix_state() {
  unset \
    model_path \
    target_tp \
    nodes \
    segment \
    distributed_executor_backend \
    enable_expert_parallel \
    mamba_ssm_cache_dtype \
    enable_stochastic_rounding \
    mamba_philox_rounds \
    model_loader_threads \
    disable_fuse_allreduce_rms \
    gpu_memory_utilization \
    static_k \
    dynamic_schedule \
    profile_key \
    context_policy \
    smoke_num_prompts \
    smoke_samples_per_prompt \
    smoke_rollout_batches \
    smoke_max_prompt_tokens \
    smoke_max_new_tokens \
    smoke_engine_max_num_seqs \
    smoke_time_limit \
    full_num_prompts \
    full_samples_per_prompt \
    full_rollout_batches \
    full_max_prompt_tokens \
    full_max_new_tokens \
    full_engine_max_num_seqs \
    full_time_limit \
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

load_nemotron_matrix_state() {
  local model_key="$1"
  local loader_output=""

  clear_nemotron_matrix_state
  if ! loader_output="$(load_nemotron_matrix_entry "${model_key}")"; then
    clear_nemotron_matrix_state
    printf 'Failed to load Nemotron matrix entry for model=%s\n' "${model_key}" >&2
    return 1
  fi
  if [[ -z "${loader_output}" ]]; then
    clear_nemotron_matrix_state
    printf 'Empty Nemotron matrix entry for model=%s\n' "${model_key}" >&2
    return 1
  fi
  if ! eval "${loader_output}"; then
    clear_nemotron_matrix_state
    printf 'Failed to evaluate Nemotron matrix entry for model=%s\n' "${model_key}" >&2
    return 1
  fi
}

load_nemotron_matrix_entry() {
  local model_key="$1"
  python3 - "$MATRIX_FILE" "$model_key" <<'PY'
import json
import shlex
import sys
from pathlib import Path

matrix = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
model_key = sys.argv[2]
model = next(
    item
    for item in matrix["models"]
    if item["launcher"] == "nemotron_sync_rl_mtp" and item["key"] == model_key
)
profile = model["profiles"][0]


def emit(name: str, value: object) -> None:
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif value is None:
        text = ""
    else:
        text = str(value)
    print(f"{name}={shlex.quote(text)}")


emit("model_path", model["target_checkpoint"])
for key, value in model["topology"].items():
    emit(key, value)
emit("profile_key", profile["key"])
emit("context_policy", profile["context_policy"])
for scope in ("smoke", "full"):
    for key, value in profile[scope].items():
        emit(f"{scope}_{key}", value)
for method in matrix["method_order"]:
    entry = model["methods"][method]
    emit(f"method_{method}_status", entry["status"])
    emit(f"method_{method}_reason_code", entry.get("reason_code", ""))
    emit(f"method_{method}_reason", entry.get("reason", ""))
    emit(f"method_{method}_variants", " ".join(entry.get("variants", [])))
PY
}

if [[ "${DRY_RUN}" != "true" && "${REQUIRE_GIT_PULL}" == "true" ]]; then
  git -C "${SCRIPT_DIR}" pull --ff-only
fi

if [[ "${DRY_RUN}" == "true" || "${TEST_ONLY}" == "true" ]]; then
  TEMP_MANIFEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/vllm024-nemotron-matrix.XXXXXX")"
  MANIFEST="${TEMP_MANIFEST_ROOT}/jobs.tsv"
  trap cleanup_temp_manifest EXIT
else
  if ! mkdir -p "${RESULT_ROOT}" 2>/dev/null; then
    mkdir -p "${RESULT_ROOT}"
  fi
fi
printf 'status\tmodel_key\tprofile_key\tmethod\tvariant\trun_dir\treason_code\treason\n' >"${MANIFEST}"

for model_key in ${MODELS}; do
  load_nemotron_matrix_state "${model_key}"

  if [[ "${SMOKE}" == "true" ]]; then
    NUM_PROMPTS="${smoke_num_prompts}"
    SAMPLES_PER_PROMPT="${smoke_samples_per_prompt}"
    ROLLOUT_BATCHES="${smoke_rollout_batches}"
    MAX_PROMPT_TOKENS="${smoke_max_prompt_tokens}"
    MAX_NEW_TOKENS="${smoke_max_new_tokens}"
    ENGINE_MAX_NUM_SEQS="${smoke_engine_max_num_seqs}"
    TIME_LIMIT="${smoke_time_limit}"
  else
    NUM_PROMPTS="${full_num_prompts}"
    SAMPLES_PER_PROMPT="${full_samples_per_prompt}"
    ROLLOUT_BATCHES="${full_rollout_batches}"
    MAX_PROMPT_TOKENS="${full_max_prompt_tokens}"
    MAX_NEW_TOKENS="${full_max_new_tokens}"
    ENGINE_MAX_NUM_SEQS="${full_engine_max_num_seqs}"
    TIME_LIMIT="${full_time_limit}"
    if [[ -z "${PROMPT_JSONL}" ]]; then
      echo "SMOKE=false requires PROMPT_JSONL from a pinned RL math dataset" >&2
      exit 2
    fi
  fi
  if [[ "${NUM_PROMPTS_OVERRIDE_SET}" == "true" ]]; then
    NUM_PROMPTS="${NUM_PROMPTS_OVERRIDE}"
  fi
  if [[ "${SAMPLES_PER_PROMPT_OVERRIDE_SET}" == "true" ]]; then
    SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT_OVERRIDE}"
  fi
  if [[ "${ROLLOUT_BATCHES_OVERRIDE_SET}" == "true" ]]; then
    ROLLOUT_BATCHES="${ROLLOUT_BATCHES_OVERRIDE}"
  fi
  if [[ "${MAX_PROMPT_TOKENS_OVERRIDE_SET}" == "true" ]]; then
    MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS_OVERRIDE}"
  fi
  if [[ "${MAX_NEW_TOKENS_OVERRIDE_SET}" == "true" ]]; then
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS_OVERRIDE}"
  fi
  if [[ "${ENGINE_MAX_NUM_SEQS_OVERRIDE_SET}" == "true" ]]; then
    ENGINE_MAX_NUM_SEQS="${ENGINE_MAX_NUM_SEQS_OVERRIDE}"
  fi
  if [[ "${TIME_LIMIT_OVERRIDE_SET}" == "true" ]]; then
    TIME_LIMIT="${TIME_LIMIT_OVERRIDE}"
  fi
  MAX_MODEL_LEN="$((MAX_PROMPT_TOKENS + MAX_NEW_TOKENS + 256))"
  if [[ "${MAX_MODEL_LEN_OVERRIDE_SET}" == "true" ]]; then
    MAX_MODEL_LEN="${MAX_MODEL_LEN_OVERRIDE}"
  fi

  for method in eagle3 pard pard2 dflash dflare; do
    reason_code_var="method_${method}_reason_code"
    reason_var="method_${method}_reason"
    record_manifest_row \
      "UNSUPPORTED" \
      "${model_key}" \
      "${profile_key}" \
      "${method}" \
      "-" \
      "-" \
      "${!reason_code_var}" \
      "${!reason_var}"
  done

  supported_variants=()
  dynamic_requested=false
  if variant_requested baseline; then
    supported_variants+=(baseline)
  fi
  if [[ "${method_mtp_static_status}" == "supported" ]] && variant_requested mtp_static; then
    supported_variants+=(mtp_static)
  fi
  if [[ "${method_mtp_dynamic_status}" == "supported" ]] && variant_requested mtp_dynamic; then
    dynamic_requested=true
  fi

  effective_dynamic_schedule="${dynamic_schedule}"
  if [[ "${dynamic_requested}" == "true" ]]; then
    calibration_plan_hash="$(request_plan_hash "${CALIBRATION_REQUEST_PLAN}")"
    effective_dynamic_schedule="$(
      validated_dynamic_schedule \
        "${model_key}" \
        "${model_path}" \
        "${calibration_plan_hash}"
    )"
    supported_variants+=(mtp_dynamic)
  fi

  for variant in "${supported_variants[@]}"; do
    method_key="${variant}"
    record_manifest_row \
      "SUPPORTED" \
      "${model_key}" \
      "${profile_key}" \
      "${method_key}" \
      "${variant}" \
      "${RESULT_ROOT}/${model_key}/${variant}" \
      "-" \
      "-"
  done

  if ((${#supported_variants[@]} == 0)); then
    continue
  fi

  env \
    CLUSTER="${CLUSTER:-auto}" \
    ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}" \
    PARTITION="${PARTITION:-batch}" \
    LUSTRE_ROOT="${LUSTRE_ROOT}" \
    HF_HOME="${HF_HOME}" \
    MODEL="${model_path}" \
    DRAFT_MODEL="" \
    CONTAINER_IMAGE="${CONTAINER_IMAGE}" \
    RAY_SITE="${RAY_SITE}" \
    RESULT_ROOT="${RESULT_ROOT}" \
    RUN_ID="${model_key}" \
    JOB_LABEL="sync-${model_key}-bf16" \
    VARIANTS="${supported_variants[*]}" \
    STATIC_K="${static_k}" \
    DYNAMIC_SCHEDULE="${effective_dynamic_schedule}" \
    TP="${target_tp}" \
    PP=1 \
    NODES="${nodes}" \
    SEGMENT="${segment}" \
    DISTRIBUTED_EXECUTOR_BACKEND="${distributed_executor_backend}" \
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
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-${gpu_memory_utilization}}" \
    CUDAGRAPH_MODE=PIECEWISE \
    ENGINE_MAX_NUM_SEQS="${ENGINE_MAX_NUM_SEQS}" \
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}" \
    TEMPERATURE="${TEMPERATURE}" \
    TOP_P="${TOP_P}" \
    SEED="${SEED}" \
    RUNTIME_IMAGE_SHA256="${RUNTIME_IMAGE_SHA256}" \
    NUM_PROMPTS="${NUM_PROMPTS}" \
    SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT}" \
    ROLLOUT_BATCHES="${ROLLOUT_BATCHES}" \
    MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS}" \
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
    MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    PROMPT_JSONL="${PROMPT_JSONL}" \
    SOURCE_RECIPE="sync-rl-math-rollout" \
    CONTEXT_PROFILE="${context_policy}" \
    SMOKE="${SMOKE}" \
    TIME_LIMIT="${TIME_LIMIT}" \
    DRY_RUN="${DRY_RUN}" \
    TEST_ONLY="${TEST_ONLY}" \
    REQUIRE_GIT_PULL=false \
    "${SCRIPT_DIR}/submit_sync_rollout.sh"
done

echo "manifest=${MANIFEST}"
