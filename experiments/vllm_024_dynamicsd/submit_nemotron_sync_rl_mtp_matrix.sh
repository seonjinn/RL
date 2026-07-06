#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATRIX_FILE="${SCRIPT_DIR}/model_method_matrix.json"
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

MANIFEST="${RESULT_ROOT}/jobs.tsv"

variant_requested() {
  [[ " ${VARIANTS} " == *" $1 "* ]]
}

record_manifest_row() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" >>"${MANIFEST}"
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

if ! mkdir -p "${RESULT_ROOT}" 2>/dev/null; then
  if [[ "${DRY_RUN}" == "true" || "${TEST_ONLY}" == "true" ]]; then
    RESULT_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/vllm024-nemotron-matrix.XXXXXX")"
    MANIFEST="${RESULT_ROOT}/jobs.tsv"
  else
    mkdir -p "${RESULT_ROOT}"
  fi
fi
printf 'status\tmodel_key\tprofile_key\tmethod\tvariant\trun_dir\treason_code\treason\n' >"${MANIFEST}"

for model_key in ${MODELS}; do
  eval "$(load_nemotron_matrix_entry "${model_key}")"

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
  if variant_requested baseline; then
    supported_variants+=(baseline)
  fi
  if [[ "${method_mtp_static_status}" == "supported" ]] && variant_requested mtp_static; then
    supported_variants+=(mtp_static)
  fi
  if [[ "${method_mtp_dynamic_status}" == "supported" ]] && variant_requested mtp_dynamic; then
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
      "${RESULT_ROOT}/${variant}" \
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
    DYNAMIC_SCHEDULE="${dynamic_schedule}" \
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

echo "manifest=${MANIFEST}"
