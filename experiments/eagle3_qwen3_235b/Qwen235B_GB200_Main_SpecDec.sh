#!/bin/bash
set -euo pipefail

SCRIPT_PATH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${NEMO_RL_DIR:-}" ]]; then
  SCRIPT_DIR="${NEMO_RL_DIR}"
elif [[ -f "${SCRIPT_PATH_DIR}/examples/run_grpo.py" ]]; then
  SCRIPT_DIR="${SCRIPT_PATH_DIR}"
else
  SCRIPT_DIR="$(cd "${SCRIPT_PATH_DIR}/../.." && pwd)"
fi
if [[ ! -f "${SCRIPT_DIR}/examples/run_grpo.py" ]]; then
  echo "ERROR: could not locate NeMo-RL repo root from ${SCRIPT_PATH_DIR}; set NEMO_RL_DIR" >&2
  exit 2
fi
if [[ "$(cd "${SCRIPT_DIR}" && pwd -P)" == *"/remote_worktree_edit"* ]]; then
  echo "ERROR: refusing to launch from stale scratch tree: ${SCRIPT_DIR}" >&2
  echo "Use the maintained SpecDec-RL checkout/overlay instead." >&2
  exit 2
fi

NUM_NODES="${NUM_NODES:-32}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
PARTITION="${PARTITION:-batch}"
ACCOUNT="${ACCOUNT:-coreai_dlalgo_nemorl}"
GRES_FLAG="${GRES_FLAG:---gres=gpu:4}"
SEGMENT="${SEGMENT:-16}"
CPUS_PER_WORKER="${CPUS_PER_WORKER:-$((GPUS_PER_NODE * 16))}"
SBATCH_RESOURCE_ARGS="${SBATCH_RESOURCE_ARGS:---ntasks-per-node=1 --cpus-per-task=${CPUS_PER_WORKER} --mem=0}"
SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:-}"
JOB_TAG="${JOB_TAG:-main-specdec}"

CONFIG_FILE="${CONFIG_FILE:-examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n8g.yaml}"
TARGET_MODEL_ID="${TARGET_MODEL_ID:-Qwen/Qwen3-235B-A22B-Thinking-2507}"
CONTAINER="${CONTAINER:-${SCRIPT_DIR}/nemo_rl_nightly.sqsh}"
HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/cache}"
MOUNTS="${MOUNTS:-/lustre:/lustre}"
NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-${SCRIPT_DIR}/nrl_megatron_ckpts_20260526}"
WANDB_PROJECT="${WANDB_PROJECT:-sync-grpo-gb200_oci-benchmark}"
UV_PYTHON="${UV_PYTHON:-3.12.13}"
RAY_VERSION="${RAY_VERSION:-2.49.2}"
DRIVER_UV_PROJECT_ENVIRONMENT="${DRIVER_UV_PROJECT_ENVIRONMENT:-${SCRIPT_DIR}/.driver_venvs/qwen235b_main_specdec_py312}"
RAY_CGRAPH_GET_TIMEOUT="${RAY_CGRAPH_GET_TIMEOUT:-3600}"
NUM_PROMPTS="${NUM_PROMPTS:-16}"
NUM_GENERATIONS="${NUM_GENERATIONS:-32}"
TRAIN_GLOBAL_BATCH_SIZE="${TRAIN_GLOBAL_BATCH_SIZE:-$((NUM_PROMPTS * NUM_GENERATIONS))}"
MAX_STEPS="${MAX_STEPS:-20}"
NRL_MEGATRON_NCCL_TIMEOUT_SECONDS="${NRL_MEGATRON_NCCL_TIMEOUT_SECONDS:-1800}"

DEFAULT_LEGACY_DRAFT_MODEL="/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/speculators/eagle3_openmath50k_dapo16k_continued/checkpoints_from50k_dapo16k_e3_lr5e5_layers93_mlen8193_cachefix_r2/2"
EXPECTED_500K_DRAFT_ROOT="/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/speculators/eagle3_qwen3_235b_mixed_math_nonopenmath_500k_parallel/checkpoints_train_500k_layers94_mlen8193"
DRAFT_MODEL="${DRAFT_MODEL:-${DEFAULT_LEGACY_DRAFT_MODEL}}"
DRAFT_MODEL_PROVENANCE="${DRAFT_MODEL_PROVENANCE:-}"
REQUIRE_DRAFT_MODEL_PROVENANCE="${REQUIRE_DRAFT_MODEL_PROVENANCE:-true}"
SPECDEC_METHOD="${SPECDEC_METHOD:-eagle3}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
DRAFT_TP="${DRAFT_TP:-1}"
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
SPECDEC_DISABLE_BY_BATCH_SIZE="${SPECDEC_DISABLE_BY_BATCH_SIZE:-}"
SPECDEC_SCHEDULER_PATCH_GATE_THRESHOLD="${SPECDEC_SCHEDULER_PATCH_GATE_THRESHOLD:-0}"
SPECDEC_SCHEDULER_PATCH_GATE_TOKEN_THRESHOLD="${SPECDEC_SCHEDULER_PATCH_GATE_TOKEN_THRESHOLD:-4096}"
SPECDEC_ADAPTIVE_GATE_MODE="${SPECDEC_ADAPTIVE_GATE_MODE:-${VLLM_SPECDEC_ADAPTIVE_GATE_MODE:-${VLLM_SPECDEC_BATCH_GATE_ADAPTIVE_MODE:-}}}"
SPECDEC_ADAPTIVE_GATE_TARGET="${SPECDEC_ADAPTIVE_GATE_TARGET:-${VLLM_SPECDEC_ADAPTIVE_TARGET_ENABLED_RATIO:-${VLLM_SPECDEC_BATCH_GATE_ADAPTIVE_TARGET:-}}}"
SPECDEC_ADAPTIVE_GATE_HYSTERESIS="${SPECDEC_ADAPTIVE_GATE_HYSTERESIS:-${VLLM_SPECDEC_ADAPTIVE_HYSTERESIS:-}}"
SPECDEC_ADAPTIVE_GATE_ADJUST_INTERVAL="${SPECDEC_ADAPTIVE_GATE_ADJUST_INTERVAL:-${VLLM_SPECDEC_ADAPTIVE_ADJUST_INTERVAL:-}}"
SPECDEC_ADAPTIVE_GATE_INITIAL_REQUEST_THRESHOLD="${SPECDEC_ADAPTIVE_GATE_INITIAL_REQUEST_THRESHOLD:-${VLLM_SPECDEC_ADAPTIVE_INITIAL_REQUEST_THRESHOLD:-}}"
SPECDEC_ADAPTIVE_GATE_INITIAL_TOKEN_THRESHOLD="${SPECDEC_ADAPTIVE_GATE_INITIAL_TOKEN_THRESHOLD:-${VLLM_SPECDEC_ADAPTIVE_INITIAL_TOKEN_THRESHOLD:-}}"
SPECDEC_ADAPTIVE_GATE_MIN_REQUEST_THRESHOLD="${SPECDEC_ADAPTIVE_GATE_MIN_REQUEST_THRESHOLD:-${VLLM_SPECDEC_ADAPTIVE_MIN_REQUEST_THRESHOLD:-${VLLM_SPECDEC_BATCH_GATE_MIN_REQUEST_THRESHOLD:-}}}"
SPECDEC_ADAPTIVE_GATE_MAX_REQUEST_THRESHOLD="${SPECDEC_ADAPTIVE_GATE_MAX_REQUEST_THRESHOLD:-${VLLM_SPECDEC_ADAPTIVE_MAX_REQUEST_THRESHOLD:-${VLLM_SPECDEC_BATCH_GATE_MAX_REQUEST_THRESHOLD:-}}}"
SPECDEC_ADAPTIVE_GATE_REQUEST_STEP="${SPECDEC_ADAPTIVE_GATE_REQUEST_STEP:-${VLLM_SPECDEC_ADAPTIVE_REQUEST_STEP:-}}"
SPECDEC_ADAPTIVE_GATE_MIN_TOKEN_THRESHOLD="${SPECDEC_ADAPTIVE_GATE_MIN_TOKEN_THRESHOLD:-${VLLM_SPECDEC_ADAPTIVE_MIN_TOKEN_THRESHOLD:-${VLLM_SPECDEC_BATCH_GATE_MIN_TOKEN_THRESHOLD:-}}}"
SPECDEC_ADAPTIVE_GATE_MAX_TOKEN_THRESHOLD="${SPECDEC_ADAPTIVE_GATE_MAX_TOKEN_THRESHOLD:-${VLLM_SPECDEC_ADAPTIVE_MAX_TOKEN_THRESHOLD:-${VLLM_SPECDEC_BATCH_GATE_MAX_TOKEN_THRESHOLD:-}}}"
SPECDEC_ADAPTIVE_GATE_TOKEN_STEP="${SPECDEC_ADAPTIVE_GATE_TOKEN_STEP:-${VLLM_SPECDEC_ADAPTIVE_TOKEN_STEP:-}}"
SPECDEC_DYNAMIC_DRAFT_TOKENS="${SPECDEC_DYNAMIC_DRAFT_TOKENS:-${VLLM_SPECDEC_DYNAMIC_DRAFT_TOKENS:-}}"
SPECDEC_DYNAMIC_DRAFT_SMALL_REQUEST_THRESHOLD="${SPECDEC_DYNAMIC_DRAFT_SMALL_REQUEST_THRESHOLD:-${VLLM_SPECDEC_DYNAMIC_DRAFT_SMALL_REQUEST_THRESHOLD:-}}"
SPECDEC_DYNAMIC_DRAFT_MEDIUM_REQUEST_THRESHOLD="${SPECDEC_DYNAMIC_DRAFT_MEDIUM_REQUEST_THRESHOLD:-${VLLM_SPECDEC_DYNAMIC_DRAFT_MEDIUM_REQUEST_THRESHOLD:-}}"
SPECDEC_DYNAMIC_DRAFT_SMALL_TOKEN_THRESHOLD="${SPECDEC_DYNAMIC_DRAFT_SMALL_TOKEN_THRESHOLD:-${VLLM_SPECDEC_DYNAMIC_DRAFT_SMALL_TOKEN_THRESHOLD:-}}"
SPECDEC_DYNAMIC_DRAFT_MEDIUM_TOKEN_THRESHOLD="${SPECDEC_DYNAMIC_DRAFT_MEDIUM_TOKEN_THRESHOLD:-${VLLM_SPECDEC_DYNAMIC_DRAFT_MEDIUM_TOKEN_THRESHOLD:-}}"
SPECDEC_DYNAMIC_DRAFT_SMALL_TOKENS="${SPECDEC_DYNAMIC_DRAFT_SMALL_TOKENS:-${VLLM_SPECDEC_DYNAMIC_DRAFT_SMALL_TOKENS:-}}"
SPECDEC_DYNAMIC_DRAFT_MEDIUM_TOKENS="${SPECDEC_DYNAMIC_DRAFT_MEDIUM_TOKENS:-${VLLM_SPECDEC_DYNAMIC_DRAFT_MEDIUM_TOKENS:-}}"
SPECDEC_DYNAMIC_DRAFT_LARGE_TOKENS="${SPECDEC_DYNAMIC_DRAFT_LARGE_TOKENS:-${VLLM_SPECDEC_DYNAMIC_DRAFT_LARGE_TOKENS:-}}"
ENABLE_RUNTIME_SPECDEC_GATE_PATCH="${ENABLE_RUNTIME_SPECDEC_GATE_PATCH:-true}"
VLLM_SPECDEC_BATCH_GATE_LOG_INTERVAL="${VLLM_SPECDEC_BATCH_GATE_LOG_INTERVAL:-256}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-}"
NRL_VLLM_DISABLE_LOG_STATS="${NRL_VLLM_DISABLE_LOG_STATS:-false}"
NRL_VLLM_OMIT_GENERATION_LOGPROBS="${NRL_VLLM_OMIT_GENERATION_LOGPROBS:-true}"
NRL_STOP_AFTER_GENERATION="${NRL_STOP_AFTER_GENERATION:-false}"
NRL_SPECDEC_STEP_ADAPTIVE_CONTROLLER="${NRL_SPECDEC_STEP_ADAPTIVE_CONTROLLER:-false}"
NRL_SPECDEC_CONTROLLER_MIN_DRAFT_TOKENS="${NRL_SPECDEC_CONTROLLER_MIN_DRAFT_TOKENS:-128}"
NRL_SPECDEC_CONTROLLER_LOW_ACCEPTANCE="${NRL_SPECDEC_CONTROLLER_LOW_ACCEPTANCE:-0.45}"
NRL_SPECDEC_CONTROLLER_HIGH_ACCEPTANCE="${NRL_SPECDEC_CONTROLLER_HIGH_ACCEPTANCE:-0.62}"
NRL_SPECDEC_CONTROLLER_POS2_FLOOR="${NRL_SPECDEC_CONTROLLER_POS2_FLOOR:-0.25}"
NRL_SPECDEC_CONTROLLER_POS3_FLOOR="${NRL_SPECDEC_CONTROLLER_POS3_FLOOR:-0.15}"
NRL_SPECDEC_CONTROLLER_MIN_K="${NRL_SPECDEC_CONTROLLER_MIN_K:-1}"
NRL_SPECDEC_CONTROLLER_MAX_K="${NRL_SPECDEC_CONTROLLER_MAX_K:-${NUM_SPECULATIVE_TOKENS}}"
NRL_SPECDEC_CONTROLLER_ALLOW_INCREASE="${NRL_SPECDEC_CONTROLLER_ALLOW_INCREASE:-false}"
REQUIRE_SPECDEC_RL_PATCHES="${REQUIRE_SPECDEC_RL_PATCHES:-true}"
WANDB_NAME="${WANDB_NAME:-Qwen235B_A22B_Main_N${NUM_NODES}xG${GPUS_PER_NODE}_specdec_k${NUM_SPECULATIVE_TOKENS}_p${NUM_PROMPTS}_g${NUM_GENERATIONS}_${MAX_STEPS}step}"

if [[ -z "${SPECDEC_ADAPTIVE_GATE_MODE}" ]] && {
  [[ -n "${SPECDEC_ADAPTIVE_GATE_TARGET}" ]] ||
  [[ -n "${SPECDEC_ADAPTIVE_GATE_HYSTERESIS}" ]] ||
  [[ -n "${SPECDEC_ADAPTIVE_GATE_ADJUST_INTERVAL}" ]] ||
  [[ -n "${SPECDEC_ADAPTIVE_GATE_INITIAL_REQUEST_THRESHOLD}" ]] ||
  [[ -n "${SPECDEC_ADAPTIVE_GATE_INITIAL_TOKEN_THRESHOLD}" ]] ||
  [[ -n "${SPECDEC_ADAPTIVE_GATE_MIN_REQUEST_THRESHOLD}" ]] ||
  [[ -n "${SPECDEC_ADAPTIVE_GATE_MAX_REQUEST_THRESHOLD}" ]] ||
  [[ -n "${SPECDEC_ADAPTIVE_GATE_REQUEST_STEP}" ]] ||
  [[ -n "${SPECDEC_ADAPTIVE_GATE_MIN_TOKEN_THRESHOLD}" ]] ||
  [[ -n "${SPECDEC_ADAPTIVE_GATE_MAX_TOKEN_THRESHOLD}" ]] ||
  [[ -n "${SPECDEC_ADAPTIVE_GATE_TOKEN_STEP}" ]]
}; then
  SPECDEC_ADAPTIVE_GATE_MODE="enabled_ratio"
fi

if {
  [[ -n "${SPECDEC_SCHEDULER_PATCH_GATE_THRESHOLD}" && "${SPECDEC_SCHEDULER_PATCH_GATE_THRESHOLD}" != "0" ]] ||
  [[ -n "${SPECDEC_SCHEDULER_PATCH_GATE_TOKEN_THRESHOLD}" && "${SPECDEC_SCHEDULER_PATCH_GATE_TOKEN_THRESHOLD}" != "0" ]] ||
  [[ -n "${SPECDEC_ADAPTIVE_GATE_MODE}" ]] ||
  [[ -n "${SPECDEC_ADAPTIVE_GATE_TARGET}" ]] ||
  [[ -n "${SPECDEC_ADAPTIVE_GATE_HYSTERESIS}" ]] ||
  [[ -n "${SPECDEC_ADAPTIVE_GATE_ADJUST_INTERVAL}" ]] ||
  [[ -n "${SPECDEC_ADAPTIVE_GATE_INITIAL_REQUEST_THRESHOLD}" ]] ||
  [[ -n "${SPECDEC_ADAPTIVE_GATE_INITIAL_TOKEN_THRESHOLD}" ]] ||
  [[ -n "${SPECDEC_ADAPTIVE_GATE_MIN_REQUEST_THRESHOLD}" ]] ||
  [[ -n "${SPECDEC_ADAPTIVE_GATE_MAX_REQUEST_THRESHOLD}" ]] ||
  [[ -n "${SPECDEC_ADAPTIVE_GATE_REQUEST_STEP}" ]] ||
  [[ -n "${SPECDEC_ADAPTIVE_GATE_MIN_TOKEN_THRESHOLD}" ]] ||
  [[ -n "${SPECDEC_ADAPTIVE_GATE_MAX_TOKEN_THRESHOLD}" ]] ||
  [[ -n "${SPECDEC_ADAPTIVE_GATE_TOKEN_STEP}" ]] ||
  [[ -n "${SPECDEC_DYNAMIC_DRAFT_TOKENS}" ]] ||
  [[ -n "${SPECDEC_DYNAMIC_DRAFT_SMALL_REQUEST_THRESHOLD}" ]] ||
  [[ -n "${SPECDEC_DYNAMIC_DRAFT_MEDIUM_REQUEST_THRESHOLD}" ]] ||
  [[ -n "${SPECDEC_DYNAMIC_DRAFT_SMALL_TOKEN_THRESHOLD}" ]] ||
  [[ -n "${SPECDEC_DYNAMIC_DRAFT_MEDIUM_TOKEN_THRESHOLD}" ]] ||
  [[ -n "${SPECDEC_DYNAMIC_DRAFT_SMALL_TOKENS}" ]] ||
  [[ -n "${SPECDEC_DYNAMIC_DRAFT_MEDIUM_TOKENS}" ]] ||
  [[ -n "${SPECDEC_DYNAMIC_DRAFT_LARGE_TOKENS}" ]]
}; then
  case "${ENABLE_RUNTIME_SPECDEC_GATE_PATCH}" in
    1|true|TRUE|yes|YES|y|Y|on|ON) ;;
    *)
      echo "ERROR: SpecDec scheduler/adaptive gate settings require ENABLE_RUNTIME_SPECDEC_GATE_PATCH=true" >&2
      exit 2
      ;;
  esac
fi
if {
  [[ -n "${SPECDEC_DYNAMIC_DRAFT_SMALL_REQUEST_THRESHOLD}" ]] ||
  [[ -n "${SPECDEC_DYNAMIC_DRAFT_MEDIUM_REQUEST_THRESHOLD}" ]] ||
  [[ -n "${SPECDEC_DYNAMIC_DRAFT_SMALL_TOKEN_THRESHOLD}" ]] ||
  [[ -n "${SPECDEC_DYNAMIC_DRAFT_MEDIUM_TOKEN_THRESHOLD}" ]] ||
  [[ -n "${SPECDEC_DYNAMIC_DRAFT_SMALL_TOKENS}" ]] ||
  [[ -n "${SPECDEC_DYNAMIC_DRAFT_MEDIUM_TOKENS}" ]] ||
  [[ -n "${SPECDEC_DYNAMIC_DRAFT_LARGE_TOKENS}" ]]
}; then
  case "${SPECDEC_DYNAMIC_DRAFT_TOKENS}" in
    1|true|TRUE|yes|YES|y|Y|on|ON) ;;
    *)
      echo "ERROR: dynamic SpecDec tier settings require SPECDEC_DYNAMIC_DRAFT_TOKENS=true" >&2
      exit 2
      ;;
  esac
fi
if [[ -n "${SPECDEC_DISABLE_BY_BATCH_SIZE}" ]]; then
  echo "ERROR: SPECDEC_DISABLE_BY_BATCH_SIZE is not the NeMo-RL long-tail gate. Use SPECDEC_SCHEDULER_PATCH_GATE_THRESHOLD or SPECDEC_SCHEDULER_PATCH_GATE_TOKEN_THRESHOLD with ENABLE_RUNTIME_SPECDEC_GATE_PATCH=true." >&2
  exit 2
fi

mkdir -p "${NRL_MEGATRON_CHECKPOINT_DIR}"
checkpoint_iter_dir="${NRL_MEGATRON_CHECKPOINT_DIR}/${TARGET_MODEL_ID}/iter_0000000"
checkpoint_run_config="${checkpoint_iter_dir}/run_config.yaml"
if [[ -d "${checkpoint_iter_dir}" && ! -s "${checkpoint_run_config}" ]]; then
  deadline=$((SECONDS + ${NRL_MEGATRON_CHECKPOINT_READY_TIMEOUT_SEC:-1800}))
  echo "Waiting for existing Megatron checkpoint conversion to finish: ${checkpoint_run_config}"
  while [[ ! -s "${checkpoint_run_config}" && "${SECONDS}" -lt "${deadline}" ]]; do
    sleep 10
  done
  if [[ ! -s "${checkpoint_run_config}" ]]; then
    echo "ERROR: Megatron checkpoint directory exists but run_config.yaml is still missing: ${checkpoint_run_config}" >&2
    echo "A concurrent HF->mcore conversion may have failed or still be incomplete." >&2
    exit 2
  fi
fi

if [[ ! -s "${CONTAINER}" ]]; then
  echo "ERROR: container not found: ${CONTAINER}" >&2
  exit 2
fi
if [[ ! -s "${SCRIPT_DIR}/ray.sub" ]]; then
  echo "ERROR: patched ray.sub not found at ${SCRIPT_DIR}/ray.sub" >&2
  exit 2
fi
if [[ "${REQUIRE_SPECDEC_RL_PATCHES}" == "true" || "${REQUIRE_SPECDEC_RL_PATCHES}" == "True" ]]; then
  require_patch_marker() {
    local file="$1"
    local marker="$2"
    if [[ ! -s "${file}" ]] || ! grep -q "${marker}" "${file}"; then
      echo "ERROR: required SpecDec-RL patch marker '${marker}' is missing from ${file}" >&2
      echo "Use NEMO_RL_DIR=/lustre/fs1/.../SpecDec-RL with the current patch bundle, or set REQUIRE_SPECDEC_RL_PATCHES=false only for diagnostics." >&2
      exit 2
    fi
  }
  require_patch_marker "${SCRIPT_DIR}/nemo_rl/models/generation/vllm/vllm_worker.py" "NRL_SPECDEC_BATCH_GATE_PATCH_V8"
  require_patch_marker "${SCRIPT_DIR}/nemo_rl/models/generation/vllm/vllm_generation.py" "acceptance_rate_reliable"
  require_patch_marker "${SCRIPT_DIR}/nemo_rl/algorithms/grpo.py" "_repair_specdec_generation_logprobs_if_safe"
  require_patch_marker "${SCRIPT_DIR}/nemo_rl/models/generation/vllm/vllm_worker.py" "NRL_VLLM_OMIT_GENERATION_LOGPROBS"
fi
if [[ ! -s "${DRAFT_MODEL}/config.json" ]]; then
  echo "ERROR: DRAFT_MODEL is not a valid HF checkpoint: ${DRAFT_MODEL}" >&2
  exit 2
fi

path_is_within() {
  canonical_path_if_exists() {
    local path="${1%/}"
    if [[ -e "${path}" ]]; then
      (cd "${path}" && pwd -P)
    elif [[ -e "$(dirname "${path}")" ]]; then
      printf "%s/%s\n" "$(cd "$(dirname "${path}")" && pwd -P)" "$(basename "${path}")"
    else
      printf "%s\n" "${path}"
    fi
  }
  local child
  local parent
  child="$(canonical_path_if_exists "$1")"
  parent="$(canonical_path_if_exists "$2")"
  [[ "${child}" == "${parent}" || "${child}" == "${parent}/"* ]]
}

if [[ "${DRAFT_MODEL}" == "${DEFAULT_LEGACY_DRAFT_MODEL}" && "${ALLOW_LEGACY_DRAFT_MODEL:-false}" != "true" && "${ALLOW_LEGACY_DRAFT_MODEL:-false}" != "True" ]]; then
  echo "ERROR: refusing to use legacy default DRAFT_MODEL=${DRAFT_MODEL}" >&2
  echo "Set DRAFT_MODEL explicitly for the in-house 500K checkpoint after training, or ALLOW_LEGACY_DRAFT_MODEL=true for an explicit legacy diagnostic run." >&2
  exit 2
fi
if [[ "${REQUIRE_DRAFT_MODEL_PROVENANCE}" == "true" || "${REQUIRE_DRAFT_MODEL_PROVENANCE}" == "True" ]]; then
  case "${DRAFT_MODEL_PROVENANCE}" in
    qwen235b_mixed_math_nonopenmath_500k_speculators)
      if ! path_is_within "${DRAFT_MODEL}" "${EXPECTED_500K_DRAFT_ROOT}"; then
        echo "ERROR: DRAFT_MODEL_PROVENANCE=${DRAFT_MODEL_PROVENANCE} but DRAFT_MODEL is outside the expected 235B 500K root." >&2
        echo "DRAFT_MODEL=${DRAFT_MODEL}" >&2
        echo "Expected prefix: ${EXPECTED_500K_DRAFT_ROOT}" >&2
        exit 2
      fi
      ;;
    legacy_diagnostic)
      if [[ "${ALLOW_LEGACY_DRAFT_MODEL:-false}" != "true" && "${ALLOW_LEGACY_DRAFT_MODEL:-false}" != "True" ]]; then
        echo "ERROR: legacy_diagnostic provenance requires ALLOW_LEGACY_DRAFT_MODEL=true." >&2
        exit 2
      fi
      ;;
    manual_diagnostic)
      if [[ "${ALLOW_NON_STANDARD_DRAFT_MODEL:-false}" != "true" && "${ALLOW_NON_STANDARD_DRAFT_MODEL:-false}" != "True" ]]; then
        echo "ERROR: manual_diagnostic provenance requires ALLOW_NON_STANDARD_DRAFT_MODEL=true." >&2
        exit 2
      fi
      ;;
    "")
      echo "ERROR: DRAFT_MODEL_PROVENANCE is required for 235B SpecDec main runs." >&2
      echo "Use qwen235b_mixed_math_nonopenmath_500k_speculators for the in-house 500K checkpoint, legacy_diagnostic for explicit legacy checks, or manual_diagnostic for non-result diagnostics." >&2
      exit 2
      ;;
    *)
      echo "ERROR: unknown DRAFT_MODEL_PROVENANCE=${DRAFT_MODEL_PROVENANCE}" >&2
      exit 2
      ;;
  esac
fi
echo "Using SpecDec drafter: ${DRAFT_MODEL}"
echo "Drafter provenance: ${DRAFT_MODEL_PROVENANCE:-unverified}"

SPECDEC_EXTRA_OVERRIDES="${SPECDEC_EXTRA_OVERRIDES:-${EXTRA_OVERRIDES:-}}"
if [[ -n "${VLLM_MAX_NUM_SEQS}" ]]; then
  SPECDEC_EXTRA_OVERRIDES+=" ++policy.generation.vllm_kwargs.max_num_seqs=${VLLM_MAX_NUM_SEQS}"
fi
if [[ -n "${VLLM_MAX_NUM_BATCHED_TOKENS}" ]]; then
  SPECDEC_EXTRA_OVERRIDES+=" ++policy.generation.vllm_kwargs.max_num_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS}"
fi

SPECDEC_ADAPTIVE_GATE_ENV=""
append_specdec_adaptive_gate_env() {
  local name="$1"
  local value="$2"
  if [[ -n "${value}" ]]; then
    printf -v value "%q" "${value}"
    SPECDEC_ADAPTIVE_GATE_ENV+=" ${name}=${value}"
  fi
}
append_specdec_adaptive_gate_env "VLLM_SPECDEC_ADAPTIVE_GATE_MODE" "${SPECDEC_ADAPTIVE_GATE_MODE}"
append_specdec_adaptive_gate_env "VLLM_SPECDEC_ADAPTIVE_TARGET_ENABLED_RATIO" "${SPECDEC_ADAPTIVE_GATE_TARGET}"
append_specdec_adaptive_gate_env "VLLM_SPECDEC_ADAPTIVE_HYSTERESIS" "${SPECDEC_ADAPTIVE_GATE_HYSTERESIS}"
append_specdec_adaptive_gate_env "VLLM_SPECDEC_ADAPTIVE_ADJUST_INTERVAL" "${SPECDEC_ADAPTIVE_GATE_ADJUST_INTERVAL}"
append_specdec_adaptive_gate_env "VLLM_SPECDEC_ADAPTIVE_INITIAL_REQUEST_THRESHOLD" "${SPECDEC_ADAPTIVE_GATE_INITIAL_REQUEST_THRESHOLD}"
append_specdec_adaptive_gate_env "VLLM_SPECDEC_ADAPTIVE_INITIAL_TOKEN_THRESHOLD" "${SPECDEC_ADAPTIVE_GATE_INITIAL_TOKEN_THRESHOLD}"
append_specdec_adaptive_gate_env "VLLM_SPECDEC_ADAPTIVE_MIN_REQUEST_THRESHOLD" "${SPECDEC_ADAPTIVE_GATE_MIN_REQUEST_THRESHOLD}"
append_specdec_adaptive_gate_env "VLLM_SPECDEC_ADAPTIVE_MAX_REQUEST_THRESHOLD" "${SPECDEC_ADAPTIVE_GATE_MAX_REQUEST_THRESHOLD}"
append_specdec_adaptive_gate_env "VLLM_SPECDEC_ADAPTIVE_REQUEST_STEP" "${SPECDEC_ADAPTIVE_GATE_REQUEST_STEP}"
append_specdec_adaptive_gate_env "VLLM_SPECDEC_ADAPTIVE_MIN_TOKEN_THRESHOLD" "${SPECDEC_ADAPTIVE_GATE_MIN_TOKEN_THRESHOLD}"
append_specdec_adaptive_gate_env "VLLM_SPECDEC_ADAPTIVE_MAX_TOKEN_THRESHOLD" "${SPECDEC_ADAPTIVE_GATE_MAX_TOKEN_THRESHOLD}"
append_specdec_adaptive_gate_env "VLLM_SPECDEC_ADAPTIVE_TOKEN_STEP" "${SPECDEC_ADAPTIVE_GATE_TOKEN_STEP}"
append_specdec_adaptive_gate_env "VLLM_SPECDEC_DYNAMIC_DRAFT_TOKENS" "${SPECDEC_DYNAMIC_DRAFT_TOKENS}"
append_specdec_adaptive_gate_env "VLLM_SPECDEC_DYNAMIC_DRAFT_SMALL_REQUEST_THRESHOLD" "${SPECDEC_DYNAMIC_DRAFT_SMALL_REQUEST_THRESHOLD}"
append_specdec_adaptive_gate_env "VLLM_SPECDEC_DYNAMIC_DRAFT_MEDIUM_REQUEST_THRESHOLD" "${SPECDEC_DYNAMIC_DRAFT_MEDIUM_REQUEST_THRESHOLD}"
append_specdec_adaptive_gate_env "VLLM_SPECDEC_DYNAMIC_DRAFT_SMALL_TOKEN_THRESHOLD" "${SPECDEC_DYNAMIC_DRAFT_SMALL_TOKEN_THRESHOLD}"
append_specdec_adaptive_gate_env "VLLM_SPECDEC_DYNAMIC_DRAFT_MEDIUM_TOKEN_THRESHOLD" "${SPECDEC_DYNAMIC_DRAFT_MEDIUM_TOKEN_THRESHOLD}"
append_specdec_adaptive_gate_env "VLLM_SPECDEC_DYNAMIC_DRAFT_SMALL_TOKENS" "${SPECDEC_DYNAMIC_DRAFT_SMALL_TOKENS}"
append_specdec_adaptive_gate_env "VLLM_SPECDEC_DYNAMIC_DRAFT_MEDIUM_TOKENS" "${SPECDEC_DYNAMIC_DRAFT_MEDIUM_TOKENS}"
append_specdec_adaptive_gate_env "VLLM_SPECDEC_DYNAMIC_DRAFT_LARGE_TOKENS" "${SPECDEC_DYNAMIC_DRAFT_LARGE_TOKENS}"

COMMAND="NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS:-true} \
RAY_CGRAPH_GET_TIMEOUT=${RAY_CGRAPH_GET_TIMEOUT} \
RAY_CGRAPH_get_timeout=${RAY_CGRAPH_GET_TIMEOUT} \
UV_PYTHON=${UV_PYTHON} \
UV_PROJECT_ENVIRONMENT=${DRIVER_UV_PROJECT_ENVIRONMENT} \
PYTHONPATH=${SCRIPT_DIR}:${PYTHONPATH:-} \
NRL_MEGATRON_CHECKPOINT_DIR=${NRL_MEGATRON_CHECKPOINT_DIR} \
NRL_MEGATRON_NCCL_TIMEOUT_SECONDS=${NRL_MEGATRON_NCCL_TIMEOUT_SECONDS} \
VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND} \
VLLM_SPECDEC_BATCH_SIZE_GATE_THRESHOLD=${SPECDEC_SCHEDULER_PATCH_GATE_THRESHOLD:-0} \
VLLM_SPECDEC_BATCH_TOKEN_GATE_THRESHOLD=${SPECDEC_SCHEDULER_PATCH_GATE_TOKEN_THRESHOLD:-0} \
VLLM_ENABLE_RUNTIME_SPECDEC_BATCH_GATE_PATCH=${ENABLE_RUNTIME_SPECDEC_GATE_PATCH} \
VLLM_SPECDEC_BATCH_GATE_LOG_INTERVAL=${VLLM_SPECDEC_BATCH_GATE_LOG_INTERVAL} \
NRL_VLLM_DISABLE_LOG_STATS=${NRL_VLLM_DISABLE_LOG_STATS} \
NRL_VLLM_OMIT_GENERATION_LOGPROBS=${NRL_VLLM_OMIT_GENERATION_LOGPROBS} \
NRL_STOP_AFTER_GENERATION=${NRL_STOP_AFTER_GENERATION} \
NRL_SPECDEC_DRAFT_MODEL_PROVENANCE=${DRAFT_MODEL_PROVENANCE:-unverified} \
NRL_SPECDEC_STEP_ADAPTIVE_CONTROLLER=${NRL_SPECDEC_STEP_ADAPTIVE_CONTROLLER} \
NRL_SPECDEC_CONTROLLER_MIN_DRAFT_TOKENS=${NRL_SPECDEC_CONTROLLER_MIN_DRAFT_TOKENS} \
NRL_SPECDEC_CONTROLLER_LOW_ACCEPTANCE=${NRL_SPECDEC_CONTROLLER_LOW_ACCEPTANCE} \
NRL_SPECDEC_CONTROLLER_HIGH_ACCEPTANCE=${NRL_SPECDEC_CONTROLLER_HIGH_ACCEPTANCE} \
NRL_SPECDEC_CONTROLLER_POS2_FLOOR=${NRL_SPECDEC_CONTROLLER_POS2_FLOOR} \
NRL_SPECDEC_CONTROLLER_POS3_FLOOR=${NRL_SPECDEC_CONTROLLER_POS3_FLOOR} \
NRL_SPECDEC_CONTROLLER_MIN_K=${NRL_SPECDEC_CONTROLLER_MIN_K} \
NRL_SPECDEC_CONTROLLER_MAX_K=${NRL_SPECDEC_CONTROLLER_MAX_K} \
NRL_SPECDEC_CONTROLLER_ALLOW_INCREASE=${NRL_SPECDEC_CONTROLLER_ALLOW_INCREASE} \
${SPECDEC_ADAPTIVE_GATE_ENV} \
uv run --python ${UV_PYTHON} --locked --extra mcore --directory ${SCRIPT_DIR} python ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.model_name=${TARGET_MODEL_ID} \
policy.generation.vllm_cfg.tensor_parallel_size=16 \
policy.generation.vllm_cfg.expert_parallel_size=1 \
policy.generation.vllm_cfg.pipeline_parallel_size=1 \
policy.generation.vllm_cfg.enforce_eager=false \
policy.generation.vllm_cfg.async_engine=false \
policy.megatron_cfg.tensor_model_parallel_size=2 \
policy.megatron_cfg.expert_model_parallel_size=16 \
policy.megatron_cfg.pipeline_model_parallel_size=8 \
policy.megatron_cfg.context_parallel_size=2 \
policy.megatron_cfg.sequence_parallel=true \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.num_prompts_per_step=${NUM_PROMPTS} \
grpo.num_generations_per_prompt=${NUM_GENERATIONS} \
policy.sequence_packing.enabled=true \
policy.train_global_batch_size=${TRAIN_GLOBAL_BATCH_SIZE} \
grpo.max_num_steps=${MAX_STEPS} \
++policy.generation.vllm_kwargs.speculative_config.method=${SPECDEC_METHOD} \
++policy.generation.vllm_kwargs.speculative_config.model=${DRAFT_MODEL} \
++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=${NUM_SPECULATIVE_TOKENS} \
++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=${DRAFT_TP} \
${SPECDEC_EXTRA_OVERRIDES} \
logger.wandb_enabled=true \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"

CONTAINER="${CONTAINER}" \
HF_HOME="${HF_HOME}" \
HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
WANDB_API_KEY="${WANDB_API_KEY:-}" \
NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR}" \
UV_PYTHON="${UV_PYTHON}" \
RAY_VERSION="${RAY_VERSION}" \
DRIVER_UV_PROJECT_ENVIRONMENT="${DRIVER_UV_PROJECT_ENVIRONMENT}" \
GPUS_PER_NODE="${GPUS_PER_NODE}" \
MOUNTS="${MOUNTS}" \
COMMAND="${COMMAND}" \
sbatch \
  --nodes="${NUM_NODES}" \
  --account="${ACCOUNT}" \
  --job-name="qwen235b-${JOB_TAG}-N${NUM_NODES}xG${GPUS_PER_NODE}" \
  --partition="${PARTITION}" \
  --time=04:00:00 \
  ${GRES_FLAG} \
  ${SBATCH_RESOURCE_ARGS} \
  ${SBATCH_EXTRA_ARGS} \
  --segment "${SEGMENT}" \
  "${SCRIPT_DIR}/ray.sub"
