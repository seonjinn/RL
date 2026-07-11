#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

MODE="${1:-test-only}"
MODEL_SELECTION="${2:-all}"
VARIANT_SELECTION="${3:-all}"
MAX_STEPS="${MAX_STEPS:-20}"
ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
PARTITION="${PARTITION:-gb200}"
CLUSTER_NAME="${CLUSTER_NAME:-lyris-gb200}"
RUNTIME_NAME="${RUNTIME_NAME:-nemo-rl}"
RUNTIME_VERSION="${RUNTIME_VERSION:-nightly-20260707}"
VLLM_VERSION="${VLLM_VERSION:-0.24.0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
MAX_OSL="${MAX_OSL:-4096}"
SPECDEC_CONTEXT_HEADROOM_TOKENS="${SPECDEC_CONTEXT_HEADROOM_TOKENS:-32}"
MAX_MODEL_LEN=$((MAX_OSL + SPECDEC_CONTEXT_HEADROOM_TOKENS))
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-4096}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
NUM_GENERATIONS="${NUM_GENERATIONS:-32}"
TRAIN_GBS="${TRAIN_GBS:-512}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1024}"
GENERATION_EP="${GENERATION_EP:-1}"
SAMPLING="${SAMPLING:-standard}"
DRAFT_SAMPLE_METHOD="${DRAFT_SAMPLE_METHOD:-probabilistic}"
TAIL_GATE_THRESHOLD="${TAIL_GATE_THRESHOLD:-32}"
TAIL_GATE_CONSECUTIVE_CHECKS="${TAIL_GATE_CONSECUTIVE_CHECKS:-10}"
CLUSTER_GPUS_PER_NODE="${CLUSTER_GPUS_PER_NODE:-4}"
CLUSTER_NUM_NODES="${CLUSTER_NUM_NODES:-4}"
LYRIS_ROOT="${LYRIS_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}"
CONTAINER="${CONTAINER:-${LYRIS_ROOT}/containers/nemo_rl_nightly_20260707.sqsh}"
HF_HOME="${HF_HOME:-${LYRIS_ROOT}/hf_home}"
WANDB_PROJECT="${WANDB_PROJECT:-nemorl-vllm024-tail-gated-lyris}"
WANDB_ENTITY="${WANDB_ENTITY:-nvidia}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-${LYRIS_ROOT}/.secrets/wandb_api_key}"
RUN_TAG="${RUN_TAG:-vllm024-tail-gated-step20-20260710}"
ATTEMPT_ID="${ATTEMPT_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${LYRIS_ROOT}/experiments/vllm024-tail-gated/${RUN_TAG}}"
QWEN30_ROOFLINE_CONFIG="${QWEN30_ROOFLINE_CONFIG:-${LYRIS_ROOT}/experiments/vllm024-tail-gated/calibrations/qwen-qwen3-30b-a3b-tp1-dtp1-lyris-gb200-k1-3-5.json}"
QWEN32_ROOFLINE_CONFIG="${QWEN32_ROOFLINE_CONFIG:-${LYRIS_ROOT}/experiments/vllm024-tail-gated/calibrations/qwen-qwen3-32b-tp2-dtp1-lyris-gb200-k1-3-5.json}"
QWEN30_TARGET_CHECKPOINT_REVISION="${QWEN30_TARGET_CHECKPOINT_REVISION:-}"
QWEN30_DRAFT_CHECKPOINT_REVISION="${QWEN30_DRAFT_CHECKPOINT_REVISION:-a7ec796dd65236f1ecd4ed2958a7f0689e5da5cf}"
QWEN30_CALIBRATION_TIMESTAMP="${QWEN30_CALIBRATION_TIMESTAMP:-}"
QWEN32_TARGET_CHECKPOINT_REVISION="${QWEN32_TARGET_CHECKPOINT_REVISION:-9216db5781bf21249d130ec9da846c4624c16137}"
QWEN32_DRAFT_CHECKPOINT_REVISION="${QWEN32_DRAFT_CHECKPOINT_REVISION:-dc84fe7ff1db31efa824776f49c141fc8195eb47}"
QWEN30_TARGET_MODEL="${QWEN30_TARGET_MODEL:-${HF_HOME}/hub/models--Qwen--Qwen3-30B-A3B/snapshots/${QWEN30_TARGET_CHECKPOINT_REVISION}}"
QWEN32_TARGET_MODEL="${QWEN32_TARGET_MODEL:-${HF_HOME}/hub/models--Qwen--Qwen3-32B/snapshots/${QWEN32_TARGET_CHECKPOINT_REVISION}}"
QWEN32_CALIBRATION_TIMESTAMP="${QWEN32_CALIBRATION_TIMESTAMP:-}"
CALIBRATION_CLUSTER="${CALIBRATION_CLUSTER:-lyris-gb200}"
VLLM_COMMIT="${VLLM_COMMIT:-ee0da84a}"
WALLTIME="${WALLTIME:-04:00:00}"
TMPDIR="${TMPDIR_OVERRIDE:-/tmp}"
PERSONAL_BRANCH_PREFIX="${PERSONAL_BRANCH_PREFIX:-sna/}"
SCHEDULER_CLASS="nemo_rl.models.generation.vllm.tail_gate_scheduler.TailGatedScheduler"

validate_positive_integer() {
  local setting="$1"
  local value="$2"

  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'ERROR: %s must be a positive integer, got %s\n' "${setting}" "${value}" >&2
    exit 2
  fi
}

build_specdec_cudagraph_capture_sizes() {
  local draft_k="$1"
  local max_requests="$2"
  local query_length=$((draft_k + 1))
  local request_count
  local -a request_buckets=()
  local -a token_buckets=()

  for request_count in 1 2 4; do
    if ((request_count <= max_requests)); then
      request_buckets+=("${request_count}")
    fi
  done
  for ((request_count = 8; request_count <= max_requests; request_count += 8)); do
    request_buckets+=("${request_count}")
  done
  if ((max_requests > 4 && max_requests % 8 != 0)); then
    request_buckets+=("${max_requests}")
  fi
  for request_count in "${request_buckets[@]}"; do
    token_buckets+=("$((request_count * query_length))")
  done

  local IFS=,
  printf '[%s]' "${token_buckets[*]}"
}

validate_positive_integer "TAIL_GATE_THRESHOLD" "${TAIL_GATE_THRESHOLD}"
validate_positive_integer "TAIL_GATE_CONSECUTIVE_CHECKS" "${TAIL_GATE_CONSECUTIVE_CHECKS}"

validate_immutable_revision() {
  local name="$1"
  local revision="$2"

  if [[ ! "${revision}" =~ ^[0-9a-f]{40}$ ]]; then
    printf 'ERROR: %s must be an exact 40-character hexadecimal revision\n' \
      "${name}" >&2
    exit 2
  fi
}
case "${DRAFT_SAMPLE_METHOD}" in
  greedy|probabilistic)
    ;;
  *)
    printf 'ERROR: DRAFT_SAMPLE_METHOD must be greedy or probabilistic, got %s\n' \
      "${DRAFT_SAMPLE_METHOD}" >&2
    exit 2
    ;;
esac

if [[ -z "${REPO_DIR:-}" ]]; then
  logical_pwd="$(pwd -L)"
  repo_prefix="$(git rev-parse --show-prefix)"
  if [[ -n "${repo_prefix}" ]]; then
    REPO_DIR="${logical_pwd%/${repo_prefix%/}}"
  else
    REPO_DIR="${logical_pwd}"
  fi
fi
if [[ -d "${REPO_DIR}" ]]; then
  REPO_DIR="$(readlink -f "${REPO_DIR}")"
fi

case "${MODE}" in
  dry-run|test-only|submit)
    ;;
  *)
    echo "ERROR: mode must be dry-run, test-only, or submit" >&2
    exit 2
    ;;
esac

case "${MODEL_SELECTION}" in
  all) models=(qwen30ba3b qwen32b) ;;
  qwen30ba3b|qwen32b) models=("${MODEL_SELECTION}") ;;
  *)
    echo "ERROR: model must be all, qwen30ba3b, or qwen32b" >&2
    exit 2
    ;;
esac

case "${VARIANT_SELECTION}" in
  all)
    variants=(
      baseline_v1
      always_on_v1_k5
      stock_dynamic_v1
      baseline_v2
      always_on_v2_k5
      fastrl_threshold_v2_k5
      efficient_roofline_v2_k5
    )
    ;;
  baseline_v1|always_on_v1_k5|stock_dynamic_v1|baseline_v2|always_on_v2_k5|fastrl_threshold_v2_k5|efficient_roofline_v2_k5)
    variants=("${VARIANT_SELECTION}")
    ;;
  *)
    echo "ERROR: variant must be all or one of the seven planned tail-gate variants" >&2
    exit 2
    ;;
esac

if [[ "${MODE}" != "dry-run" && ! -f "${CONTAINER}" ]]; then
  echo "ERROR: container not found: ${CONTAINER}" >&2
  exit 2
fi

if [[ "${MODE}" == "submit" && -z "${WANDB_API_KEY:-}" ]]; then
  if [[ -r "${WANDB_API_KEY_FILE}" ]]; then
    WANDB_API_KEY="$(<"${WANDB_API_KEY_FILE}")"
  else
    echo "ERROR: set WANDB_API_KEY or create ${WANDB_API_KEY_FILE}" >&2
    exit 2
  fi
  export WANDB_API_KEY
fi

sha256_file() {
  shasum -a 256 "$1" | awk '{print $1}'
}

json_argv() {
  python3 - "$@" <<'PY'
import json
import sys

print(json.dumps(sys.argv[1:], ensure_ascii=True, separators=(",", ":")))
PY
}

validate_roofline_config() {
  local config_path="$1"
  local expected_model="$2"
  local expected_target_tp="$3"
  local expected_draft_tp="$4"
  local expected_container="$5"
  local expected_container_sha256="$6"
  local expected_target_revision="$7"
  local expected_draft_revision="$8"
  local expected_calibration_timestamp="$9"
  local expected_cluster="${10}"
  local expected_vllm_commit="${11}"

  python3 - \
    "${config_path}" \
    "${expected_model}" \
    "${expected_target_tp}" \
    "${expected_draft_tp}" \
    "${expected_container}" \
    "${expected_container_sha256}" \
    "${expected_target_revision}" \
    "${expected_draft_revision}" \
    "${expected_calibration_timestamp}" \
    "${expected_cluster}" \
    "${expected_vllm_commit}" <<'PY'
import json
import math
import sys

(
    config_path,
    expected_model,
    expected_target_tp,
    expected_draft_tp,
    expected_container,
    expected_container_sha256,
    expected_target_revision,
    expected_draft_revision,
    expected_calibration_timestamp,
    expected_cluster,
    expected_vllm_commit,
) = sys.argv[1:]

try:
    with open(config_path, encoding="utf-8") as config_file:
        payload = json.load(config_file)
    metadata = payload["metadata"]
except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
    print(f"ERROR: invalid roofline config: {config_path}: {error}", file=sys.stderr)
    raise SystemExit(2)

expected = {
    "model": expected_model,
    "target_tp": int(expected_target_tp),
    "draft_tp": int(expected_draft_tp),
    "container": expected_container,
    "container_sha256": expected_container_sha256,
    "target_checkpoint_revision": expected_target_revision,
    "draft_checkpoint_revision": expected_draft_revision,
    "calibration_timestamp": expected_calibration_timestamp,
    "cluster": expected_cluster,
    "vllm_commit": expected_vllm_commit,
}
for field, expected_value in expected.items():
    actual_value = metadata.get(field)
    if actual_value != expected_value:
        print(
            f"ERROR: roofline metadata mismatch: {field}: "
            f"expected {expected_value!r}, got {actual_value!r}",
            file=sys.stderr,
        )
        raise SystemExit(2)

k_values = metadata.get("k_values")
if (
    not isinstance(k_values, list)
    or not all(type(value) is int for value in k_values)
    or 5 not in k_values
):
    print(
        f"ERROR: roofline metadata mismatch: k_values: "
        f"expected exact integer K5, got {k_values!r}",
        file=sys.stderr,
    )
    raise SystemExit(2)

per_gamma = payload.get("calibration", {}).get("per_gamma", {})
k5_calibration = per_gamma.get("5") if isinstance(per_gamma, dict) else None
if not isinstance(k5_calibration, dict):
    print(
        'ERROR: roofline config requires exact calibration.per_gamma["5"]',
        file=sys.stderr,
    )
    raise SystemExit(2)
for field in ("c_T", "c_D", "c_V"):
    value = k5_calibration.get(field)
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        print(
            f'ERROR: calibration.per_gamma["5"].{field} must be positive',
            file=sys.stderr,
        )
        raise SystemExit(2)

if payload.get("model", {}).get("name") != expected_model:
    print("ERROR: roofline metadata mismatch: model", file=sys.stderr)
    raise SystemExit(2)
if payload.get("hardware", {}).get("tp") != int(expected_target_tp):
    print("ERROR: roofline metadata mismatch: target_tp", file=sys.stderr)
    raise SystemExit(2)
PY
}

require_submit_checkout() {
  local branch
  local fork_head
  local head
  local remote_ref
  local untracked

  if ! git -C "${REPO_DIR}" diff --quiet --ignore-submodules=none \
    || ! git -C "${REPO_DIR}" diff --cached --quiet --ignore-submodules=none; then
    echo "ERROR: submit requires a clean tracked checkout" >&2
    exit 2
  fi
  untracked="$(git -C "${REPO_DIR}" status --porcelain=v1 \
    --untracked-files=all --ignore-submodules=none \
    | awk '$1 == "??" {print $2}' \
    | grep -Ev '^(tests/unit/unit_results\.json|tests/unit/unit_results/.+)$' || true)"
  if [[ -n "${untracked}" ]]; then
    echo "ERROR: submit rejects untracked files (except tests/unit/unit_results artifacts)" >&2
    printf '%s\n' "${untracked}" >&2
    exit 2
  fi
  if ! git -C "${REPO_DIR}" ls-files --error-unmatch \
    experiments/vllm_024_upgrade/submit_tail_gated_specdec_step20.sh >/dev/null 2>&1; then
    echo "ERROR: launcher must be committed before submit" >&2
    exit 2
  fi
  branch="$(git -C "${REPO_DIR}" branch --show-current)"
  if [[ "${branch}" != "${PERSONAL_BRANCH_PREFIX}"* ]]; then
    echo "ERROR: submit requires a ${PERSONAL_BRANCH_PREFIX} personal branch" >&2
    exit 2
  fi
  remote_ref="refs/remotes/fork/${branch}"
  if ! git -C "${REPO_DIR}" fetch --quiet --no-tags fork \
    "+refs/heads/${branch}:${remote_ref}"; then
    echo "ERROR: submit could not fetch fork/${branch}" >&2
    exit 2
  fi
  head="$(git -C "${REPO_DIR}" rev-parse HEAD)"
  fork_head="$(git -C "${REPO_DIR}" rev-parse --verify "${remote_ref}")"
  if [[ "${head}" != "${fork_head}" ]]; then
    echo "ERROR: submit HEAD must exactly match ${remote_ref} after fetch" >&2
    exit 2
  fi
}

if [[ "${MODE}" == "submit" ]]; then
  require_submit_checkout
fi

submit_one() {
  local model="$1"
  local variant="$2"
  local recipe
  local target_tp
  local draft_tp=1
  local dp
  local draft_model
  local target_model
  local expected_model
  local expected_target_revision
  local expected_draft_revision
  local expected_calibration_timestamp
  local roofline_config
  local runner
  local use_v2_runner
  local graph_mode
  local gate_mode="off"
  local draft_k=0
  local threshold=""
  local consecutive_checks=""
  local roofline_hash=""
  local manifest_draft_sample_method="not_applicable"
  local manifest_draft_checkpoint="not_applicable"
  local local_rollout_capacity
  local cudagraph_max_requests=""
  local cudagraph_max_tokens=""
  local cudagraph_capture_sizes=""
  local job_attempt_id="${ATTEMPT_ID//[^[:alnum:]_.-]/-}"
  local container_path="${CONTAINER}"
  local index
  local run_dir="${EXPERIMENT_ROOT}/${model}/${variant}"
  local wandb_run_id="${RUN_TAG}-${ATTEMPT_ID}-${model}-${variant}"
  local triton_cache_dir="/tmp/nemorl-tail-gate-triton-${RUN_TAG}-${model}-${variant}"
  local inductor_cache_dir="/tmp/nemorl-tail-gate-inductor-${RUN_TAG}-${model}-${variant}"
  local venv_dir="/tmp/nemorl-tail-gate-venvs-${RUN_TAG}-${ATTEMPT_ID}-${model}-${variant}"

  case "${model}" in
    qwen30ba3b)
      recipe="examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml"
      target_tp=1
      expected_model="Qwen/Qwen3-30B-A3B"
      expected_target_revision="${QWEN30_TARGET_CHECKPOINT_REVISION}"
      expected_draft_revision="${QWEN30_DRAFT_CHECKPOINT_REVISION}"
      expected_calibration_timestamp="${QWEN30_CALIBRATION_TIMESTAMP}"
      target_model="${QWEN30_TARGET_MODEL}"
      draft_model="${QWEN30_DRAFT_MODEL:-${HF_HOME}/hub/models--RedHatAI--Qwen3-30B-A3B-Thinking-2507-speculator.eagle3/snapshots/a7ec796dd65236f1ecd4ed2958a7f0689e5da5cf}"
      roofline_config="${QWEN30_ROOFLINE_CONFIG}"
      ;;
    qwen32b)
      recipe="examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml"
      target_tp=2
      expected_model="Qwen/Qwen3-32B"
      expected_target_revision="${QWEN32_TARGET_CHECKPOINT_REVISION}"
      expected_draft_revision="${QWEN32_DRAFT_CHECKPOINT_REVISION}"
      expected_calibration_timestamp="${QWEN32_CALIBRATION_TIMESTAMP}"
      target_model="${QWEN32_TARGET_MODEL}"
      draft_model="${QWEN32_DRAFT_MODEL:-${HF_HOME}/hub/models--RedHatAI--Qwen3-32B-speculator.eagle3/snapshots/dc84fe7ff1db31efa824776f49c141fc8195eb47}"
      roofline_config="${QWEN32_ROOFLINE_CONFIG}"
      ;;
  esac

  dp=$((CLUSTER_GPUS_PER_NODE * CLUSTER_NUM_NODES / target_tp))

  case "${variant}" in
    baseline_v1)
      runner=v1
      use_v2_runner=0
      graph_mode="PIECEWISE"
      ;;
    always_on_v1_k5)
      runner=v1
      use_v2_runner=0
      graph_mode="PIECEWISE"
      draft_k=5
      ;;
    stock_dynamic_v1)
      runner=v1
      use_v2_runner=0
      graph_mode="PIECEWISE"
      draft_k=5
      ;;
    baseline_v2)
      runner=v2
      use_v2_runner=1
      graph_mode="FULL_AND_PIECEWISE"
      ;;
    always_on_v2_k5)
      runner=v2
      use_v2_runner=1
      graph_mode="FULL_AND_PIECEWISE"
      draft_k=5
      ;;
    fastrl_threshold_v2_k5)
      runner=v2
      use_v2_runner=1
      graph_mode="FULL_AND_PIECEWISE"
      gate_mode="threshold"
      draft_k=5
      threshold="${TAIL_GATE_THRESHOLD}"
      consecutive_checks="${TAIL_GATE_CONSECUTIVE_CHECKS}"
      ;;
    efficient_roofline_v2_k5)
      runner=v2
      use_v2_runner=1
      graph_mode="FULL_AND_PIECEWISE"
      gate_mode="roofline"
      draft_k=5
      threshold="${TAIL_GATE_THRESHOLD}"
      consecutive_checks="${TAIL_GATE_CONSECUTIVE_CHECKS}"
      if [[ "${MODE}" != "dry-run" && ! -f "${roofline_config}" ]]; then
        echo "ERROR: roofline config not found: ${roofline_config}" >&2
        exit 2
      fi
      if [[ -f "${roofline_config}" ]]; then
        roofline_hash="$(sha256_file "${roofline_config}")"
      fi
      if [[ "${MODE}" == "submit" ]]; then
        if [[ -z "${expected_target_revision}" || -z "${expected_draft_revision}" \
          || -z "${expected_calibration_timestamp}" ]]; then
          echo "ERROR: submit requires exact target/draft revisions and calibration timestamp for ${model}" >&2
          exit 2
        fi
        validate_roofline_config \
          "${roofline_config}" \
          "${expected_model}" \
          "${target_tp}" \
          "${draft_tp}" \
          "${CONTAINER}" \
          "$(sha256_file "${CONTAINER}")" \
          "${expected_target_revision}" \
          "${expected_draft_revision}" \
          "${expected_calibration_timestamp}" \
          "${CALIBRATION_CLUSTER}" \
          "${VLLM_COMMIT}"
      fi
      ;;
  esac

  local_rollout_capacity=$(((NUM_PROMPTS * NUM_GENERATIONS + dp - 1) / dp))
  if [[ "${draft_k}" != "0" ]]; then
    cudagraph_max_requests="${CUDAGRAPH_MAX_REQUESTS:-${local_rollout_capacity}}"
    validate_positive_integer "CUDAGRAPH_MAX_REQUESTS" "${cudagraph_max_requests}"
    if ((cudagraph_max_requests > MAX_NUM_SEQS)); then
      printf 'ERROR: CUDAGRAPH_MAX_REQUESTS=%s exceeds MAX_NUM_SEQS=%s\n' \
        "${cudagraph_max_requests}" "${MAX_NUM_SEQS}" >&2
      exit 2
    fi
    cudagraph_max_tokens=$((cudagraph_max_requests * (draft_k + 1)))
    cudagraph_capture_sizes="$(
      build_specdec_cudagraph_capture_sizes "${draft_k}" "${cudagraph_max_requests}"
    )"
  fi

  if [[ "${MODE}" != "dry-run" && ! -f "${REPO_DIR}/${recipe}" ]]; then
    echo "ERROR: recipe not found: ${REPO_DIR}/${recipe}" >&2
    exit 2
  fi
  if [[ "${MODE}" == "submit" ]]; then
    validate_immutable_revision "target checkpoint revision for ${model}" \
      "${expected_target_revision}"
    if [[ ! -d "${target_model}" ]]; then
      echo "ERROR: immutable target checkpoint not found: ${target_model}" >&2
      exit 2
    fi
    target_model="$(readlink -f "${target_model}")"
    if [[ "$(basename "${target_model}")" != "${expected_target_revision}" ]]; then
      echo "ERROR: target checkpoint path does not match revision ${expected_target_revision}: ${target_model}" >&2
      exit 2
    fi
  fi
  if [[ "${MODE}" == "submit" ]] \
    && ! git -C "${REPO_DIR}" ls-files --error-unmatch "${recipe}" >/dev/null 2>&1; then
    echo "ERROR: recipe must be tracked before submit: ${recipe}" >&2
    exit 2
  fi
  if [[ "${MODE}" != "dry-run" && "${draft_k}" != "0" && ! -d "${draft_model}" ]]; then
    echo "ERROR: draft model directory not found: ${draft_model}" >&2
    exit 2
  fi
  if [[ "${draft_k}" != "0" && -d "${draft_model}" ]]; then
    draft_model="$(readlink -f "${draft_model}")"
  fi

  local overrides=(
    "grpo.max_num_steps=${MAX_STEPS}"
    "grpo.num_prompts_per_step=${NUM_PROMPTS}"
    "grpo.num_generations_per_prompt=${NUM_GENERATIONS}"
    "checkpointing.enabled=false"
    "checkpointing.checkpoint_dir=${run_dir}/checkpoints"
    "policy.model_name=${target_model}"
    "policy.train_global_batch_size=${TRAIN_GBS}"
    "policy.max_total_sequence_length=${MAX_SEQUENCE_LENGTH}"
    "policy.generation.max_new_tokens=${MAX_OSL}"
    "policy.generation.temperature=${TEMPERATURE}"
    "policy.generation.top_p=${TOP_P}"
    "++policy.generation._output_max_model_len=${MAX_OSL}"
    "policy.generation.vllm_cfg.max_model_len=${MAX_MODEL_LEN}"
    "policy.generation.vllm_cfg.tensor_parallel_size=${target_tp}"
    "policy.generation.vllm_cfg.expert_parallel_size=${GENERATION_EP}"
    "policy.generation.vllm_cfg.enforce_eager=false"
    "policy.generation.vllm_cfg.enable_vllm_metrics_logger=true"
    "policy.generation.vllm_cfg.vllm_metrics_logger_interval=0.5"
    "++policy.generation.vllm_cfg.env_vars.NRL_VLLM_ENABLE_CUDAGRAPH_DISPATCH_METRICS=true"
    "++policy.generation.vllm_kwargs.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
    "++policy.generation.vllm_kwargs.max_num_seqs=${MAX_NUM_SEQS}"
    "++policy.generation.vllm_kwargs.moe_backend=triton"
    "++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=${graph_mode}"
    "cluster.gpus_per_node=${CLUSTER_GPUS_PER_NODE}"
    "cluster.num_nodes=${CLUSTER_NUM_NODES}"
    "cluster.segment_size=${CLUSTER_GPUS_PER_NODE}"
    "logger.wandb_enabled=true"
    "logger.tensorboard_enabled=false"
    "logger.wandb.project=${WANDB_PROJECT}"
    "logger.wandb.name=${wandb_run_id}"
    "++logger.wandb.entity=${WANDB_ENTITY}"
    "logger.log_dir=${run_dir}/nemo_logs"
  )

  if [[ "${draft_k}" != "0" ]]; then
    manifest_draft_sample_method="${DRAFT_SAMPLE_METHOD}"
    manifest_draft_checkpoint="${draft_model}"
    overrides+=(
      "++policy.generation.vllm_kwargs.speculative_config.method=eagle3"
      "++policy.generation.vllm_kwargs.speculative_config.model=${draft_model}"
      "++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=${draft_k}"
      "++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=${draft_tp}"
      "++policy.generation.vllm_kwargs.speculative_config.rejection_sample_method=${SAMPLING}"
      "++policy.generation.vllm_kwargs.speculative_config.draft_sample_method=${DRAFT_SAMPLE_METHOD}"
      "++policy.generation.vllm_kwargs.compilation_config.max_cudagraph_capture_size=${cudagraph_max_tokens}"
      "++policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes=${cudagraph_capture_sizes}"
    )
  fi
  if [[ "${variant}" == "stock_dynamic_v1" ]]; then
    overrides+=(
      "++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens_per_batch_size=[[1,16,5],[17,32,4],[33,64,3],[65,128,1],[129,512,0]]"
    )
  fi
  if [[ "${gate_mode}" != "off" ]]; then
    overrides+=(
      "++policy.generation.vllm_kwargs.scheduler_cls=${SCHEDULER_CLASS}"
      "++policy.generation.vllm_kwargs.speculative_config.sd_tail_gate_mode=${gate_mode}"
      "++policy.generation.vllm_kwargs.speculative_config.sd_tail_gate_threshold=${threshold}"
      "++policy.generation.vllm_kwargs.speculative_config.sd_tail_gate_consecutive_checks=${consecutive_checks}"
      "++policy.generation.vllm_kwargs.speculative_config.sd_tail_gate_off_mode=advance_only"
    )
    if [[ "${gate_mode}" == "roofline" ]]; then
      overrides+=(
        "++policy.generation.vllm_kwargs.speculative_config.sd_tail_gate_margin=0.05"
        "++policy.generation.vllm_kwargs.speculative_config.sd_tail_gate_config_path=${roofline_config}"
      )
    fi
  fi

  local command_parts=(
    env
    "VLLM_USE_V2_MODEL_RUNNER=${use_v2_runner}"
    "NRL_VLLM_ENABLE_CUDAGRAPH_DISPATCH_METRICS=true"
    "WANDB_RUN_ID=${wandb_run_id}"
    "WANDB_RUN_GROUP=${RUN_TAG}"
    "WANDB_RESUME=never"
    "NEMO_RL_VENV_DIR=${venv_dir}"
    "NRL_FORCE_REBUILD_VENVS=true"
    "PYTHONPATH=${REPO_DIR}"
    "TRITON_CACHE_DIR=${triton_cache_dir}"
    "TORCHINDUCTOR_CACHE_DIR=${inductor_cache_dir}"
    uv
    run
    examples/run_grpo.py
    --config "${recipe}"
    "${overrides[@]}"
  )
  local command
  printf -v command '%q ' "${command_parts[@]}"
  command="${command% }"
  local command_argv_json
  command_argv_json="$(json_argv "${command_parts[@]}")"

  if [[ -f "${CONTAINER}" ]]; then
    container_path="$(readlink -f "${CONTAINER}")"
  fi

  local environment=(
    "CONTAINER=${container_path}"
    "MOUNTS=/lustre:/lustre"
    "CONTAINER_WORKDIR=${REPO_DIR}"
    "COMMAND=${command}"
    "BASE_LOG_DIR=${run_dir}"
    "GPUS_PER_NODE=${CLUSTER_GPUS_PER_NODE}"
    "HF_HOME=${HF_HOME}"
    "PYTHONPATH=${REPO_DIR}"
    "PYTHONDONTWRITEBYTECODE=1"
    "RAY_LOG_SYNC_FREQUENCY=60"
    "TMPDIR=${TMPDIR}"
    "TRITON_CACHE_DIR=${triton_cache_dir}"
    "TORCHINDUCTOR_CACHE_DIR=${inductor_cache_dir}"
  )
  local runtime_commit="${RUNTIME_COMMIT:-dry-run-unresolved}"
  if [[ "${MODE}" == "submit" ]]; then
    runtime_commit="$(git -C "${REPO_DIR}" rev-parse HEAD)"
  fi
  environment+=(
    "NRL_RUNTIME_CHECKOUT=${REPO_DIR}"
    "NRL_EXPECTED_RUNTIME_COMMIT=${runtime_commit}"
  )
  local sbatch_args=(
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --nodes="${CLUSTER_NUM_NODES}"
    --ntasks-per-node=1
    --exclusive
    --time="${WALLTIME}"
    --segment="${CLUSTER_GPUS_PER_NODE}"
    --job-name="${ACCOUNT}-nemorl.tail-gate-${model}-${variant}-${job_attempt_id}"
    --output="${run_dir}/slurm-%j.out"
    --open-mode=append
    --comment=metrics
  )
  local submission_argv=(
    env
    "${environment[@]}"
    sbatch
    --parsable
    "${sbatch_args[@]}"
    "${REPO_DIR}/ray.sub"
  )
  local launcher_command
  printf -v launcher_command '%q ' "${submission_argv[@]}"
  launcher_command="${launcher_command% }"
  local launcher_argv_json
  launcher_argv_json="$(json_argv "${submission_argv[@]}")"

  case "${MODE}" in
    dry-run)
      printf '[DRY-RUN] job model=%s variant=%s runner=%s graph_mode=%s gate_mode=%s k=%s\n' \
        "${model}" "${variant}" "${runner}" "${graph_mode}" "${gate_mode}" "${draft_k}"
      printf '[DRY-RUN] env'
      printf ' %q' "${environment[@]}"
      printf ' sbatch'
      printf ' %q' "${sbatch_args[@]}"
      printf ' %q\n' "${REPO_DIR}/ray.sub"
      printf '[DRY-RUN] command %s\n' "${command}"
      ;;
    test-only)
      mkdir -p "${run_dir}"
      local test_only_argv=("${submission_argv[@]}")
      for index in "${!test_only_argv[@]}"; do
        if [[ "${test_only_argv[${index}]}" == "--parsable" ]]; then
          test_only_argv[${index}]=--test-only
          break
        fi
      done
      "${test_only_argv[@]}"
      ;;
    submit)
      mkdir -p "${run_dir}"
      local manifest="${EXPERIMENT_ROOT}/submissions.tsv"
      local manifest_header=$'timestamp\tmodel\tvariant\tgate_mode\tk\tthreshold\tconsecutive_checks\troofline_config_sha256\tcluster\truntime\truntime_version\truntime_commit\tvllm_version\tvllm_commit\ttarget_tp\tdraft_tp\tdp\tep\ttemperature\ttop_p\tmax_osl\tmax_model_len\tmax_sequence_length\tnum_prompts\tnum_generations\ttrain_gbs\tmax_num_batched_tokens\tmax_num_seqs\tcudagraph_max_requests\tcudagraph_max_tokens\tcudagraph_capture_sizes\trecipe\tcontainer\tcontainer_sha256\trunner\tgraph_mode\tsampling\tdraft_sample_method\tjob_id\twandb_run_id\twandb_url\trun_dir\tslurm_log_path\tray_driver_log_path\tray_log_dir\tlauncher_command\tcommand\tcheckout_path\tray_sub_path\ttarget_checkpoint\ttarget_checkpoint_revision\tdraft_checkpoint\tcommand_argv_json\tlauncher_argv_json'
      if [[ -f "${manifest}" && "$(head -n 1 "${manifest}")" != "${manifest_header}" ]]; then
        echo "ERROR: submissions manifest header mismatch: ${manifest}" >&2
        exit 2
      fi
      if [[ ! -f "${manifest}" ]]; then
        printf '%s\n' "${manifest_header}" >"${manifest}"
      fi
      local test_only_argv=("${submission_argv[@]}")
      for index in "${!test_only_argv[@]}"; do
        if [[ "${test_only_argv[${index}]}" == "--parsable" ]]; then
          test_only_argv[${index}]=--test-only
          break
        fi
      done
      "${test_only_argv[@]}"
      local job_id
      job_id="$("${submission_argv[@]}")"
      local job_log_dir="${run_dir}/${job_id}-logs"
      local manifest_values=(
        "$(date --iso-8601=seconds)"
        "${model}"
        "${variant}"
        "${gate_mode}"
        "${draft_k}"
        "${threshold}"
        "${consecutive_checks}"
        "${roofline_hash}"
        "${CLUSTER_NAME}"
        "${RUNTIME_NAME}"
        "${RUNTIME_VERSION}"
        "${runtime_commit}"
        "${VLLM_VERSION}"
        "${VLLM_COMMIT}"
        "${target_tp}"
        "${draft_tp}"
        "${dp}"
        "${GENERATION_EP}"
        "${TEMPERATURE}"
        "${TOP_P}"
        "${MAX_OSL}"
        "${MAX_MODEL_LEN}"
        "${MAX_SEQUENCE_LENGTH}"
        "${NUM_PROMPTS}"
        "${NUM_GENERATIONS}"
        "${TRAIN_GBS}"
        "${MAX_NUM_BATCHED_TOKENS}"
        "${MAX_NUM_SEQS}"
        "${cudagraph_max_requests:-not_applicable}"
        "${cudagraph_max_tokens:-not_applicable}"
        "${cudagraph_capture_sizes:-not_applicable}"
        "${recipe}"
        "$(readlink -f "${CONTAINER}")"
        "$(sha256_file "${CONTAINER}")"
        "${runner}"
        "${graph_mode}"
        "${SAMPLING}"
        "${manifest_draft_sample_method}"
        "${job_id}"
        "${wandb_run_id}"
        "https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}/runs/${wandb_run_id}"
        "${run_dir}"
        "${run_dir}/slurm-${job_id}.out"
        "${job_log_dir}/ray-driver.log"
        "${job_log_dir}/ray"
        "${launcher_command}"
        "${command}"
        "${REPO_DIR}"
        "${REPO_DIR}/ray.sub"
        "${target_model}"
        "${expected_target_revision}"
        "${manifest_draft_checkpoint}"
        "${command_argv_json}"
        "${launcher_argv_json}"
      )
      (
        IFS=$'\t'
        printf '%s\n' "${manifest_values[*]}"
      ) >>"${manifest}"
      printf '%s\t%s\t%s\n' "${job_id}" "${model}" "${variant}"
      ;;
  esac
}

if [[ "${MODE}" != "dry-run" ]]; then
  mkdir -p "${EXPERIMENT_ROOT}"
fi
for model in "${models[@]}"; do
  for variant in "${variants[@]}"; do
    if [[ "${model}" == "qwen30ba3b" && "${variant}" == *_v2* \
      && ( "${MODEL_SELECTION}" == "all" || "${VARIANT_SELECTION}" == "all" ) ]]; then
      printf '[SKIP] model=%s variant=%s: Qwen30 V2 requires explicit variant selection\n' \
        "${model}" "${variant}" >&2
      continue
    fi
    submit_one "${model}" "${variant}"
  done
done
