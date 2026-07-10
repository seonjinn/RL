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
WALLTIME="${WALLTIME:-04:00:00}"
TMPDIR="${TMPDIR_OVERRIDE:-/tmp}"
PERSONAL_BRANCH_PREFIX="${PERSONAL_BRANCH_PREFIX:-sna/}"
SCHEDULER_CLASS="nemo_rl.models.generation.vllm.tail_gate_scheduler.TailGatedScheduler"

if [[ -z "${REPO_DIR:-}" ]]; then
  logical_pwd="$(pwd -L)"
  repo_prefix="$(git rev-parse --show-prefix)"
  if [[ -n "${repo_prefix}" ]]; then
    REPO_DIR="${logical_pwd%/${repo_prefix%/}}"
  else
    REPO_DIR="${logical_pwd}"
  fi
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

validate_roofline_config() {
  local config_path="$1"
  local expected_model="$2"
  local expected_target_tp="$3"
  local expected_draft_tp="$4"
  local expected_container="$5"
  local expected_container_sha256="$6"

  python3 - \
    "${config_path}" \
    "${expected_model}" \
    "${expected_target_tp}" \
    "${expected_draft_tp}" \
    "${expected_container}" \
    "${expected_container_sha256}" <<'PY'
import json
import sys

(
    config_path,
    expected_model,
    expected_target_tp,
    expected_draft_tp,
    expected_container,
    expected_container_sha256,
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
  if ! git -C "${REPO_DIR}" branch -r --contains HEAD | grep -q .; then
    echo "ERROR: submit requires HEAD to be present on a remote branch; push it yourself first" >&2
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
  local draft_model
  local expected_model
  local roofline_config
  local runner
  local use_v2_runner
  local graph_mode
  local gate_mode="off"
  local draft_k=0
  local threshold=""
  local consecutive_checks=""
  local roofline_hash=""
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
      draft_model="${QWEN30_DRAFT_MODEL:-${HF_HOME}/hub/models--RedHatAI--Qwen3-30B-A3B-Thinking-2507-speculator.eagle3/snapshots/a7ec796dd65236f1ecd4ed2958a7f0689e5da5cf}"
      roofline_config="${QWEN30_ROOFLINE_CONFIG}"
      ;;
    qwen32b)
      recipe="examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml"
      target_tp=2
      expected_model="Qwen/Qwen3-32B"
      draft_model="${QWEN32_DRAFT_MODEL:-${HF_HOME}/hub/models--RedHatAI--Qwen3-32B-speculator.eagle3/snapshots/dc84fe7ff1db31efa824776f49c141fc8195eb47}"
      roofline_config="${QWEN32_ROOFLINE_CONFIG}"
      ;;
  esac

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
      threshold=32
      consecutive_checks=10
      ;;
    efficient_roofline_v2_k5)
      runner=v2
      use_v2_runner=1
      graph_mode="FULL_AND_PIECEWISE"
      gate_mode="roofline"
      draft_k=5
      threshold=32
      consecutive_checks=10
      if [[ "${MODE}" != "dry-run" && ! -f "${roofline_config}" ]]; then
        echo "ERROR: roofline config not found: ${roofline_config}" >&2
        exit 2
      fi
      if [[ -f "${roofline_config}" ]]; then
        roofline_hash="$(sha256_file "${roofline_config}")"
      fi
      if [[ "${MODE}" == "submit" ]]; then
        validate_roofline_config \
          "${roofline_config}" \
          "${expected_model}" \
          "${target_tp}" \
          "${draft_tp}" \
          "${CONTAINER}" \
          "$(sha256_file "${CONTAINER}")"
      fi
      ;;
  esac

  if [[ "${MODE}" != "dry-run" && ! -f "${REPO_DIR}/${recipe}" ]]; then
    echo "ERROR: recipe not found: ${REPO_DIR}/${recipe}" >&2
    exit 2
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

  local overrides=(
    "grpo.max_num_steps=${MAX_STEPS}"
    "grpo.num_prompts_per_step=64"
    "grpo.num_generations_per_prompt=32"
    "checkpointing.enabled=false"
    "checkpointing.checkpoint_dir=${run_dir}/checkpoints"
    "policy.train_global_batch_size=512"
    "policy.max_total_sequence_length=4096"
    "policy.generation.max_new_tokens=4096"
    "policy.generation._output_max_model_len=4096"
    "policy.generation.vllm_cfg.max_model_len=4128"
    "policy.generation.vllm_cfg.tensor_parallel_size=${target_tp}"
    "policy.generation.vllm_cfg.enforce_eager=false"
    "++policy.generation.vllm_kwargs.max_num_batched_tokens=16384"
    "++policy.generation.vllm_kwargs.max_num_seqs=1024"
    "++policy.generation.vllm_kwargs.moe_backend=triton"
    "++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=${graph_mode}"
    "cluster.gpus_per_node=4"
    "cluster.num_nodes=4"
    "cluster.segment_size=4"
    "logger.wandb_enabled=true"
    "logger.tensorboard_enabled=false"
    "logger.wandb.project=${WANDB_PROJECT}"
    "logger.wandb.name=${wandb_run_id}"
    "++logger.wandb.entity=${WANDB_ENTITY}"
    "logger.log_dir=${run_dir}/nemo_logs"
  )

  if [[ "${draft_k}" != "0" ]]; then
    overrides+=(
      "++policy.generation.vllm_kwargs.speculative_config.method=eagle3"
      "++policy.generation.vllm_kwargs.speculative_config.model=${draft_model}"
      "++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=${draft_k}"
      "++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=${draft_tp}"
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
    "WANDB_RUN_ID=${wandb_run_id}"
    "WANDB_RUN_GROUP=${RUN_TAG}"
    "WANDB_RESUME=never"
    "NEMO_RL_VENV_DIR=${venv_dir}"
    "NRL_FORCE_REBUILD_VENVS=true"
    "PYTHONPATH=${REPO_DIR}"
    "TRITON_CACHE_DIR=${triton_cache_dir}"
    "TORCHINDUCTOR_CACHE_DIR=${inductor_cache_dir}"
    /opt/nemo_rl_venv/bin/python
    examples/run_grpo.py
    --config "${recipe}"
    "${overrides[@]}"
  )
  local command
  printf -v command '%q ' "${command_parts[@]}"
  command="${command% }"

  local environment=(
    "CONTAINER=${CONTAINER}"
    "MOUNTS=/lustre:/lustre"
    "CONTAINER_WORKDIR=${REPO_DIR}"
    "COMMAND=${command}"
    "BASE_LOG_DIR=${run_dir}"
    "GPUS_PER_NODE=4"
    "HF_HOME=${HF_HOME}"
    "PYTHONPATH=${REPO_DIR}"
    "PYTHONDONTWRITEBYTECODE=1"
    "RAY_LOG_SYNC_FREQUENCY=60"
    "TMPDIR=${TMPDIR}"
    "TRITON_CACHE_DIR=${triton_cache_dir}"
    "TORCHINDUCTOR_CACHE_DIR=${inductor_cache_dir}"
  )
  local sbatch_args=(
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --nodes=4
    --ntasks-per-node=1
    --exclusive
    --time="${WALLTIME}"
    --segment=4
    --job-name="${ACCOUNT}-nemorl.tail-gate-${model}-${variant}"
    --output="${run_dir}/slurm-%j.out"
    --comment=metrics
  )

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
      env "${environment[@]}" sbatch --test-only "${sbatch_args[@]}" "${REPO_DIR}/ray.sub"
      ;;
    submit)
      mkdir -p "${run_dir}"
      local manifest="${EXPERIMENT_ROOT}/submissions.tsv"
      local manifest_header=$'timestamp\tmodel\tvariant\trunner\tgraph_mode\tgate_mode\tk\tthreshold\tconsecutive_checks\troofline_config_sha256\tcommit\tcontainer\tcontainer_sha256\trecipe\tjob_id\twandb_run_id\twandb_url\tcommand'
      if [[ -f "${manifest}" && "$(head -n 1 "${manifest}")" != "${manifest_header}" ]]; then
        echo "ERROR: submissions manifest header mismatch: ${manifest}" >&2
        exit 2
      fi
      if [[ ! -f "${manifest}" ]]; then
        printf '%s\n' "${manifest_header}" >"${manifest}"
      fi
      env "${environment[@]}" sbatch --test-only "${sbatch_args[@]}" "${REPO_DIR}/ray.sub"
      local job_id
      job_id="$(env "${environment[@]}" sbatch --parsable "${sbatch_args[@]}" "${REPO_DIR}/ray.sub")"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date --iso-8601=seconds)" "${model}" "${variant}" "${runner}" \
        "${graph_mode}" "${gate_mode}" "${draft_k}" "${threshold}" \
        "${consecutive_checks}" "${roofline_hash}" "$(git -C "${REPO_DIR}" rev-parse HEAD)" \
        "$(readlink -f "${CONTAINER}")" "$(sha256_file "${CONTAINER}")" "${recipe}" \
        "${job_id}" "${wandb_run_id}" \
        "https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}/runs/${wandb_run_id}" "${command}" \
        >>"${manifest}"
      printf '%s\t%s\t%s\n' "${job_id}" "${model}" "${variant}"
      ;;
  esac
}

if [[ "${MODE}" != "dry-run" ]]; then
  mkdir -p "${EXPERIMENT_ROOT}"
fi
for model in "${models[@]}"; do
  for variant in "${variants[@]}"; do
    submit_one "${model}" "${variant}"
  done
done
