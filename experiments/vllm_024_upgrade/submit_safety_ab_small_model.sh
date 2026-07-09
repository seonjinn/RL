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
COHORT_SELECTION="${2:-all}"
RECIPE_SELECTION="${3:-sync}"
STEP_SELECTION="${4:-10}"
VALIDATION_MODE="${VALIDATION_MODE:-recipe}"
REPLICATES="${REPLICATES:-1}"

if [[ "${DRY_RUN:-false}" == "true" || "${DRY_RUN:-false}" == "1" ]]; then
  MODE="dry-run"
fi

if [[ -z "${SOURCE_REPO_DIR:-}" ]]; then
  logical_pwd="$(pwd -L)"
  repo_prefix="$(git rev-parse --show-prefix)"
  if [[ -n "${repo_prefix}" ]]; then
    SOURCE_REPO_DIR="${logical_pwd%/${repo_prefix%/}}"
  else
    SOURCE_REPO_DIR="${logical_pwd}"
  fi
fi

ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
PARTITION="${PARTITION:-gb200}"
LYRIS_ROOT="${LYRIS_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}"
CONTAINER="${CONTAINER:-${LYRIS_ROOT}/containers/nemo_rl_nightly_20260707.sqsh}"
HF_HOME="${HF_HOME:-${LYRIS_ROOT}/hf_home}"
WANDB_PROJECT="${WANDB_PROJECT:-nemorl-vllm024-safety-ab-small-model}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-${LYRIS_ROOT}/.secrets/wandb_api_key}"
WANDB_NETRC_HOME="${WANDB_NETRC_HOME:-}"
RUN_TAG="${RUN_TAG:-vllm024-safety-ab-small-model}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${LYRIS_ROOT}/experiments/vllm024-safety-ab-small-model/${RUN_TAG}}"
CHECKOUT_ROOT="${CHECKOUT_ROOT:-${LYRIS_ROOT}/checkouts/nemorl-vllm024-safety-ab}"
WALLTIME="${WALLTIME:-04:00:00}"
TMPDIR="${TMPDIR_OVERRIDE:-/tmp}"
CONTAINER_SHA256="${CONTAINER_SHA256:-}"
EXPECTED_VLLM_VERSION_PREFIX="${EXPECTED_VLLM_VERSION_PREFIX:-0.24.}"
sync_recipe_blob_sha=""
async_recipe_blob_sha=""

if [[ "${MODE}" != "dry-run" && ! -f "${CONTAINER}" ]]; then
  echo "ERROR: container not found: ${CONTAINER}" >&2
  exit 2
fi

if [[ "${MODE}" == "submit" ]]; then
  if command -v sha256sum >/dev/null 2>&1; then
    computed_container_sha256="$(sha256sum "${CONTAINER}" | awk '{print $1}')"
  else
    computed_container_sha256="$(shasum -a 256 "${CONTAINER}" | awk '{print $1}')"
  fi
  if [[ -n "${CONTAINER_SHA256}" && "${CONTAINER_SHA256}" != "${computed_container_sha256}" ]]; then
    echo "ERROR: container SHA256 mismatch for ${CONTAINER}" >&2
    exit 2
  fi
  CONTAINER_SHA256="${computed_container_sha256}"
fi

if [[ "${MODE}" == "submit" && -z "${WANDB_API_KEY:-}" ]]; then
  if [[ -r "${WANDB_API_KEY_FILE}" ]]; then
    WANDB_API_KEY="$(<"${WANDB_API_KEY_FILE}")"
  elif [[ -n "${WANDB_NETRC_HOME}" && -r "${WANDB_NETRC_HOME}/.netrc" ]]; then
    WANDB_API_KEY="$(awk '{for (i = 1; i <= NF; i++) if ($i == "password") {print $(i + 1); exit}}' "${WANDB_NETRC_HOME}/.netrc")"
  else
    echo "ERROR: set WANDB_API_KEY or create ${WANDB_API_KEY_FILE}" >&2
    exit 2
  fi
  export WANDB_API_KEY
fi

case "${MODE}" in
  dry-run|test-only|submit) ;;
  *)
    echo "ERROR: mode must be dry-run, test-only, or submit" >&2
    exit 2
    ;;
esac

case "${COHORT_SELECTION}" in
  all) cohorts=(control candidate) ;;
  control|candidate) cohorts=("${COHORT_SELECTION}") ;;
  *)
    echo "ERROR: cohort must be all, control, or candidate" >&2
    exit 2
    ;;
esac

case "${RECIPE_SELECTION}" in
  all) recipe_modes=(sync async-1off) ;;
  sync|async-1off) recipe_modes=("${RECIPE_SELECTION}") ;;
  *)
    echo "ERROR: recipe must be all, sync, or async-1off" >&2
    exit 2
    ;;
esac

case "${STEP_SELECTION}" in
  all) step_counts=(10 20 40) ;;
  10|20|40) step_counts=("${STEP_SELECTION}") ;;
  *)
    echo "ERROR: steps must be all, 10, 20, or 40" >&2
    exit 2
    ;;
esac

case "${VALIDATION_MODE}" in
  recipe|off) ;;
  *)
    echo "ERROR: VALIDATION_MODE must be recipe or off" >&2
    exit 2
    ;;
esac

if [[ ! "${REPLICATES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: REPLICATES must be a positive integer" >&2
  exit 2
fi

submit_one() {
  local cohort="$1"
  local recipe_mode="$2"
  local max_steps="$3"
  local replicate="$4"
  local configured_repo
  local requested_commit
  local input_name
  if [[ "${cohort}" == "control" ]]; then
    configured_repo="${CONTROL_REPO_DIR:-}"
    requested_commit="${CONTROL_COMMIT:-}"
    input_name="CONTROL"
  else
    configured_repo="${CANDIDATE_REPO_DIR:-}"
    requested_commit="${CANDIDATE_COMMIT:-}"
    input_name="CANDIDATE"
  fi
  if [[ -n "${configured_repo}" && -n "${requested_commit}" ]]; then
    echo "ERROR: set only ${input_name}_REPO_DIR or ${input_name}_COMMIT" >&2
    exit 2
  fi
  if [[ -z "${configured_repo}" && -z "${requested_commit}" ]]; then
    echo "ERROR: set ${input_name}_REPO_DIR or ${input_name}_COMMIT" >&2
    exit 2
  fi
  if [[ "${MODE}" == "submit" && -n "${configured_repo}" ]]; then
    echo "ERROR: submit requires ${input_name}_COMMIT, not a mutable ${input_name}_REPO_DIR" >&2
    exit 2
  fi

  local repo_dir
  local commit
  if [[ -n "${configured_repo}" ]]; then
    repo_dir="${configured_repo}"
    commit="$(git -C "${repo_dir}" rev-parse HEAD)"
  else
    commit="$(git -C "${SOURCE_REPO_DIR}" rev-parse "${requested_commit}^{commit}")"
    repo_dir="${CHECKOUT_ROOT}/${cohort}-${commit:0:12}"
    if [[ "${MODE}" != "dry-run" ]]; then
      mkdir -p "${CHECKOUT_ROOT}"
      if [[ -e "${repo_dir}" ]]; then
        local checkout_commit
        checkout_commit="$(git -C "${repo_dir}" rev-parse HEAD)"
        if [[ "${checkout_commit}" != "${commit}" ]]; then
          echo "ERROR: ${repo_dir} is at ${checkout_commit}, expected ${commit}" >&2
          exit 2
        fi
      else
        git -C "${SOURCE_REPO_DIR}" worktree add --detach "${repo_dir}" "${commit}"
      fi
    fi
  fi
  if [[ "${MODE}" == "submit" ]]; then
    if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=all --ignore-submodules=dirty)" ]]; then
      echo "ERROR: submit requires a clean checkout for ${cohort}" >&2
      exit 2
    fi
    if ! git -C "${repo_dir}" branch -r --contains "${commit}" | grep -q .; then
      echo "ERROR: ${cohort} commit ${commit} is not on a known remote branch" >&2
      exit 2
    fi
  fi
  local short_commit="${commit:0:12}"
  local recipe_suffix=""
  if [[ "${recipe_mode}" == "async-1off" ]]; then
    recipe_suffix="-async-1off"
  fi
  local recipe="examples/configs/recipes/llm/performance/grpo-llama3.1-8b-instruct-2n4g${recipe_suffix}.yaml"
  if [[ "${MODE}" != "dry-run" ]]; then
    if [[ ! -f "${repo_dir}/${recipe}" ]]; then
      echo "ERROR: recipe not found: ${repo_dir}/${recipe}" >&2
      exit 2
    fi
    if [[ ! -f "${repo_dir}/ray.sub" ]]; then
      echo "ERROR: ray launcher not found: ${repo_dir}/ray.sub" >&2
      exit 2
    fi
  fi
  local recipe_blob_sha
  local blob_repo="${repo_dir}"
  if [[ ! -d "${blob_repo}" ]]; then
    blob_repo="${SOURCE_REPO_DIR}"
  fi
  if ! recipe_blob_sha="$(git -C "${blob_repo}" rev-parse "${commit}:${recipe}" 2>/dev/null)"; then
    echo "ERROR: recipe is not tracked at ${commit}: ${recipe}" >&2
    exit 2
  fi
  if [[ "${recipe_mode}" == "sync" ]]; then
    if [[ -z "${sync_recipe_blob_sha}" ]]; then
      sync_recipe_blob_sha="${recipe_blob_sha}"
    elif [[ "${sync_recipe_blob_sha}" != "${recipe_blob_sha}" ]]; then
      echo "ERROR: recipe blob mismatch for ${recipe_mode}: expected ${sync_recipe_blob_sha}, got ${recipe_blob_sha}" >&2
      exit 2
    fi
  else
    if [[ -z "${async_recipe_blob_sha}" ]]; then
      async_recipe_blob_sha="${recipe_blob_sha}"
    elif [[ "${async_recipe_blob_sha}" != "${recipe_blob_sha}" ]]; then
      echo "ERROR: recipe blob mismatch for ${recipe_mode}: expected ${async_recipe_blob_sha}, got ${recipe_blob_sha}" >&2
      exit 2
    fi
  fi
  local validation_tag="val-${VALIDATION_MODE}"
  local run_name="${RUN_TAG}-${cohort}-${recipe_mode}-step${max_steps}-${validation_tag}-r${replicate}-${short_commit}"
  local run_dir="${EXPERIMENT_ROOT}/${cohort}/${recipe_mode}/step${max_steps}-${validation_tag}-r${replicate}-${short_commit}"
  local cache_key="${cohort}-${recipe_mode}-step${max_steps}-${validation_tag}-r${replicate}-${short_commit}"
  local triton_cache_dir="/tmp/nemorl-safety-ab-triton-${cache_key}"
  local inductor_cache_dir="/tmp/nemorl-safety-ab-inductor-${cache_key}"
  local command_parts=(
    env
    "PYTHONPATH=${repo_dir}"
    "HF_HOME=${HF_HOME}"
    "TRITON_CACHE_DIR=${triton_cache_dir}"
    "TORCHINDUCTOR_CACHE_DIR=${inductor_cache_dir}"
    /opt/nemo_rl_venv/bin/python
    examples/run_grpo.py
    --config
    "${recipe}"
    "grpo.max_num_steps=${max_steps}"
    checkpointing.enabled=false
    "checkpointing.checkpoint_dir=${run_dir}/checkpoints"
    policy.generation.vllm_cfg.enforce_eager=false
    cluster.segment_size=2
    "logger.wandb.project=${WANDB_PROJECT}"
    "logger.wandb.name=${run_name}"
    "logger.log_dir=${run_dir}/nemo_logs"
  )
  if [[ "${VALIDATION_MODE}" == "off" ]]; then
    command_parts+=(
      grpo.val_period=0
      grpo.val_at_start=false
      grpo.val_at_end=false
    )
  fi
  local command
  local workload_command
  printf -v workload_command '%q ' "${command_parts[@]}"
  workload_command="${workload_command% }"
  local preflight_code
  preflight_code="import sys, vllm; print(f'NeMo-RL safety A/B runtime: python={sys.executable} vllm={vllm.__version__}'); assert vllm.__version__.startswith('${EXPECTED_VLLM_VERSION_PREFIX}'), f'expected vLLM ${EXPECTED_VLLM_VERSION_PREFIX}*, got {vllm.__version__}'"
  local preflight_parts=(
    /opt/nemo_rl_venv/bin/python
    -c
    "${preflight_code}"
  )
  local preflight_command
  printf -v preflight_command '%q ' "${preflight_parts[@]}"
  preflight_command="${preflight_command% }"
  command="${preflight_command} && exec ${workload_command}"

  local environment=(
    "CONTAINER=${CONTAINER}"
    "MOUNTS=/lustre:/lustre"
    "CONTAINER_WORKDIR=${repo_dir}"
    "COMMAND=${command}"
    "BASE_LOG_DIR=${run_dir}"
    "GPUS_PER_NODE=4"
    "HF_HOME=${HF_HOME}"
    "PYTHONPATH=${repo_dir}"
    "PYTHONDONTWRITEBYTECODE=1"
    "RAY_LOG_SYNC_FREQUENCY=60"
    "TMPDIR=${TMPDIR}"
    "TRITON_CACHE_DIR=${triton_cache_dir}"
    "TORCHINDUCTOR_CACHE_DIR=${inductor_cache_dir}"
  )
  local sbatch_args=(
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --nodes=2
    --ntasks-per-node=1
    --exclusive
    --time="${WALLTIME}"
    --segment=2
    --job-name="${ACCOUNT}-nemorl.safety-ab-${cohort}-${recipe_mode}-s${max_steps}-r${replicate}"
    --output="${run_dir}/slurm-%j.out"
    --comment=metrics
  )

  case "${MODE}" in
    dry-run)
      printf '[DRY-RUN] job %s env' "${run_name}"
      printf ' %q' "${environment[@]}"
      printf ' sbatch'
      printf ' %q' "${sbatch_args[@]}"
      printf ' %q\n' "${repo_dir}/ray.sub"
      printf '[DRY-RUN] command %s\n' "${command}"
      printf '[DRY-RUN] provenance %s commit=%s repo=%s\n' \
        "${cohort}" "${commit}" "${repo_dir}"
      ;;
    test-only)
      mkdir -p "${run_dir}"
      env "${environment[@]}" sbatch --test-only "${sbatch_args[@]}" \
        "${repo_dir}/ray.sub"
      ;;
    submit)
      mkdir -p "${run_dir}"
      local job_id
      job_id="$(env "${environment[@]}" sbatch --parsable "${sbatch_args[@]}" \
        "${repo_dir}/ray.sub")"
      local manifest="${EXPERIMENT_ROOT}/submissions.tsv"
      if [[ ! -f "${manifest}" ]]; then
        printf 'timestamp\tcohort\trecipe\tsteps\tvalidation\treplicate\tjob_id\tcommit\trepo_dir\trecipe_path\trecipe_blob_sha\tcontainer\tcontainer_sha256\texpected_vllm_version_prefix\twandb_project\twandb_name\trun_dir\tcommand\n' \
          >"${manifest}"
      fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${cohort}" "${recipe_mode}" \
        "${max_steps}" "${VALIDATION_MODE}" "${replicate}" "${job_id}" \
        "${commit}" "${repo_dir}" "${recipe}" \
        "${recipe_blob_sha}" "${CONTAINER}" "${CONTAINER_SHA256}" \
        "${EXPECTED_VLLM_VERSION_PREFIX}" "${WANDB_PROJECT}" \
        "${run_name}" "${run_dir}" "${command}" >>"${manifest}"
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${job_id}" "${cohort}" \
        "${recipe_mode}" "${max_steps}" "${VALIDATION_MODE}" "${replicate}"
      ;;
  esac
}

for cohort in "${cohorts[@]}"; do
  for recipe_mode in "${recipe_modes[@]}"; do
    for max_steps in "${step_counts[@]}"; do
      for ((replicate = 1; replicate <= REPLICATES; replicate++)); do
        submit_one "${cohort}" "${recipe_mode}" "${max_steps}" "${replicate}"
      done
    done
  done
done
