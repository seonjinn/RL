#!/bin/bash
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

fail() {
  echo "$*" >&2
  exit 2
}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd "${script_dir}/../../.." && pwd -P)
bridge_root=${repo_root}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
mcore_root=${bridge_root}/3rdparty/Megatron-LM
source_provenance_verifier=${script_dir}/scripts/verify_source_provenance.sh
runtime_attestation_validator=${script_dir}/verify_runtime_attestation.py
cd "${repo_root}"

: "${MODEL:?Set MODEL to nano, super, ultra, or qwen3_30ba3b}"
: "${SCOPE:?Set SCOPE through a persistent scope or variant launcher}"
: "${SCOPE_NAME:?Set SCOPE_NAME through a persistent launcher}"
: "${CLUSTER:?Set CLUSTER to ptyche, oci-hsg, or lyris}"

MODE=${MODE:-nemorl}
STEPS=${STEPS:-5}
TEST_ONLY=${TEST_ONLY:-0}
SBATCH_TEST_ONLY=${SBATCH_TEST_ONLY:-0}
RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_GROUP=${RUN_GROUP:-adhoc-${MODEL}-${MODE}-${CLUSTER}-${RUN_TAG}}
REPEAT_INDEX=${REPEAT_INDEX:-0}

[[ "${RUN_GROUP}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || \
  fail "RUN_GROUP must be filesystem-safe"
[[ "${REPEAT_INDEX}" =~ ^(0|[1-9][0-9]*)$ ]] || \
  fail "REPEAT_INDEX must be a non-negative integer"

case "${MODEL}" in
  nano|super|ultra|qwen3_30ba3b) ;;
  *) fail "MODEL must be nano, super, ultra, or qwen3_30ba3b" ;;
esac
case "${MODE}" in
  nemorl|mcore) ;;
  *) fail "MODE must be nemorl or mcore" ;;
esac
case "${CLUSTER}" in
  ptyche|oci-hsg|lyris) ;;
  *) fail "CLUSTER must be ptyche, oci-hsg, or lyris" ;;
esac
case "${STEPS}" in
  5) PHASE=smoke ;;
  20) PHASE=performance ;;
  100) PHASE=accuracy ;;
  *) fail "STEPS must be 5, 20, or 100" ;;
esac
[[ "${WARMUP_STEPS:-}" == "3" ]] || fail "WARMUP_STEPS must be exactly 3"
[[ "${THD_MAX_PACKED_SEQUENCES:-}" == "16" ]] || \
  fail "THD_MAX_PACKED_SEQUENCES must be 16"
[[ "${CHECKPOINTING_ENABLED:-}" == "false" ]] || \
  fail "CHECKPOINTING_ENABLED must be false"
[[ "${WANDB_PROJECT:-}" == "sna-cg-study" ]] || \
  fail "WANDB_PROJECT must be sna-cg-study"

model_file=${script_dir}/models/${MODEL}.env
[[ -f "${model_file}" ]] || fail "Missing model selector: ${model_file}"
# shellcheck source=/dev/null
source "${model_file}"
MODEL_NUM_NODES=${NUM_NODES}
MODEL_GPUS_PER_NODE=${GPUS_PER_NODE}

if [[ -n "${PROFILE_FILE:-}" ]]; then
  profile_file=${PROFILE_FILE}
elif [[ -f "${script_dir}/profiles/${CLUSTER}.env" ]]; then
  profile_file=${script_dir}/profiles/${CLUSTER}.env
else
  profile_file=${script_dir}/profiles/${CLUSTER}.env.example
fi
[[ -f "${profile_file}" ]] || fail "Missing cluster profile: ${profile_file}"
# shellcheck source=/dev/null
source "${profile_file}"
[[ "${PARTITION}" == "batch" ]] || fail "All production jobs require PARTITION=batch"
[[ "${SBATCH_GPUS_PER_NODE:-}" == "${MODEL_GPUS_PER_NODE}" ]] || \
  fail "Profile SBATCH_GPUS_PER_NODE must match model GPUS_PER_NODE=${MODEL_GPUS_PER_NODE}"
case "${SBATCH_GRES:-}" in
  none) ;;
  "gpu:${MODEL_GPUS_PER_NODE}") ;;
  *) fail "SBATCH_GRES must be none or gpu:${MODEL_GPUS_PER_NODE}" ;;
esac
if [[ -n "${SBATCH_SEGMENT_SIZE:-}" && \
      ! "${SBATCH_SEGMENT_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  fail "SBATCH_SEGMENT_SIZE must be empty or a positive integer"
fi

unresolved=()
for field in \
  ACCOUNT \
  SBATCH_GPUS_PER_NODE \
  SBATCH_GRES \
  CONTAINER \
  CONTAINER_SHA256 \
  MOUNTS \
  RUNTIME_ATTESTATION \
  RUNTIME_PREFLIGHT_JOB_ID \
  EXPECTED_TE_SHA \
  EXPECTED_NEMORL_SHA \
  EXPECTED_BRIDGE_SHA \
  EXPECTED_MCORE_SHA; do
  value=${!field:-}
  case "${value}" in
    ""|__REQUIRED_*__) unresolved+=("${field}") ;;
  esac
done
case "${RUNTIME_ATTESTATION:-}" in
  ""|__REQUIRED_*__) ;;
  /*) ;;
  *) fail "RUNTIME_ATTESTATION must be an absolute path" ;;
esac
case "${RUNTIME_PREFLIGHT_JOB_ID:-}" in
  ""|__REQUIRED_*__) ;;
  *[!0-9]*|0) fail "RUNTIME_PREFLIGHT_JOB_ID must be a positive SLURM job ID" ;;
esac
case "${EXPECTED_TE_SHA:-}" in
  ""|__REQUIRED_*__) ;;
  *)
    [[ "${EXPECTED_TE_SHA}" =~ ^[0-9a-f]{40}$ ]] || \
      fail "EXPECTED_TE_SHA must be a full lowercase SHA"
    ;;
esac

classifier_args=(
  classify
  --model "${MODEL}"
  --scope "${SCOPE}"
  --mode "${MODE}"
)
if [[ -n "${MCORE_DRIVER:-}" ]]; then
  classifier_args+=(--mcore-driver "${MCORE_DRIVER}")
fi
if [[ -e "${ULTRA_MODEL_PATH:-/__MISSING__}" && \
      -e "${ULTRA_DATA_PATH:-/__MISSING__}" && \
      -f "${ULTRA_JUDGE_CONFIG:-/__MISSING__}" && \
      -f "${ULTRA_LAUNCH_PROFILE:-/__MISSING__}" ]]; then
  classifier_args+=(--external-dependencies-ready)
fi
if [[ "${TEST_ONLY}" != "1" && ${#unresolved[@]} -gt 0 ]]; then
  classifier_args+=(--profile-blocked)
fi
classification=$(uv run --no-project "${script_dir}/scope_matrix.py" "${classifier_args[@]}")
IFS=$'\t' read -r status reason <<<"${classification}"
printf 'STATUS: %s\n' "${status}"
printf 'REASON: %s\n' "${reason}"
if [[ "${status}" != "runnable" ]]; then
  exit 0
fi

run_name=${SCOPE_NAME}-${MODEL}-${MODE}-${CLUSTER}-${STEPS}step-${RUN_TAG}
run_log_dir=${LOG_ROOT_OVERRIDE:-exp_logs/nemotron_thd_te_graph_20260731}/${run_name}

extra_overrides=()
case "${MOE_SHARED_EXPERT_OVERLAP:-}" in
  "") ;;
  true|false)
    extra_overrides+=(
      "policy.megatron_cfg.moe_shared_expert_overlap=${MOE_SHARED_EXPERT_OVERLAP}"
    )
    ;;
  *) fail "MOE_SHARED_EXPERT_OVERLAP must be true or false" ;;
esac
case "${MOE_ACT_RECOMPUTE:-false}" in
  false) ;;
  true)
    extra_overrides+=(
      policy.megatron_cfg.activation_checkpointing=true
      policy.megatron_cfg.recompute_granularity=selective
      policy.megatron_cfg.recompute_modules=[moe_act]
    )
    ;;
  *) fail "MOE_ACT_RECOMPUTE must be true or false" ;;
esac

if [[ "${MODE}" == "nemorl" ]]; then
  render_args=(
    render
    --model "${MODEL}"
    --scope "${SCOPE}"
    --steps "${STEPS}"
    --run-name "${run_name}"
    --log-dir "${run_log_dir}"
  )
  if ((${#extra_overrides[@]})); then
    for override in "${extra_overrides[@]}"; do
      render_args+=(--override "${override}")
    done
  fi
  COMMAND=$(uv run --no-project "${script_dir}/scope_matrix.py" "${render_args[@]}")
  job_script=${script_dir}/scripts/run_nemorl_scope.sub
else
  scope_modules=${SCOPE}
  [[ "${SCOPE}" == "whole_layer" ]] && scope_modules=""
  [[ "${SCOPE}" == "baseline" ]] && scope_modules="disabled"
  mcore_command=(
    "${MCORE_DRIVER}"
    --recipe "${MCORE_RECIPE}"
    --cuda-graph-modules "${scope_modules}"
    --train-iters "${STEPS}"
    --disable-checkpointing
  )
  printf -v COMMAND '%q ' "${mcore_command[@]}"
  COMMAND=${COMMAND% }
  job_script=${script_dir}/scripts/run_mcore_scope.sub
fi

runtime_attestation_command=(
  /usr/bin/python3
  "${runtime_attestation_validator}"
  --attestation "${RUNTIME_ATTESTATION}"
  --container "${CONTAINER}"
  --expected-container-sha256 "${CONTAINER_SHA256}"
  --nemo-rl-commit "${EXPECTED_NEMORL_SHA}"
  --bridge-commit "${EXPECTED_BRIDGE_SHA}"
  --mcore-commit "${EXPECTED_MCORE_SHA}"
  --uv-lock "${repo_root}/uv.lock"
  --expected-te-commit "${EXPECTED_TE_SHA}"
  --expected-device-count "${MODEL_GPUS_PER_NODE}"
)
printf -v RUNTIME_ATTESTATION_COMMAND '%q ' "${runtime_attestation_command[@]}"
RUNTIME_ATTESTATION_COMMAND=${RUNTIME_ATTESTATION_COMMAND% }

sbatch_command=(
  sbatch
  --parsable
  "--chdir=${repo_root}"
  "--nodes=${MODEL_NUM_NODES}"
  "--account=${ACCOUNT}"
  "--partition=${PARTITION}"
  "--time=${TIME_LIMIT}"
  "--job-name=cg-${run_name}"
  "--output=${run_log_dir}/slurm-%j.log"
  --export=ALL
)
case "${RUNTIME_PREFLIGHT_JOB_ID:-}" in
  ""|__REQUIRED_*__) ;;
  *) sbatch_command+=("--dependency=afterok:${RUNTIME_PREFLIGHT_JOB_ID}") ;;
esac
if [[ "${SBATCH_GRES}" != "none" ]]; then
  sbatch_command+=("--gres=${SBATCH_GRES}")
fi
if [[ -n "${SBATCH_SEGMENT_SIZE:-}" ]]; then
  sbatch_command+=("--segment=${SBATCH_SEGMENT_SIZE}")
fi
sbatch_command+=("${job_script}")
if [[ "${SBATCH_TEST_ONLY}" == "1" ]]; then
  sbatch_command=(sbatch --parsable --test-only "${sbatch_command[@]:2}")
fi

printf 'MODEL: %s\n' "${MODEL}"
printf 'DISPATCHER: %s\n' "${DISPATCHER}"
printf 'SCOPE: %s\n' "${SCOPE}"
printf 'STEPS: %s\n' "${STEPS}"
printf 'PHASE: %s\n' "${PHASE}"
printf 'RUN_GROUP: %s\n' "${RUN_GROUP}"
printf 'REPEAT_INDEX: %s\n' "${REPEAT_INDEX}"
printf 'PROFILE: %s\n' "${PROFILE_ID}"
printf 'RUN_LOG_DIR: %s\n' "${run_log_dir}"
printf 'RUNTIME_ATTESTATION: %q\n' "${RUNTIME_ATTESTATION_COMMAND}"
printf 'COMMAND: %q\n' "${COMMAND}"
printf 'SBATCH:'
printf ' %q' "${sbatch_command[@]}"
printf '\n'

if [[ "${TEST_ONLY}" == "1" ]]; then
  echo "TEST_ONLY: no submission performed"
  exit 0
fi
if ((${#unresolved[@]})); then
  fail "Refusing submission with unresolved profile fields: ${unresolved[*]}"
fi
[[ -f "${CONTAINER}" ]] || fail "Immutable container is missing: ${CONTAINER}"
[[ ! -L "${CONTAINER}" ]] || fail "Immutable container path must not be a symlink"
[[ -x "${source_provenance_verifier}" ]] || \
  fail "Source provenance verifier is missing or not executable"
[[ -f "${runtime_attestation_validator}" ]] || \
  fail "Runtime attestation validator is missing"
"${source_provenance_verifier}" \
  "${repo_root}" "${EXPECTED_NEMORL_SHA}" \
  "${bridge_root}" "${EXPECTED_BRIDGE_SHA}" \
  "${mcore_root}" "${EXPECTED_MCORE_SHA}"

mkdir -p "${run_log_dir}"
{
  printf 'model=%s\n' "${MODEL}"
  printf 'dispatcher=%s\n' "${DISPATCHER}"
  printf 'scope=%s\n' "${SCOPE}"
  printf 'scope_name=%s\n' "${SCOPE_NAME}"
  printf 'mode=%s\n' "${MODE}"
  printf 'cluster=%s\n' "${CLUSTER}"
  printf 'profile=%s\n' "${PROFILE_ID}"
  printf 'phase=%s\n' "${PHASE}"
  printf 'steps=%s\n' "${STEPS}"
  printf 'run_group=%s\n' "${RUN_GROUP}"
  printf 'repeat=%s\n' "${REPEAT_INDEX}"
  printf 'nemo_rl_commit=%s\n' "${EXPECTED_NEMORL_SHA}"
  printf 'bridge_commit=%s\n' "${EXPECTED_BRIDGE_SHA}"
  printf 'mcore_commit=%s\n' "${EXPECTED_MCORE_SHA}"
  printf 'transformer_engine_commit=%s\n' "${EXPECTED_TE_SHA}"
  printf 'container_sha256=%s\n' "${CONTAINER_SHA256}"
  printf 'runtime_preflight_job_id=%s\n' "${RUNTIME_PREFLIGHT_JOB_ID}"
  printf 'runtime_attestation=%s\n' "${RUNTIME_ATTESTATION}"
} >"${run_log_dir}/run-metadata.env"
export COMMAND CONTAINER CONTAINER_SHA256 MOUNTS RUNTIME_ATTESTATION_COMMAND
export BASE_LOG_DIR=${run_log_dir}
export GPUS_PER_NODE=${MODEL_GPUS_PER_NODE}
export NRL_FORCE_REBUILD_VENVS=true
export REPO_ROOT=${repo_root}
export MODEL DISPATCHER SCOPE SCOPE_NAME MODE CLUSTER PROFILE_ID PHASE STEPS
export RUN_GROUP REPEAT_INDEX
export EXPECTED_NEMORL_SHA EXPECTED_BRIDGE_SHA EXPECTED_MCORE_SHA
export EXPECTED_TE_SHA RUNTIME_ATTESTATION RUNTIME_PREFLIGHT_JOB_ID
export SOURCE_PROVENANCE_VERIFIER=${source_provenance_verifier}
job_id=$("${sbatch_command[@]}")
printf 'SLURM_JOB_ID: %s\n' "${job_id}"
