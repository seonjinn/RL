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
dockerfile=${repo_root}/docker/Dockerfile
bridge_root=${repo_root}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
mcore_root=${bridge_root}/3rdparty/Megatron-LM
source_provenance_verifier=${script_dir}/scripts/verify_source_provenance.sh
runtime_attestation_validator=${script_dir}/verify_runtime_attestation.py
profile_snapshot_helper=${script_dir}/profile_snapshot.py
campaign_gate_validator=${script_dir}/validate_campaign_gate.py
cd "${repo_root}"

: "${MODEL:?Set MODEL to nano, super, ultra, qwen3_30ba3b, or qwen3_235b}"
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
ROUTER_REPLAY=${ROUTER_REPLAY:-off}
NVTE_WITH_NCCL_EP=0

[[ "${RUN_GROUP}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || \
  fail "RUN_GROUP must be filesystem-safe"
[[ "${REPEAT_INDEX}" =~ ^(0|[1-9][0-9]*)$ ]] || \
  fail "REPEAT_INDEX must be a non-negative integer"

case "${MODEL}" in
  nano|super|ultra|qwen3_30ba3b|qwen3_235b) ;;
  *) fail "MODEL must be nano, super, ultra, qwen3_30ba3b, or qwen3_235b" ;;
esac
case "${ROUTER_REPLAY}" in
  off) R3_NAME=r3off ;;
  on) R3_NAME=r3on ;;
  *) fail "ROUTER_REPLAY must be off or on" ;;
esac
if [[ "${ROUTER_REPLAY}" == "on" ]]; then
  case ",${SCOPE}," in
    *,moe_router,*|*,moe_preprocess,*)
      fail "Router Replay cannot be combined with router CUDA Graph scopes"
      ;;
  esac
fi
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

profile_dir=${script_dir}/profiles
if [[ -n "${PROFILE_FILE:-}" ]]; then
  profile_file=${PROFILE_FILE}
elif [[ -e "${profile_dir}/${CLUSTER}.env" || -L "${profile_dir}/${CLUSTER}.env" ]]; then
  profile_file=${profile_dir}/${CLUSTER}.env
else
  profile_file=${profile_dir}/${CLUSTER}.env.example
fi
[[ -f "${profile_snapshot_helper}" ]] || fail "Missing profile snapshot helper"
profile_snapshot_command=(
  python3 "${profile_snapshot_helper}"
  --profile-dir "${profile_dir}"
  --cluster "${CLUSTER}"
  --profile-file "${profile_file}"
)
if [[ -n "${VALIDATED_PROFILE_SHA256:-}" ]]; then
  profile_snapshot_command+=(--expected-sha256 "${VALIDATED_PROFILE_SHA256}")
fi
profile_snapshot_output=$("${profile_snapshot_command[@]}") || fail "Cluster profile rejected"
PROFILE_SHA256=
while IFS=$'\t' read -r field value; do
  case "${field}" in
    PROFILE_SHA256|PROFILE_ID|ACCOUNT|PARTITION|CONTAINER|CONTAINER_SHA256|MOUNTS|SBATCH_GPUS_PER_NODE|SBATCH_GRES|SBATCH_SEGMENT_SIZE|TIME_LIMIT|RUNTIME_ATTESTATION|RUNTIME_PREFLIGHT_JOB_ID|EXPECTED_TE_SHA|EXPECTED_NEMORL_SHA|EXPECTED_BRIDGE_SHA|EXPECTED_MCORE_SHA)
      printf -v "${field}" '%s' "${value}"
      ;;
    *) fail "Cluster profile snapshot returned an unknown field" ;;
  esac
done <<<"${profile_snapshot_output}"
[[ "${PROFILE_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "Cluster profile snapshot omitted its digest"
if [[ ( "${MODEL}" == qwen3_30ba3b || "${MODEL}" == qwen3_235b ) && ( "${STEPS}" == 20 || ( "${MODEL}" == qwen3_235b && "${ROUTER_REPLAY}" == on ) ) ]]; then
  case "${QWEN_CAMPAIGN_ARM:-}" in
    A) [[ "${SCOPE}" == baseline && "${ROUTER_REPLAY}" == off ]] || fail "QWEN_CAMPAIGN_ARM A mismatch" ;;
    B) [[ "${SCOPE}" == moe_router && "${ROUTER_REPLAY}" == off ]] || fail "QWEN_CAMPAIGN_ARM B mismatch" ;;
    C) [[ "${SCOPE}" == baseline && "${ROUTER_REPLAY}" == on ]] || fail "QWEN_CAMPAIGN_ARM C mismatch" ;;
    E) [[ "${SCOPE}" == attn && "${ROUTER_REPLAY}" == on ]] || fail "QWEN_CAMPAIGN_ARM E mismatch" ;;
    *) fail "Qwen campaign launch requires QWEN_CAMPAIGN_ARM" ;;
  esac
  [[ "${VALIDATED_PROFILE_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || fail "Qwen campaign launch requires a validated profile digest"
  [[ -f "${campaign_gate_validator}" ]] || fail "Missing campaign gate validator"
  gate_common=(--model "${MODEL}" --profile-file "${profile_file}" --profile-dir "${profile_dir}" --cluster "${CLUSTER}" --expected-profile-sha256 "${PROFILE_SHA256}")
  if [[ "${MODEL}" == qwen3_235b && "${ROUTER_REPLAY}" == on ]]; then
    [[ -n "${R3_PREFLIGHT_FILE:-}" && -n "${R3_PREFLIGHT_SHA256:-}" ]] || fail "Qwen235 Router Replay requires R3 preflight evidence"
    gate_output=$(python3 "${campaign_gate_validator}" r3 "${gate_common[@]}" --gate-file "${R3_PREFLIGHT_FILE}" --gate-sha256 "${R3_PREFLIGHT_SHA256}") || fail "Qwen235 R3 campaign gate validation failed"
    [[ "${gate_output}" == "PROFILE_SHA256=${PROFILE_SHA256}" ]] || fail "Qwen235 R3 gate profile digest mismatch"
  fi
  if [[ "${STEPS}" == 20 ]]; then
    [[ -n "${SMOKE_PROMOTION_FILE:-}" && -n "${SMOKE_PROMOTION_SHA256:-}" ]] || fail "Qwen performance requires smoke promotion evidence"
    gate_output=$(python3 "${campaign_gate_validator}" promotion "${gate_common[@]}" --gate-file "${SMOKE_PROMOTION_FILE}" --gate-sha256 "${SMOKE_PROMOTION_SHA256}" --arm "${QWEN_CAMPAIGN_ARM}") || fail "Qwen promotion campaign gate validation failed"
    [[ "${gate_output}" == "PROFILE_SHA256=${PROFILE_SHA256}" ]] || fail "Qwen promotion gate profile digest mismatch"
  fi
fi
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
[[ -f "${repo_root}/.python-version" ]] || \
  fail "NeMo-RL source snapshot is missing .python-version"
[[ -f "${dockerfile}" ]] || fail "NeMo-RL source snapshot is missing docker/Dockerfile"
MANAGED_PYTHON_VERSION=$(tr -d '[:space:]' <"${repo_root}/.python-version")
[[ "${MANAGED_PYTHON_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || \
  fail ".python-version must contain one exact X.Y.Z version"
PINNED_UV_VERSION=$(sed -nE 's/^ARG UV_VERSION=([0-9]+\.[0-9]+\.[0-9]+)$/\1/p' "${dockerfile}")
[[ "${PINNED_UV_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || \
  fail "docker/Dockerfile must contain one exact ARG UV_VERSION=X.Y.Z pin"
case "${RUNTIME_ATTESTATION:-}" in
  /*)
    MANAGED_PYTHON_INSTALL_DIR=$(dirname "${RUNTIME_ATTESTATION}")/uv-python-installations
    ;;
  *)
    MANAGED_PYTHON_INSTALL_DIR=__DERIVED_FROM_RUNTIME_ATTESTATION__/uv-python-installations
    ;;
esac
case "${RUNTIME_ATTESTATION:-}:${RUNTIME_PREFLIGHT_JOB_ID:-}" in
  /*:[1-9]*)
    UV_EXECUTABLE=$(dirname "${RUNTIME_ATTESTATION}")/uv-${PINNED_UV_VERSION}-${RUNTIME_PREFLIGHT_JOB_ID}/uv
    ;;
  *)
    UV_EXECUTABLE=__DERIVED_FROM_RUNTIME_ATTESTATION__/uv-${PINNED_UV_VERSION}-__PREFLIGHT_JOB_ID__/uv
    ;;
esac
if [[ "${MANAGED_PYTHON_INSTALL_DIR}" == /* ]]; then
  managed_python_is_mounted=false
  IFS=',' read -r -a mount_specs <<<"${MOUNTS}"
  for mount_spec in "${mount_specs[@]}"; do
    IFS=':' read -r _ container_mount_path _ <<<"${mount_spec}"
    [[ "${container_mount_path:-}" == /* ]] || continue
    case "${MANAGED_PYTHON_INSTALL_DIR}" in
      "${container_mount_path}"|"${container_mount_path}"/*)
        managed_python_is_mounted=true
        break
        ;;
    esac
  done
  [[ "${managed_python_is_mounted}" == "true" ]] || \
    fail "managed Python install directory is not container-mounted: ${MANAGED_PYTHON_INSTALL_DIR}"
fi
if [[ "${UV_EXECUTABLE}" == /* ]]; then
  uv_is_mounted=false
  IFS=',' read -r -a mount_specs <<<"${MOUNTS}"
  for mount_spec in "${mount_specs[@]}"; do
    IFS=':' read -r _ container_mount_path _ <<<"${mount_spec}"
    [[ "${container_mount_path:-}" == /* ]] || continue
    case "${UV_EXECUTABLE}" in
      "${container_mount_path}"|"${container_mount_path}"/*)
        uv_is_mounted=true
        break
        ;;
    esac
  done
  [[ "${uv_is_mounted}" == "true" ]] || \
    fail "pinned uv executable is not container-mounted: ${UV_EXECUTABLE}"
fi
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
classification=$(python3 "${script_dir}/scope_matrix.py" "${classifier_args[@]}")
IFS=$'\t' read -r status reason <<<"${classification}"
printf 'STATUS: %s\n' "${status}"
printf 'REASON: %s\n' "${reason}"
if [[ "${status}" != "runnable" ]]; then
  exit 0
fi

run_name=${SCOPE_NAME}-${MODEL}-${MODE}-${CLUSTER}-${STEPS}step-${R3_NAME}-${RUN_TAG}
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
    --router-replay "${ROUTER_REPLAY}"
  )
  if ((${#extra_overrides[@]})); then
    for override in "${extra_overrides[@]}"; do
      render_args+=(--override "${override}")
    done
  fi
  COMMAND=$(python3 "${script_dir}/scope_matrix.py" "${render_args[@]}")
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
  /opt/nemo_rl_venv/bin/python
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
  --expected-python-version "${MANAGED_PYTHON_VERSION}"
  --expected-python-install-dir "${MANAGED_PYTHON_INSTALL_DIR}"
  --expected-uv-version "${PINNED_UV_VERSION}"
  --expected-uv-executable "${UV_EXECUTABLE}"
  --expected-nvte-with-nccl-ep "${NVTE_WITH_NCCL_EP}"
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
printf 'MANAGED_PYTHON_VERSION: %s\n' "${MANAGED_PYTHON_VERSION}"
printf 'MANAGED_PYTHON_INSTALL_DIR: %s\n' "${MANAGED_PYTHON_INSTALL_DIR}"
printf 'PINNED_UV_VERSION: %s\n' "${PINNED_UV_VERSION}"
printf 'UV_EXECUTABLE: %s\n' "${UV_EXECUTABLE}"
printf 'NVTE_WITH_NCCL_EP: %s\n' "${NVTE_WITH_NCCL_EP}"
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
  printf 'router_replay=%s\n' "${ROUTER_REPLAY}"
  printf 'nemo_rl_commit=%s\n' "${EXPECTED_NEMORL_SHA}"
  printf 'bridge_commit=%s\n' "${EXPECTED_BRIDGE_SHA}"
  printf 'mcore_commit=%s\n' "${EXPECTED_MCORE_SHA}"
  printf 'transformer_engine_commit=%s\n' "${EXPECTED_TE_SHA}"
  printf 'container_sha256=%s\n' "${CONTAINER_SHA256}"
  printf 'runtime_preflight_job_id=%s\n' "${RUNTIME_PREFLIGHT_JOB_ID}"
  printf 'runtime_attestation=%s\n' "${RUNTIME_ATTESTATION}"
  printf 'managed_python_version=%s\n' "${MANAGED_PYTHON_VERSION}"
  printf 'managed_python_install_dir=%s\n' "${MANAGED_PYTHON_INSTALL_DIR}"
  printf 'pinned_uv_version=%s\n' "${PINNED_UV_VERSION}"
  printf 'uv_executable=%s\n' "${UV_EXECUTABLE}"
  printf 'nvte_with_nccl_ep=%s\n' "${NVTE_WITH_NCCL_EP}"
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
export UV_PYTHON=${MANAGED_PYTHON_VERSION}
export UV_PYTHON_INSTALL_DIR=${MANAGED_PYTHON_INSTALL_DIR}
export UV_MANAGED_PYTHON=1
export UV_PYTHON_DOWNLOADS=never
export PINNED_UV_VERSION UV_EXECUTABLE
export NVTE_WITH_NCCL_EP
export SOURCE_PROVENANCE_VERIFIER=${source_provenance_verifier}
job_id=$("${sbatch_command[@]}")
printf 'SLURM_JOB_ID: %s\n' "${job_id}"
