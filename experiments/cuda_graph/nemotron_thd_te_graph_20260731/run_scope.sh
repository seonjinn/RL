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

sha256_regular_file() {
  python3 - "$1" <<'PY'
import hashlib
import os
import stat
import sys

path = sys.argv[1]
flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(path, flags)
try:
    details = os.fstat(descriptor)
    if not stat.S_ISREG(details.st_mode):
        raise ValueError(f"not a regular file: {path}")
    digest = hashlib.sha256()
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
finally:
    os.close(descriptor)
print(digest.hexdigest())
PY
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

if [[ -n "${MCORE_CANDIDATE_SHA:-}" ]]; then
  [[ "${MCORE_CANDIDATE_SHA}" =~ ^[0-9a-f]{40}$ ]] || \
    fail "MCORE_CANDIDATE_SHA must be a full lowercase SHA"
  [[ "${RUN_LOG_ROOT:-}" == /* ]] || \
    fail "RUN_LOG_ROOT must be absolute when MCORE_CANDIDATE_SHA is set"
  git -C "${mcore_root}" cat-file -e "${MCORE_CANDIDATE_SHA}^{commit}" || \
    fail "MCORE_CANDIDATE_SHA is absent from the local object store"
  remote_matches=$(git -C "${mcore_root}" ls-remote origin | \
    awk -v sha="${MCORE_CANDIDATE_SHA}" '$1 == sha {count += 1} END {print count + 0}')
  [[ "${remote_matches}" -gt 0 ]] || \
    fail "MCORE_CANDIDATE_SHA is absent from the pushed remote"
  MCORE_CANDIDATE_SOURCE_ROOT=${RUN_LOG_ROOT}/source-snapshots/mcore/${MCORE_CANDIDATE_SHA}
  if [[ ! -d "${MCORE_CANDIDATE_SOURCE_ROOT}" ]]; then
    mkdir -p "$(dirname "${MCORE_CANDIDATE_SOURCE_ROOT}")"
    candidate_tmp=$(mktemp -d "$(dirname "${MCORE_CANDIDATE_SOURCE_ROOT}")/.${MCORE_CANDIDATE_SHA}.XXXXXX")
    git -C "${mcore_root}" archive "${MCORE_CANDIDATE_SHA}" | tar -x -C "${candidate_tmp}"
    printf '%s\n' "${MCORE_CANDIDATE_SHA}" >"${candidate_tmp}/.candidate-sha"
    mv "${candidate_tmp}" "${MCORE_CANDIDATE_SOURCE_ROOT}"
  fi
  export MCORE_CANDIDATE_SOURCE_ROOT
fi

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

case "${TEST_ONLY}" in
  0|1) ;;
  *) fail "TEST_ONLY must be 0 or 1" ;;
esac
case "${SBATCH_TEST_ONLY}" in
  0|1) ;;
  *) fail "SBATCH_TEST_ONLY must be 0 or 1" ;;
esac
if [[ "${TEST_ONLY}" == "1" && "${SBATCH_TEST_ONLY}" == "1" ]]; then
  fail "TEST_ONLY and SBATCH_TEST_ONLY are mutually exclusive"
fi

[[ "${RUN_GROUP}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || \
  fail "RUN_GROUP must be filesystem-safe"
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || \
  fail "RUN_TAG must be filesystem-safe"
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
  [[ "${VALIDATED_PROFILE_SHA256}" =~ ^[0-9a-f]{64}$ ]] || \
    fail "VALIDATED_PROFILE_SHA256 must be a full lowercase SHA256"
  profile_snapshot_command+=(--expected-sha256 "${VALIDATED_PROFILE_SHA256}")
fi
profile_snapshot_output=$("${profile_snapshot_command[@]}") || fail "Cluster profile rejected"
PROFILE_SHA256=
while IFS=$'\t' read -r field value; do
  case "${field}" in
    PROFILE_SHA256|PROFILE_ID|ACCOUNT|PARTITION|CONTAINER|CONTAINER_SHA256|MOUNTS|SBATCH_GPUS_PER_NODE|SBATCH_GRES|SBATCH_SEGMENT_SIZE|TIME_LIMIT|RUNTIME_ATTESTATION|RUNTIME_PREFLIGHT_JOB_ID|EXPECTED_TE_SHA|EXPECTED_TE_VERSION_BASE_SHA|EXPECTED_NEMORL_SHA|EXPECTED_BRIDGE_SHA|EXPECTED_MCORE_SHA|RUN_LOG_ROOT)
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
  EXPECTED_TE_VERSION_BASE_SHA \
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
case "${EXPECTED_TE_VERSION_BASE_SHA:-}" in
  ""|__REQUIRED_*__) ;;
  *)
    [[ "${EXPECTED_TE_VERSION_BASE_SHA}" =~ ^[0-9a-f]{40}$ ]] || \
      fail "EXPECTED_TE_VERSION_BASE_SHA must be a full lowercase SHA"
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
log_root=${LOG_ROOT_OVERRIDE:-exp_logs/nemotron_thd_te_graph_20260731}
if [[ "${log_root}" == /* ]]; then
  run_log_dir=${log_root}/${run_name}
else
  run_log_dir=${repo_root}/${log_root}/${run_name}
fi

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
  RENDERED_DRIVER_COMMAND=$(python3 "${script_dir}/scope_matrix.py" "${render_args[@]}")
  COMMAND=${RENDERED_DRIVER_COMMAND}
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
  printf -v RENDERED_DRIVER_COMMAND '%q ' "${mcore_command[@]}"
  RENDERED_DRIVER_COMMAND=${RENDERED_DRIVER_COMMAND% }
  COMMAND=${RENDERED_DRIVER_COMMAND}
  job_script=${script_dir}/scripts/run_mcore_scope.sub
fi

R3_DRIVER_COMMAND_FILE=
R3_DRIVER_COMMAND_SHA256=
R3_CHECKER_PATH=
R3_CHECKER_SHA256=
R3_RECORD_PYTHON=/opt/nemo_rl_venv/bin/python
R3_VALIDATION_RECORD_PATTERN=
R3_VALIDATION_RECORD_INITIAL_PATH=
if [[ "${MODE}" == "nemorl" && "${ROUTER_REPLAY}" == "on" ]]; then
  r3_wrapper=${script_dir}/scripts/run_r3_validated_command.sh
  [[ -x "${r3_wrapper}" ]] || fail "R3 validation wrapper is missing or not executable"
  R3_DRIVER_COMMAND_FILE=${run_log_dir}/r3-driver-command.sh
  R3_VALIDATION_RECORD_PATTERN=${run_log_dir}/r3-validation-job-{slurm_job_id}-restart-{slurm_restart_count}/r3-validation.json
  R3_CHECKER_PATH=${repo_root}/tools/check_r3_trace.py
  R3_DRIVER_COMMAND_SHA256=$(printf '%s' "${RENDERED_DRIVER_COMMAND}" | python3 -c 'import hashlib, sys; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())') || \
    fail "R3 driver command digest failed"
  R3_CHECKER_SHA256=$(sha256_regular_file "${R3_CHECKER_PATH}") || \
    fail "R3 checker is missing, unsafe, or unreadable"
  [[ "${R3_DRIVER_COMMAND_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "R3 driver command digest failed"
  [[ "${R3_CHECKER_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "R3 checker digest failed"
  wrapper_command=("${r3_wrapper}" "${R3_RECORD_PYTHON}" "${run_log_dir}" "${repo_root}" "${UV_EXECUTABLE}" "${R3_DRIVER_COMMAND_FILE}" "${R3_DRIVER_COMMAND_SHA256}" "${R3_CHECKER_SHA256}")
  printf -v COMMAND '%q ' "${wrapper_command[@]}"
  COMMAND=${COMMAND% }
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
  --expected-te-version-base-commit "${EXPECTED_TE_VERSION_BASE_SHA}"
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
RUNTIME_ATTESTATION_SHA256=$(sha256_regular_file "${RUNTIME_ATTESTATION}") || \
  fail "Runtime attestation is missing, unsafe, or unreadable"
[[ "${RUNTIME_ATTESTATION_SHA256}" =~ ^[0-9a-f]{64}$ ]] || \
  fail "Runtime attestation digest failed"

export COMMAND CONTAINER CONTAINER_SHA256 MOUNTS RUNTIME_ATTESTATION_COMMAND
export BASE_LOG_DIR=${run_log_dir}
export GPUS_PER_NODE=${MODEL_GPUS_PER_NODE}
export NRL_FORCE_REBUILD_VENVS=true
export REPO_ROOT=${repo_root}
export MODEL DISPATCHER SCOPE SCOPE_NAME MODE CLUSTER PROFILE_ID PHASE STEPS
export RUN_GROUP REPEAT_INDEX
export EXPECTED_NEMORL_SHA EXPECTED_BRIDGE_SHA EXPECTED_MCORE_SHA
export EXPECTED_TE_SHA EXPECTED_TE_VERSION_BASE_SHA
export RUNTIME_ATTESTATION RUNTIME_PREFLIGHT_JOB_ID
export UV_PYTHON=${MANAGED_PYTHON_VERSION}
export UV_PYTHON_INSTALL_DIR=${MANAGED_PYTHON_INSTALL_DIR}
export UV_MANAGED_PYTHON=1
export UV_PYTHON_DOWNLOADS=never
export PINNED_UV_VERSION UV_EXECUTABLE
export NVTE_WITH_NCCL_EP
export SOURCE_PROVENANCE_VERIFIER=${source_provenance_verifier}
export R3_DRIVER_COMMAND_FILE

if [[ "${SBATCH_TEST_ONLY}" == "1" ]]; then
  scheduler_test_output=$("${sbatch_command[@]}")
  printf 'SBATCH_TEST_ONLY_OUTPUT: %s\n' "${scheduler_test_output}"
  exit 0
fi

mkdir -p "${run_log_dir}"
if [[ -n "${R3_DRIVER_COMMAND_FILE}" ]]; then
  r3_command_tmp=$(mktemp "${run_log_dir}/.r3-driver-command.XXXXXX")
  printf '%s' "${RENDERED_DRIVER_COMMAND}" >"${r3_command_tmp}"
  mv -f "${r3_command_tmp}" "${R3_DRIVER_COMMAND_FILE}"
  written_r3_driver_sha256=$(sha256_regular_file "${R3_DRIVER_COMMAND_FILE}") || \
    fail "Written R3 driver command is unsafe or unreadable"
  [[ "${written_r3_driver_sha256}" == "${R3_DRIVER_COMMAND_SHA256}" ]] || \
    fail "Written R3 driver command digest mismatch"
fi

job_id=$("${sbatch_command[@]}")
[[ "${job_id}" =~ ^[1-9][0-9]*$ ]] || fail "sbatch --parsable returned an invalid job ID"
if [[ -n "${R3_VALIDATION_RECORD_PATTERN}" ]]; then
  R3_VALIDATION_RECORD_INITIAL_PATH=${run_log_dir}/r3-validation-job-${job_id}-restart-0/r3-validation.json
fi

sbatch_argv_json=$(python3 -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "${sbatch_command[@]}")
rendered_driver_command_base64=$(printf '%s' "${RENDERED_DRIVER_COMMAND}" | base64 | tr -d '\n')
effective_command_base64=$(printf '%s' "${COMMAND}" | base64 | tr -d '\n')
sbatch_argv_json_base64=$(printf '%s' "${sbatch_argv_json}" | base64 | tr -d '\n')
output_pattern_base64=$(printf '%s' "${run_log_dir}/slurm-%j.log" | base64 | tr -d '\n')
resolved_output_path_base64=$(printf '%s' "${run_log_dir}/slurm-${job_id}.log" | base64 | tr -d '\n')
run_log_dir_base64=$(printf '%s' "${run_log_dir}" | base64 | tr -d '\n')
container_path_base64=$(printf '%s' "${CONTAINER}" | base64 | tr -d '\n')
runtime_attestation_base64=$(printf '%s' "${RUNTIME_ATTESTATION}" | base64 | tr -d '\n')
managed_python_install_dir_base64=$(printf '%s' "${MANAGED_PYTHON_INSTALL_DIR}" | base64 | tr -d '\n')
uv_executable_base64=$(printf '%s' "${UV_EXECUTABLE}" | base64 | tr -d '\n')
r3_validation_record_pattern_base64=$(printf '%s' "${R3_VALIDATION_RECORD_PATTERN}" | base64 | tr -d '\n')
r3_validation_record_initial_path_base64=$(printf '%s' "${R3_VALIDATION_RECORD_INITIAL_PATH}" | base64 | tr -d '\n')
r3_driver_command_file_base64=$(printf '%s' "${R3_DRIVER_COMMAND_FILE}" | base64 | tr -d '\n')
r3_checker_path_base64=$(printf '%s' "${R3_CHECKER_PATH}" | base64 | tr -d '\n')
r3_record_python_base64=$(printf '%s' "${R3_RECORD_PYTHON}" | base64 | tr -d '\n')
metadata_env_tmp=
metadata_json_tmp=
cleanup_metadata_temps() {
  [[ -z "${metadata_env_tmp}" ]] || rm -f -- "${metadata_env_tmp}"
  [[ -z "${metadata_json_tmp}" ]] || rm -f -- "${metadata_json_tmp}"
}
trap cleanup_metadata_temps EXIT
metadata_env_tmp=$(mktemp "${run_log_dir}/.run-metadata.env.XXXXXX")
metadata_json_tmp=$(mktemp "${run_log_dir}/.run-metadata.json.XXXXXX")
{
  printf 'schema_version=1\n'
  printf 'job_id=%s\n' "${job_id}"
  printf 'rendered_driver_command_base64=%s\n' "${rendered_driver_command_base64}"
  printf 'effective_command_base64=%s\n' "${effective_command_base64}"
  printf 'sbatch_argv_json_base64=%s\n' "${sbatch_argv_json_base64}"
  printf 'output_pattern_base64=%s\n' "${output_pattern_base64}"
  printf 'resolved_output_path_base64=%s\n' "${resolved_output_path_base64}"
  printf 'run_log_dir_base64=%s\n' "${run_log_dir_base64}"
  printf 'container_path_base64=%s\n' "${container_path_base64}"
  printf 'model=%s\n' "${MODEL}"
  printf 'dispatcher=%s\n' "${DISPATCHER}"
  printf 'scope=%s\n' "${SCOPE}"
  printf 'scope_name=%s\n' "${SCOPE_NAME}"
  printf 'mode=%s\n' "${MODE}"
  printf 'cluster=%s\n' "${CLUSTER}"
  printf 'profile=%s\n' "${PROFILE_ID}"
  printf 'profile_sha256=%s\n' "${PROFILE_SHA256}"
  printf 'phase=%s\nsteps=%s\nrun_group=%s\nrepeat=%s\nrouter_replay=%s\n' "${PHASE}" "${STEPS}" "${RUN_GROUP}" "${REPEAT_INDEX}" "${ROUTER_REPLAY}"
  printf 'tensorboard_enabled=%s\n' "${NEMORL_TENSORBOARD_ENABLED}"
  printf 'num_nodes=%s\ngpus_per_node=%s\n' "${MODEL_NUM_NODES}" "${MODEL_GPUS_PER_NODE}"
  printf 'nemo_rl_commit=%s\nbridge_commit=%s\nmcore_commit=%s\ntransformer_engine_commit=%s\ncontainer_sha256=%s\n' "${EXPECTED_NEMORL_SHA}" "${EXPECTED_BRIDGE_SHA}" "${EXPECTED_MCORE_SHA}" "${EXPECTED_TE_SHA}" "${CONTAINER_SHA256}"
  printf 'runtime_preflight_job_id=%s\nruntime_attestation_base64=%s\nruntime_attestation_sha256=%s\nmanaged_python_version=%s\nmanaged_python_install_dir_base64=%s\npinned_uv_version=%s\nuv_executable_base64=%s\n' "${RUNTIME_PREFLIGHT_JOB_ID}" "${runtime_attestation_base64}" "${RUNTIME_ATTESTATION_SHA256}" "${MANAGED_PYTHON_VERSION}" "${managed_python_install_dir_base64}" "${PINNED_UV_VERSION}" "${uv_executable_base64}"
  printf 'r3_validation_record_pattern_base64=%s\nr3_validation_record_initial_path_base64=%s\nr3_driver_command_file_base64=%s\nr3_driver_command_sha256=%s\nr3_checker_path_base64=%s\nr3_checker_sha256=%s\nr3_record_python_base64=%s\n' "${r3_validation_record_pattern_base64}" "${r3_validation_record_initial_path_base64}" "${r3_driver_command_file_base64}" "${R3_DRIVER_COMMAND_SHA256}" "${r3_checker_path_base64}" "${R3_CHECKER_SHA256}" "${r3_record_python_base64}"
} >"${metadata_env_tmp}"
METADATA_JOB_ID=${job_id} METADATA_RENDERED_DRIVER=${RENDERED_DRIVER_COMMAND} METADATA_COMMAND=${COMMAND} METADATA_RUN_LOG_DIR=${run_log_dir} METADATA_SCHEDULER_ARGV_JSON=${sbatch_argv_json} METADATA_OUTPUT_PATTERN=${run_log_dir}/slurm-%j.log METADATA_OUTPUT_PATH=${run_log_dir}/slurm-${job_id}.log METADATA_R3_RECORD_PATTERN=${R3_VALIDATION_RECORD_PATTERN} METADATA_R3_RECORD_INITIAL=${R3_VALIDATION_RECORD_INITIAL_PATH} METADATA_R3_DRIVER_FILE=${R3_DRIVER_COMMAND_FILE} METADATA_R3_DRIVER_SHA=${R3_DRIVER_COMMAND_SHA256} METADATA_R3_CHECKER_PATH=${R3_CHECKER_PATH} METADATA_R3_CHECKER_SHA=${R3_CHECKER_SHA256} METADATA_R3_RECORD_PYTHON=${R3_RECORD_PYTHON} METADATA_MODEL=${MODEL} METADATA_DISPATCHER=${DISPATCHER} METADATA_SCOPE=${SCOPE} METADATA_SCOPE_NAME=${SCOPE_NAME} METADATA_MODE=${MODE} METADATA_CLUSTER=${CLUSTER} METADATA_PROFILE=${PROFILE_ID} METADATA_PROFILE_SHA256=${PROFILE_SHA256} METADATA_PHASE=${PHASE} METADATA_STEPS=${STEPS} METADATA_RUN_GROUP=${RUN_GROUP} METADATA_REPEAT=${REPEAT_INDEX} METADATA_ROUTER_REPLAY=${ROUTER_REPLAY} METADATA_TENSORBOARD=${NEMORL_TENSORBOARD_ENABLED} METADATA_NUM_NODES=${MODEL_NUM_NODES} METADATA_GPUS_PER_NODE=${MODEL_GPUS_PER_NODE} METADATA_NEMORL_SHA=${EXPECTED_NEMORL_SHA} METADATA_BRIDGE_SHA=${EXPECTED_BRIDGE_SHA} METADATA_MCORE_SHA=${EXPECTED_MCORE_SHA} METADATA_TE_SHA=${EXPECTED_TE_SHA} METADATA_CONTAINER_PATH=${CONTAINER} METADATA_CONTAINER_SHA=${CONTAINER_SHA256} METADATA_RUNTIME_ATTESTATION=${RUNTIME_ATTESTATION} METADATA_RUNTIME_ATTESTATION_SHA=${RUNTIME_ATTESTATION_SHA256} METADATA_RUNTIME_PREFLIGHT=${RUNTIME_PREFLIGHT_JOB_ID} METADATA_PYTHON_VERSION=${MANAGED_PYTHON_VERSION} METADATA_PYTHON_DIR=${MANAGED_PYTHON_INSTALL_DIR} METADATA_UV_VERSION=${PINNED_UV_VERSION} METADATA_UV_EXECUTABLE=${UV_EXECUTABLE} python3 - "${metadata_json_tmp}" <<'PY'
import json
import os
import sys

record = {
    "schema_version": 1, "job_id": int(os.environ["METADATA_JOB_ID"]),
    "rendered_driver_command": os.environ["METADATA_RENDERED_DRIVER"],
    "command": os.environ["METADATA_COMMAND"],
    "sbatch_argv": json.loads(os.environ["METADATA_SCHEDULER_ARGV_JSON"]),
    "output_pattern": os.environ["METADATA_OUTPUT_PATTERN"],
    "resolved_output_path": os.environ["METADATA_OUTPUT_PATH"],
    "run_log_dir": os.environ["METADATA_RUN_LOG_DIR"],
    "r3_validation_record_pattern": os.environ["METADATA_R3_RECORD_PATTERN"],
    "r3_validation_record_initial_path": os.environ["METADATA_R3_RECORD_INITIAL"],
    "r3_driver_command_file": os.environ["METADATA_R3_DRIVER_FILE"],
    "r3_driver_command_sha256": os.environ["METADATA_R3_DRIVER_SHA"],
    "r3_checker_path": os.environ["METADATA_R3_CHECKER_PATH"],
    "r3_checker_sha256": os.environ["METADATA_R3_CHECKER_SHA"],
    "r3_record_python": os.environ["METADATA_R3_RECORD_PYTHON"],
    "model": os.environ["METADATA_MODEL"], "dispatcher": os.environ["METADATA_DISPATCHER"],
    "scope": os.environ["METADATA_SCOPE"], "scope_name": os.environ["METADATA_SCOPE_NAME"],
    "mode": os.environ["METADATA_MODE"], "cluster": os.environ["METADATA_CLUSTER"],
    "profile": os.environ["METADATA_PROFILE"], "profile_sha256": os.environ["METADATA_PROFILE_SHA256"],
    "phase": os.environ["METADATA_PHASE"], "steps": int(os.environ["METADATA_STEPS"]),
    "run_group": os.environ["METADATA_RUN_GROUP"], "repeat": int(os.environ["METADATA_REPEAT"]),
    "router_replay": os.environ["METADATA_ROUTER_REPLAY"], "tensorboard_enabled": os.environ["METADATA_TENSORBOARD"] == "true",
    "topology": {"num_nodes": int(os.environ["METADATA_NUM_NODES"]), "gpus_per_node": int(os.environ["METADATA_GPUS_PER_NODE"])},
    "nemo_rl_commit": os.environ["METADATA_NEMORL_SHA"], "bridge_commit": os.environ["METADATA_BRIDGE_SHA"],
    "mcore_commit": os.environ["METADATA_MCORE_SHA"], "transformer_engine_commit": os.environ["METADATA_TE_SHA"],
    "container_path": os.environ["METADATA_CONTAINER_PATH"], "container_sha256": os.environ["METADATA_CONTAINER_SHA"],
    "runtime_attestation": os.environ["METADATA_RUNTIME_ATTESTATION"], "runtime_attestation_sha256": os.environ["METADATA_RUNTIME_ATTESTATION_SHA"],
    "runtime_preflight_job_id": int(os.environ["METADATA_RUNTIME_PREFLIGHT"]), "managed_python_version": os.environ["METADATA_PYTHON_VERSION"],
    "managed_python_install_dir": os.environ["METADATA_PYTHON_DIR"], "pinned_uv_version": os.environ["METADATA_UV_VERSION"],
    "uv_executable": os.environ["METADATA_UV_EXECUTABLE"],
}
with open(sys.argv[1], "w", encoding="utf-8") as output:
    json.dump(record, output, sort_keys=True)
    output.write("\n")
PY
mv -f "${metadata_env_tmp}" "${run_log_dir}/run-metadata.env"
metadata_env_tmp=
if ! mv -f "${metadata_json_tmp}" "${run_log_dir}/run-metadata.json"; then
  rm -f -- "${run_log_dir}/run-metadata.env"
  fail "Failed to publish authoritative run metadata JSON"
fi
metadata_json_tmp=
trap - EXIT
printf 'SLURM_JOB_ID: %s\n' "${job_id}"
