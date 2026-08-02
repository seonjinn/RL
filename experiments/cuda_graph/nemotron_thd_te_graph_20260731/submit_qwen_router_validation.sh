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

validate_test_controls() {
  case "${TEST_ONLY:-0}" in
    0|1) ;;
    *) fail "TEST_ONLY must be 0 or 1" ;;
  esac
  case "${SBATCH_TEST_ONLY:-0}" in
    0|1) ;;
    *) fail "SBATCH_TEST_ONLY must be 0 or 1" ;;
  esac
  if [[ "${TEST_ONLY:-0}" == "1" && "${SBATCH_TEST_ONLY:-0}" == "1" ]]; then
    fail "TEST_ONLY and SBATCH_TEST_ONLY are mutually exclusive"
  fi
}

require_gate_inputs() {
  local file_name=$1
  local digest_name=$2
  local file_value=${!file_name:-}
  local digest_value=${!digest_name:-}

  [[ -n "${file_value}" && -n "${digest_value}" ]] || \
    fail "${file_name} and ${digest_name} are required"
}

validate_gate() {
  local kind=$1
  local gate_file=$2
  local gate_sha256=$3
  shift 3
  local -a command=(
    python3
    "${validator}"
    "${kind}"
    --gate-file "${gate_file}"
    --gate-sha256 "${gate_sha256}"
    --model "${MODEL}"
    --profile-file "${profile_file}"
    --profile-dir "${profile_dir}"
    --cluster "${CLUSTER}"
  )
  local arm
  for arm in "$@"; do
    command+=(--arm "${arm}")
  done
  local validation_output
  validation_output=$("${command[@]}") || fail "${kind} campaign gate validation failed"
  [[ "${validation_output}" =~ ^PROFILE_SHA256=([0-9a-f]{64})$ ]] || \
    fail "${kind} campaign gate returned malformed profile digest"
  VALIDATED_PROFILE_SHA256=${BASH_REMATCH[1]}
}

resolve_leaf() {
  local arm=$1
  local relative_leaf
  local leaf
  local conditions_dir
  local resolved_leaf

  case "${arm}" in
    A) relative_leaf=conditions/qwen_A_baseline_r3off.sh ;;
    B) relative_leaf=conditions/qwen_B_moe_router_r3off.sh ;;
    C) relative_leaf=conditions/qwen_C_baseline_r3on.sh ;;
    E) relative_leaf=conditions/qwen_E_attn_r3on.sh ;;
    *) fail "Qwen router validation arms must be A, B, C, or E" ;;
  esac
  conditions_dir=$(realpath "${script_dir}/conditions") || \
    fail "Missing Qwen router validation conditions directory"
  [[ "${conditions_dir}" == "${script_dir}/conditions" ]] || \
    fail "Qwen router validation conditions directory escapes the experiment"
  leaf=${script_dir}/${relative_leaf}
  [[ -f "${leaf}" ]] || fail "Missing Qwen router validation leaf: ${relative_leaf}"
  resolved_leaf=$(realpath "${leaf}") || \
    fail "Cannot resolve Qwen router validation leaf: ${relative_leaf}"
  case "${resolved_leaf}" in
    "${conditions_dir}"/*) ;;
    *) fail "Qwen router validation leaf escapes its persistent directory" ;;
  esac
  printf '%s\n' "${resolved_leaf}"
}

: "${CLUSTER:?Set CLUSTER to ptyche, oci-hsg, or lyris}"
: "${MODEL:?Set MODEL to qwen3_30ba3b or qwen3_235b}"
submitter_path=$(realpath "${BASH_SOURCE[0]}") || fail "Cannot resolve Qwen router submitter"
script_dir=$(dirname "${submitter_path}")
validator=${script_dir}/validate_campaign_gate.py
[[ -f "${validator}" && ! -L "${validator}" ]] || \
  fail "Campaign gate validator must be a regular non-symlink file"
run_tag=${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
mode=${MODE:-nemorl}
phase=${PHASE:-smoke}
repeats=${REPEATS:-1}

validate_test_controls

case "${CLUSTER}" in
  ptyche|oci-hsg|lyris) ;;
  *) fail "CLUSTER must be ptyche, oci-hsg, or lyris" ;;
esac

case "${MODEL}" in
  qwen3_30ba3b|qwen3_235b) ;;
  *) fail "MODEL must be qwen3_30ba3b or qwen3_235b" ;;
esac
case "${phase}" in
  smoke) steps=5 ;;
  performance) steps=20 ;;
  *) fail "PHASE must be smoke or performance" ;;
esac
case "${repeats}" in
  1|3) ;;
  *) fail "REPEATS must be 1 or 3" ;;
esac

if (($#)); then
  arms=("$@")
else
  case "${phase}:${MODEL}" in
    smoke:qwen3_30ba3b) arms=(A B C E) ;;
    smoke:qwen3_235b|performance:qwen3_30ba3b|performance:qwen3_235b)
      arms=(A B)
      ;;
  esac
fi

selected_arms=" "
for arm in "${arms[@]}"; do
  case "${arm}" in
    A|B|C|E) ;;
    *) fail "Qwen router validation arms must be A, B, C, or E" ;;
  esac
  [[ "${selected_arms}" != *" ${arm} "* ]] || \
    fail "Qwen router validation arms must not contain duplicates"
  selected_arms+="${arm} "
done

requires_r3_gate=false
for arm in "${arms[@]}"; do
  if [[ "${MODEL}" == "qwen3_235b" && ( "${arm}" == C || "${arm}" == E ) ]]; then
    requires_r3_gate=true
  fi
done
if [[ "${requires_r3_gate}" == "true" || "${phase}" == "performance" ]]; then
  profile_dir=${script_dir}/profiles
  if [[ -n "${PROFILE_FILE:-}" ]]; then
    profile_file=${PROFILE_FILE}
  elif [[ -e "${profile_dir}/${CLUSTER}.env" || -L "${profile_dir}/${CLUSTER}.env" ]]; then
    profile_file=${profile_dir}/${CLUSTER}.env
  else
    profile_file=${profile_dir}/${CLUSTER}.env.example
  fi
fi
if [[ "${requires_r3_gate}" == "true" ]]; then
  require_gate_inputs R3_PREFLIGHT_FILE R3_PREFLIGHT_SHA256
  validate_gate r3 "${R3_PREFLIGHT_FILE}" "${R3_PREFLIGHT_SHA256}"
fi
if [[ "${phase}" == "performance" ]]; then
  require_gate_inputs SMOKE_PROMOTION_FILE SMOKE_PROMOTION_SHA256
  validate_gate promotion "${SMOKE_PROMOTION_FILE}" "${SMOKE_PROMOTION_SHA256}" "${arms[@]}"
fi

resolved_leaves=()
for arm in "${arms[@]}"; do
  resolved_leaves+=("$(resolve_leaf "${arm}")")
done

for repeat_index in $(seq 1 "${repeats}"); do
  repeat_tag=${run_tag}-r${repeat_index}
  for arm_index in "${!arms[@]}"; do
    arm=${arms[${arm_index}]}
    case "${arm}" in
      A|B) router_replay=off; r3_name=r3off ;;
      C|E) router_replay=on; r3_name=r3on ;;
      *) fail "Qwen router validation arms must be A, B, C, or E" ;;
    esac
    run_group=qwen-router-${r3_name}-${MODEL}-${mode}-${CLUSTER}-${run_tag}
    printf 'MATRIX_ROW: arm=%s phase=%s repeat=%s group=%s\n' \
      "${arm}" "${phase}" "${repeat_index}" "${run_group}"
    CLUSTER="${CLUSTER}" \
    MODEL="${MODEL}" \
    MODE="${mode}" \
    STEPS="${steps}" \
    TEST_ONLY="${TEST_ONLY:-0}" \
    SBATCH_TEST_ONLY="${SBATCH_TEST_ONLY:-0}" \
    PROFILE_FILE="${profile_file:-}" \
    VALIDATED_PROFILE_SHA256="${VALIDATED_PROFILE_SHA256:-}" \
    RUN_GROUP="${run_group}" \
    REPEAT_INDEX="${repeat_index}" \
    RUN_TAG="${repeat_tag}" \
    bash "${resolved_leaves[${arm_index}]}"
  done
done
