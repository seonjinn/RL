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
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
run_tag=${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
mode=${MODE:-nemorl}
phase=${PHASE:-smoke}
repeats=${REPEATS:-1}

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
  arms=(A B C E)
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
    RUN_GROUP="${run_group}" \
    REPEAT_INDEX="${repeat_index}" \
    RUN_TAG="${repeat_tag}" \
    bash "${resolved_leaves[${arm_index}]}"
  done
done
