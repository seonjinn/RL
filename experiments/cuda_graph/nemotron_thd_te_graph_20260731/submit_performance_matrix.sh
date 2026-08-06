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

resolve_launcher() {
  local relative_launcher=$1
  local launcher_group
  local allowed_dir
  local launcher
  local resolved_launcher

  if [[ ! "${relative_launcher}" =~ ^(scopes|variants)/[[:alnum:]][[:alnum:]_.-]*\.sh$ ]]; then
    fail "Performance launchers must be a single persistent scopes/ or variants/ leaf"
  fi
  launcher_group=${relative_launcher%%/*}
  allowed_dir=$(realpath "${script_dir}/${launcher_group}") || \
    fail "Missing performance launcher directory: ${launcher_group}"
  [[ "${allowed_dir}" == "${script_dir}/${launcher_group}" ]] || \
    fail "Performance launcher directory escapes the experiment"
  launcher=${script_dir}/${relative_launcher}
  [[ -f "${launcher}" ]] || \
    fail "Missing performance launcher: ${relative_launcher}"
  resolved_launcher=$(realpath "${launcher}") || \
    fail "Cannot resolve performance launcher: ${relative_launcher}"
  case "${resolved_launcher}" in
    "${allowed_dir}"/*) ;;
    *) fail "Performance launcher escapes its persistent directory: ${relative_launcher}" ;;
  esac
  printf '%s\n' "${resolved_launcher}"
}

: "${CLUSTER:?Set CLUSTER to ptyche, oci-hsg, or lyris}"
: "${MODEL:?Set MODEL to a committed model selector}"
case "${MODEL}" in
  qwen3_30ba3b|qwen3_235b)
    fail "Qwen campaigns must use submit_qwen_router_validation.sh"
    ;;
esac
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
run_tag=${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
mode=${MODE:-nemorl}
run_group=${RUN_GROUP:-performance-${MODEL}-${mode}-${CLUSTER}-${run_tag}}
baseline=scopes/00_baseline_no_cg.sh

if (($#)); then
  requested_launchers=("$@")
elif [[ -n "${PERFORMANCE_SCRIPTS:-}" ]]; then
  read -r -a requested_launchers <<<"${PERFORMANCE_SCRIPTS}"
else
  requested_launchers=(
    "${baseline}"
    scopes/17_attn.sh
  )
  case "${MODEL}" in
    nano)
      requested_launchers+=(
        scopes/09_mlp.sh
        scopes/05_mamba.sh
        scopes/03_moe_router.sh
        scopes/31_attn_mlp_mamba_moe_router.sh
      )
      ;;
    super)
      requested_launchers+=(
        scopes/09_mlp.sh
        scopes/05_mamba.sh
        scopes/03_moe_router.sh
        scopes/04_moe_router_preprocess.sh
        scopes/32_attn_mlp_mamba_moe_router_preprocess.sh
      )
      ;;
    ultra)
      requested_launchers+=(
        scopes/03_moe_router.sh
        scopes/04_moe_router_preprocess.sh
        scopes/20_attn_moe_router_preprocess.sh
      )
      ;;
    *) fail "MODEL must be nano, super, or ultra" ;;
  esac
fi

launchers=("${baseline}")
for relative_launcher in "${requested_launchers[@]}"; do
  already_selected=false
  for selected_launcher in "${launchers[@]}"; do
    if [[ "${relative_launcher}" == "${selected_launcher}" ]]; then
      already_selected=true
      break
    fi
  done
  [[ "${already_selected}" == "true" ]] && continue
  launchers+=("${relative_launcher}")
done

resolved_launchers=()
for relative_launcher in "${launchers[@]}"; do
  resolved_launchers+=("$(resolve_launcher "${relative_launcher}")")
done

for repeat_index in 1 2 3; do
  repeat_tag=${run_tag}-r${repeat_index}
  for launcher_index in "${!launchers[@]}"; do
    relative_launcher=${launchers[${launcher_index}]}
    printf 'MATRIX_ROW: %s repeat=%s group=%s\n' \
      "${relative_launcher}" "${repeat_index}" "${run_group}"
    CLUSTER="${CLUSTER}" \
    MODEL="${MODEL}" \
    MODE="${mode}" \
    STEPS=20 \
    TEST_ONLY="${TEST_ONLY:-0}" \
    SBATCH_TEST_ONLY="${SBATCH_TEST_ONLY:-0}" \
    RUN_GROUP="${run_group}" \
    REPEAT_INDEX="${repeat_index}" \
    RUN_TAG="${repeat_tag}" \
    bash "${resolved_launchers[${launcher_index}]}"
  done
done
