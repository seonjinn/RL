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

  if [[ ! "${relative_launcher}" =~ ^scopes/[[:alnum:]][[:alnum:]_.-]*\.sh$ ]]; then
    fail "Smoke launchers must be a single persistent scopes/ leaf"
  fi
  launcher_group=${relative_launcher%%/*}
  allowed_dir=$(realpath "${script_dir}/${launcher_group}") || \
    fail "Missing smoke launcher directory: ${launcher_group}"
  [[ "${allowed_dir}" == "${script_dir}/scopes" ]] || \
    fail "Smoke launcher directory escapes the experiment"
  launcher=${script_dir}/${relative_launcher}
  [[ -f "${launcher}" ]] || fail "Missing smoke launcher: ${relative_launcher}"
  resolved_launcher=$(realpath "${launcher}") || \
    fail "Cannot resolve smoke launcher: ${relative_launcher}"
  case "${resolved_launcher}" in
    "${allowed_dir}"/*) ;;
    *) fail "Smoke launcher escapes its persistent directory: ${relative_launcher}" ;;
  esac
  printf '%s\n' "${resolved_launcher}"
}

: "${CLUSTER:?Set CLUSTER to ptyche, oci-hsg, or lyris}"
: "${MODEL:?Set MODEL to a committed model selector}"
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
run_tag=${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}

relative_launchers=()
resolved_launchers=()
for launcher in "${script_dir}"/scopes/*.sh; do
  relative_launcher=${launcher#${script_dir}/}
  relative_launchers+=("${relative_launcher}")
  resolved_launchers+=("$(resolve_launcher "${relative_launcher}")")
done

for launcher_index in "${!relative_launchers[@]}"; do
  relative_launcher=${relative_launchers[${launcher_index}]}
  printf 'MATRIX_ROW: %s\n' "${relative_launcher}"
  CLUSTER="${CLUSTER}" \
  MODEL="${MODEL}" \
  MODE="${MODE:-nemorl}" \
  STEPS=5 \
  TEST_ONLY="${TEST_ONLY:-0}" \
  SBATCH_TEST_ONLY="${SBATCH_TEST_ONLY:-0}" \
  RUN_TAG="${run_tag}" \
  bash "${resolved_launchers[${launcher_index}]}"
done
