#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "--mark-manifest-failed" ]]; then
  manifest="${2:?missing manifest path}"
  temporary="${manifest}.$$.$RANDOM.tmp"
  trap 'rm -f "${temporary}"' EXIT
  awk '
    {
      gsub(/"status": "queued"/, "\"status\": \"failed\"")
      if ($0 == "  \"status\": \"failed\"") {
        print "  \"error\": \"worker failed before terminal manifest\","
      }
      print
    }
  ' "${manifest}" > "${temporary}"
  mv -f "${temporary}" "${manifest}"
  trap - EXIT
  exit 0
elif [[ "${1:-}" == "--worker-script" ]]; then
  shift
  worker_script="${1:?missing absolute worker script path}"
  shift
  python_bin="/opt/nemo_rl_venv/bin/python"
  worker_args=("$@")
  output_dir=""
  for ((index = 0; index < ${#worker_args[@]}; index++)); do
    if [[ "${worker_args[index]}" == "--output-dir" ]]; then
      output_dir="${worker_args[index + 1]:-}"
      break
    fi
  done
  set +e
  "${python_bin}" "${worker_script}" --worker "${worker_args[@]}"
  status=$?
  set -e
  if ((status != 0)) && [[ -n "${output_dir}" ]]; then
    manifest="${output_dir}/drafter-staging-manifest.json"
    if ! grep -Eq '"status": "(failed|staged)"' "${manifest}" 2>/dev/null; then
      "${BASH_SOURCE[0]}" --mark-manifest-failed "${manifest}"
    fi
  fi
  exit "${status}"
else
  python_bin="${PYTHON_BIN:-python3}"
fi
exec "${python_bin}" "${script_dir}/stage_drafters.py" "$@"
