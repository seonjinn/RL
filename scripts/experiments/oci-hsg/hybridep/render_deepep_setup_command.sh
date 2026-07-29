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

: "${DEEPEP_OVERLAY:?DEEPEP_OVERLAY is required}"
: "${DEEPEP_WHEEL:?DEEPEP_WHEEL is required}"
: "${DEEPEP_WHEEL_SHA256:?DEEPEP_WHEEL_SHA256 is required}"

runtime_python=/opt/nemo_rl_venv/bin/python
if [[ -n "${RAY_VENV:-}" ]]; then
  runtime_python="${RAY_VENV}/bin/python"
fi

printf -v overlay_assignment 'overlay=%q' "${DEEPEP_OVERLAY}"
printf -v wheel_assignment 'wheel=%q' "${DEEPEP_WHEEL}"
printf -v wheel_sha_assignment 'expected_wheel_sha256=%q' \
  "${DEEPEP_WHEEL_SHA256}"
printf -v runtime_python_assignment 'runtime_python=%q' "${runtime_python}"

printf '%s\n' \
  'set -euo pipefail' \
  "${overlay_assignment}" \
  "${wheel_assignment}" \
  "${wheel_sha_assignment}" \
  "${runtime_python_assignment}" \
  '[[ "${overlay}" == /tmp/nemo-rl-deepep-* && "${overlay}" != /tmp/nemo-rl-deepep- ]]' \
  'actual_wheel_sha256=$(sha256sum "${wheel}" | cut -d" " -f1)' \
  '[[ "${actual_wheel_sha256}" == "${expected_wheel_sha256}" ]]' \
  'rm -rf -- "${overlay}"' \
  'mkdir -p "${overlay}"' \
  'unset UV_CONFIG_FILE' \
  'UV_NO_CONFIG=1 uv pip install --python "${runtime_python}" --target "${overlay}" --reinstall --no-deps --no-index "${wheel}"' \
  'PYTHONPATH="${overlay}" "${runtime_python}" -c "import importlib.metadata as md, os; import deep_ep, deep_ep_cpp, hybrid_ep_cpp; root = os.path.realpath(os.environ[\"PYTHONPATH\"]); paths = [os.path.realpath(deep_ep.__file__), os.path.realpath(deep_ep_cpp.__file__), os.path.realpath(hybrid_ep_cpp.__file__)]; assert all(os.path.commonpath([root, path]) == root for path in paths), paths; print(\"DEEPEP_RUNTIME_VERSION\", md.version(\"deep_ep\")); print(\"DEEPEP_RUNTIME_PATHS\", *paths)"'
