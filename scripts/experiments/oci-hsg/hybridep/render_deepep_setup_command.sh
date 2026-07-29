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
: "${DEEPEP_VARIANT:?DEEPEP_VARIANT is required}"
: "${NCCL_WHEEL:?NCCL_WHEEL is required}"
: "${NCCL_WHEEL_SHA256:?NCCL_WHEEL_SHA256 is required}"

case "${DEEPEP_VARIANT}" in
  deepep)
    runtime_probe="import importlib.metadata as md, os; import deep_ep; import deep_ep._C; from deep_ep import Buffer, ElasticBuffer, EventOverlap, EventHandle; root = os.path.realpath(os.environ['PYTHONPATH']); paths = [os.path.realpath(deep_ep.__file__), os.path.realpath(deep_ep._C.__file__)]; assert all(os.path.commonpath([root, path]) == root for path in paths), paths; print('DEEPEP_RUNTIME_VARIANT', 'deepep'); print('DEEPEP_RUNTIME_VERSION', md.version('deep_ep')); print('DEEPEP_RUNTIME_PATHS', *paths); print('DEEPEP_RUNTIME_NCCL', md.version('nvidia-nccl-cu13'))"
    ;;
  hybridep)
    runtime_probe="import importlib.metadata as md, os; import deep_ep; import deep_ep_cpp, hybrid_ep_cpp; from deep_ep import Buffer, HybridEPBuffer; root = os.path.realpath(os.environ['PYTHONPATH']); paths = [os.path.realpath(deep_ep.__file__), os.path.realpath(deep_ep_cpp.__file__), os.path.realpath(hybrid_ep_cpp.__file__)]; assert all(os.path.commonpath([root, path]) == root for path in paths), paths; print('DEEPEP_RUNTIME_VARIANT', 'hybridep'); print('DEEPEP_RUNTIME_VERSION', md.version('deep_ep')); print('DEEPEP_RUNTIME_PATHS', *paths); print('DEEPEP_RUNTIME_NCCL', md.version('nvidia-nccl-cu13'))"
    ;;
  *)
    printf 'DEEPEP_VARIANT must be deepep or hybridep: %s\n' \
      "${DEEPEP_VARIANT}" >&2
    exit 2
    ;;
esac

canonicalize_path() {
  python3 - "${1}" <<'PY'
import os
import sys

print(os.path.realpath(sys.argv[1]))
PY
}

if ! DEEPEP_WHEEL=$(canonicalize_path "${DEEPEP_WHEEL}"); then
  printf 'Failed to resolve DEEPEP_WHEEL: %s\n' "${DEEPEP_WHEEL}" >&2
  exit 2
fi
if ! NCCL_WHEEL=$(canonicalize_path "${NCCL_WHEEL}"); then
  printf 'Failed to resolve NCCL_WHEEL: %s\n' "${NCCL_WHEEL}" >&2
  exit 2
fi
case "${DEEPEP_WHEEL}" in
  /lustre/*) ;;
  *)
    printf 'DEEPEP_WHEEL must be under /lustre: %s\n' "${DEEPEP_WHEEL}" >&2
    exit 2
    ;;
esac
case "${NCCL_WHEEL}" in
  /lustre/*) ;;
  *)
    printf 'NCCL_WHEEL must be under /lustre: %s\n' "${NCCL_WHEEL}" >&2
    exit 2
    ;;
esac

overlay_parent=$(dirname -- "${DEEPEP_OVERLAY}")
overlay_name=$(basename -- "${DEEPEP_OVERLAY}")
if [[ "${overlay_parent}" != "/tmp" || \
  "${overlay_name}" != nemo-rl-deepep-* || \
  "${overlay_name}" == "nemo-rl-deepep-" ]]; then
  printf 'DEEPEP_OVERLAY must be an immediate child of /tmp named nemo-rl-deepep-*: %s\n' \
    "${DEEPEP_OVERLAY}" >&2
  exit 2
fi
canonical_tmp=$(cd -P -- /tmp && pwd -P)
if ! canonical_overlay_parent=$(cd -P -- "${overlay_parent}" 2>/dev/null && pwd -P); then
  printf 'DEEPEP_OVERLAY parent does not exist: %s\n' "${overlay_parent}" >&2
  exit 2
fi
if [[ "${canonical_overlay_parent}" != "${canonical_tmp}" ]]; then
  printf 'DEEPEP_OVERLAY parent must resolve to /tmp: %s\n' \
    "${DEEPEP_OVERLAY}" >&2
  exit 2
fi
DEEPEP_OVERLAY="${canonical_tmp}/${overlay_name}"
if [[ -L "${DEEPEP_OVERLAY}" ]]; then
  printf 'DEEPEP_OVERLAY must not be a symlink: %s\n' "${DEEPEP_OVERLAY}" >&2
  exit 2
fi

runtime_python=/opt/nemo_rl_venv/bin/python
if [[ -n "${RAY_VENV:-}" ]]; then
  runtime_python="${RAY_VENV}/bin/python"
fi

printf -v overlay_assignment 'overlay=%q' "${DEEPEP_OVERLAY}"
printf -v canonical_tmp_assignment 'canonical_tmp=%q' "${canonical_tmp}"
printf -v deepep_variant_assignment 'deepep_variant=%q' "${DEEPEP_VARIANT}"
printf -v deepep_wheel_assignment 'deepep_wheel=%q' "${DEEPEP_WHEEL}"
printf -v deepep_wheel_sha_assignment 'expected_deepep_wheel_sha256=%q' \
  "${DEEPEP_WHEEL_SHA256}"
printf -v nccl_wheel_assignment 'nccl_wheel=%q' "${NCCL_WHEEL}"
printf -v nccl_wheel_sha_assignment 'expected_nccl_wheel_sha256=%q' \
  "${NCCL_WHEEL_SHA256}"
printf -v runtime_python_assignment 'runtime_python=%q' "${runtime_python}"
printf -v probe_command 'PYTHONPATH="${overlay}" "${runtime_python}" -c "%s"' \
  "${runtime_probe}"

printf '%s\n' \
  'set -euo pipefail' \
  "${overlay_assignment}" \
  "${canonical_tmp_assignment}" \
  "${deepep_variant_assignment}" \
  "${deepep_wheel_assignment}" \
  "${deepep_wheel_sha_assignment}" \
  "${nccl_wheel_assignment}" \
  "${nccl_wheel_sha_assignment}" \
  "${runtime_python_assignment}" \
  '[[ "$(dirname -- "${overlay}")" == "${canonical_tmp}" ]]' \
  '[[ "$(basename -- "${overlay}")" == nemo-rl-deepep-* && "$(basename -- "${overlay}")" != nemo-rl-deepep- ]]' \
  'actual_nccl_wheel_sha256=$(sha256sum "${nccl_wheel}" | cut -d" " -f1)' \
  '[[ "${actual_nccl_wheel_sha256}" == "${expected_nccl_wheel_sha256}" ]]' \
  'actual_deepep_wheel_sha256=$(sha256sum "${deepep_wheel}" | cut -d" " -f1)' \
  '[[ "${actual_deepep_wheel_sha256}" == "${expected_deepep_wheel_sha256}" ]]' \
  '[[ ! -L "${overlay}" ]]' \
  'rm -rf -- "${overlay}"' \
  'mkdir -p "${overlay}"' \
  'unset UV_CONFIG_FILE' \
  'UV_NO_CONFIG=1 uv pip install --python "${runtime_python}" --target "${overlay}" --reinstall --no-deps --no-index "${nccl_wheel}" "${deepep_wheel}"' \
  'export LD_LIBRARY_PATH="${overlay}/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"' \
  "${probe_command}"
