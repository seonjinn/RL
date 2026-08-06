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

: "${BRIDGE_REPOSITORY:?Set BRIDGE_REPOSITORY}"
: "${EXPECTED_BRIDGE_SHA:?Set EXPECTED_BRIDGE_SHA}"
: "${EXPECTED_MCORE_SHA:?Set EXPECTED_MCORE_SHA}"
: "${ARTIFACT_DIR:?Set ARTIFACT_DIR}"
: "${CONTAINER:?Set CONTAINER}"
: "${CONTAINER_SHA256:?Set CONTAINER_SHA256}"

LOCK_PYTHON=${LOCK_PYTHON:-python3.12}
CONTAINER_PYTHON=${CONTAINER_PYTHON:-/opt/nemo_rl_venv/bin/python}

valid_bridge_remote() {
  local repository=$1
  local remainder
  local host
  local path

  if [[ "${repository}" == *\?* || "${repository}" == *\#* || \
        "${repository}" =~ [[:space:][:cntrl:]] ]]; then
    return 1
  fi
  case "${repository}" in
    https://*)
      remainder=${repository#https://}
      [[ "${remainder}" == */* ]] || return 1
      host=${remainder%%/*}
      path=${remainder#*/}
      [[ "${host}" =~ ^[A-Za-z0-9][A-Za-z0-9.-]*(:[0-9]+)?$ ]] || return 1
      ;;
    git@*)
      [[ "${repository}" =~ ^git@[A-Za-z0-9][A-Za-z0-9.-]*: ]] || return 1
      path=${repository#*:}
      ;;
    *) return 1 ;;
  esac
  case "/${path}/" in
    */../*|*/./*) return 1 ;;
  esac
  [[ "${path}" =~ ^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$ ]]
}

valid_bridge_source() {
  local repository=$1
  if [[ "${repository}" == /* ]]; then
    [[ ! "${repository}" =~ [[:space:][:cntrl:]] && \
       "${repository}" != *\?* && "${repository}" != *\#* ]]
    return
  fi
  valid_bridge_remote "${repository}"
}

if ! valid_bridge_source "${BRIDGE_REPOSITORY}"; then
  echo "BRIDGE_REPOSITORY is not an approved credential-free source" >&2
  exit 2
fi

job_key=${SLURM_JOB_ID:-local}
work_parent=${WORK_ROOT:-${SLURM_TMPDIR:-/tmp}}
if [[ "${work_parent}" != /* ]]; then
  echo "WORK_ROOT must be an absolute parent directory" >&2
  exit 2
fi
mkdir -p "${work_parent}"
work_parent=$(cd "${work_parent}" && pwd -P)
if [[ "${work_parent}" == "/" ]]; then
  echo "WORK_ROOT must not resolve to the filesystem root" >&2
  exit 2
fi
work_root=$(mktemp -d "${work_parent}/bridge-bootstrap-${job_key}.XXXXXX")
checkout=${work_root}/Megatron-Bridge
result_dir=${ARTIFACT_DIR}/bridge-${EXPECTED_BRIDGE_SHA}-${job_key}
partial_result_dir=${result_dir}.partial

if [[ -e "${result_dir}" || -e "${partial_result_dir}" ]]; then
  echo "Bootstrap result already exists: ${result_dir}" >&2
  exit 1
fi

mkdir -p "${partial_result_dir}"

status_file=${partial_result_dir}/status.txt
cleanup() {
  rm -rf "${work_root}"
}
record_failure() {
  exit_code=$?
  trap - ERR
  printf 'failed exit_code=%s\n' "${exit_code}" >"${status_file}"
  mv "${partial_result_dir}" "${result_dir}"
  cleanup
  trap - EXIT
  exit "${exit_code}"
}
trap record_failure ERR
trap cleanup EXIT

git clone --no-checkout "${BRIDGE_REPOSITORY}" "${checkout}"
git -C "${checkout}" checkout --detach "${EXPECTED_BRIDGE_SHA}"
git -C "${checkout}" submodule update --init --recursive

actual_bridge_sha=$(git -C "${checkout}" rev-parse HEAD)
actual_mcore_sha=$(git -C "${checkout}/3rdparty/Megatron-LM" rev-parse HEAD)
if [[ "${actual_bridge_sha}" != "${EXPECTED_BRIDGE_SHA}" ]]; then
  echo "Bridge SHA mismatch: expected ${EXPECTED_BRIDGE_SHA}, got ${actual_bridge_sha}" >&2
  exit 1
fi
if [[ "${actual_mcore_sha}" != "${EXPECTED_MCORE_SHA}" ]]; then
  echo "MCore SHA mismatch: expected ${EXPECTED_MCORE_SHA}, got ${actual_mcore_sha}" >&2
  exit 1
fi

export UV_CACHE_DIR=${UV_CACHE_DIR:-${work_root}/uv-cache}
export FAST_HADAMARD_TRANSFORM_SKIP_CUDA_BUILD=TRUE
(
  cd "${checkout}"
  uv lock \
    --no-build-isolation-package fast-hadamard-transform \
    --python "${LOCK_PYTHON}" \
    --no-python-downloads
)

cp "${checkout}/uv.lock" "${partial_result_dir}/uv.lock"
lock_sha256=$(sha256sum "${partial_result_dir}/uv.lock" | awk '{print $1}')
printf '%s  uv.lock\n' "${lock_sha256}" >"${partial_result_dir}/uv.lock.sha256"
container_python_version=$("${CONTAINER_PYTHON}" --version 2>&1)
{
  printf 'bridge_sha=%s\n' "${actual_bridge_sha}"
  printf 'mcore_sha=%s\n' "${actual_mcore_sha}"
  printf 'container=%s\n' "${CONTAINER}"
  printf 'container_sha256=%s\n' "${CONTAINER_SHA256}"
  printf 'uv_lock_sha256=%s\n' "${lock_sha256}"
  printf 'lock_python=%s\n' "${LOCK_PYTHON}"
  printf 'container_python=%s\n' "${CONTAINER_PYTHON}"
  printf 'container_python_version=%s\n' "${container_python_version}"
  printf 'slurm_job_id=%s\n' "${job_key}"
  printf 'created_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"${partial_result_dir}/provenance.env"

recipe_tests=(
  tests/unit_tests/recipes/nemotronh/test_nemotron_3_nano.py
  tests/unit_tests/recipes/nemotronh/test_nemotron_3_super.py
  tests/unit_tests/recipes/nemotronh/test_nemotron_3_ultra.py
)
export PYTHONPATH="${checkout}/src:${checkout}/3rdparty/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}"
"${CONTAINER_PYTHON}" -c \
  "import megatron.bridge; import megatron.core; import torch; import transformer_engine"
(
  cd "${checkout}"
  "${CONTAINER_PYTHON}" -m pytest -q \
    --junitxml="${partial_result_dir}/recipe-tests.junit.xml" \
    "${recipe_tests[@]}"
) 2>&1 | tee "${partial_result_dir}/recipe-tests.log"

echo "passed" >"${status_file}"
trap - ERR
mv "${partial_result_dir}" "${result_dir}"
echo "bridge_bootstrap_artifact=${result_dir}"
