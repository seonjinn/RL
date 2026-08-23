#!/usr/bin/env bash

set -euo pipefail

: "${SOURCE_ROOT:?set SOURCE_ROOT to the exact source checkout}"
: "${NODE_SCRATCH_ROOT:?set NODE_SCRATCH_ROOT to the node-local scratch root}"
: "${DURABLE_RESULT_ROOT:?set DURABLE_RESULT_ROOT to the durable result directory}"
: "${EXPECTED_HEAD:?set EXPECTED_HEAD to the exact source commit}"
: "${EXPECTED_UV_LOCK_SHA:?set EXPECTED_UV_LOCK_SHA to the exact uv.lock SHA256}"
: "${SLURM_JOB_ID:?set SLURM_JOB_ID}"
: "${SLURMD_NODENAME:?set SLURMD_NODENAME}"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

test -d "${SOURCE_ROOT}"
test -d "${DURABLE_RESULT_ROOT}"
SOURCE_ROOT="$(realpath "${SOURCE_ROOT}")"
DURABLE_RESULT_ROOT="$(realpath "${DURABLE_RESULT_ROOT}")"
if [[ "${PR11_BOOTSTRAP_ALLOW_TEST_PATHS:-0}" != 1 ]]; then
  case "${SOURCE_ROOT}" in
    /home/*) ;;
    *) echo 'SOURCE_ROOT must resolve below /home' >&2; exit 2 ;;
  esac
  case "${NODE_SCRATCH_ROOT}" in
    /raid/scratch/*) ;;
    *) echo 'NODE_SCRATCH_ROOT must be below /raid/scratch' >&2; exit 2 ;;
  esac
  case "${DURABLE_RESULT_ROOT}" in
    /lustre/*) ;;
    *) echo 'DURABLE_RESULT_ROOT must resolve below /lustre' >&2; exit 2 ;;
  esac
fi

actual_head="$(git -C "${SOURCE_ROOT}" rev-parse HEAD)"
test "${actual_head}" = "${EXPECTED_HEAD}"
actual_lock_sha="$(sha256_file "${SOURCE_ROOT}/uv.lock")"
test "${actual_lock_sha}" = "${EXPECTED_UV_LOCK_SHA}"

test ! -e "${NODE_SCRATCH_ROOT}"
mkdir -p "${NODE_SCRATCH_ROOT}"
venv_dir="${NODE_SCRATCH_ROOT}/venv"
export UV_PROJECT_ENVIRONMENT="${venv_dir}"
export UV_CACHE_DIR="${NODE_SCRATCH_ROOT}/uv-cache"

cd "${SOURCE_ROOT}"
uv sync --locked --extra mcore --group test

shopt -s nullglob
cudnn_sonames=(
  "${venv_dir}"/lib/python*/site-packages/nvidia/cudnn/lib/libcudnn.so.9
)
shopt -u nullglob
if [[ "${#cudnn_sonames[@]}" -ne 1 ]]; then
  echo "expected exactly one node-local libcudnn.so.9, found ${#cudnn_sonames[@]}" >&2
  exit 3
fi
cudnn_lib="$(dirname "$(realpath "${cudnn_sonames[0]}")")"
export CUDNN_HOME="${cudnn_lib}"
export CUDNN_PATH="${cudnn_lib}"
export LD_LIBRARY_PATH="${cudnn_lib}:${LD_LIBRARY_PATH:-}"

marker_tmp="${venv_dir}/.nemo-rl-build-marker.tmp.${SLURM_JOB_ID}"
build_marker="${venv_dir}/.nemo-rl-build-marker"
printf '%s\n' \
  "expected_head=${EXPECTED_HEAD}" \
  "expected_uv_lock_sha=${EXPECTED_UV_LOCK_SHA}" \
  "creation_slurm_job_id=${SLURM_JOB_ID}" \
  "cudnn_lib=${cudnn_lib}" \
  'uv_sync_locked=complete' \
  > "${marker_tmp}"
mv "${marker_tmp}" "${build_marker}"

python_bin="${venv_dir}/bin/python"
test -x "${python_bin}"
"${python_bin}" -c 'import torch; import transformer_engine; import megatron.core'

installed_tmp="${NODE_SCRATCH_ROOT}/installed-distributions.txt"
"${python_bin}" - <<'PY' > "${installed_tmp}"
import re
from importlib.metadata import distributions

installed = set()
for distribution in distributions():
    name = distribution.metadata.get("Name")
    if name:
        normalized_name = re.sub(r"[-_.]+", "-", name).lower()
        installed.add(f"{normalized_name}=={distribution.version}")

for requirement in sorted(installed):
    print(requirement)
PY
installed_sha="$(sha256_file "${installed_tmp}")"
installed_count="$(wc -l < "${installed_tmp}" | awk '{print $1}')"
installed_durable="${DURABLE_RESULT_ROOT}/installed-distributions-${SLURM_JOB_ID}-${SLURMD_NODENAME}.txt"
cp "${installed_tmp}" "${installed_durable}"

receipt_tmp="${NODE_SCRATCH_ROOT}/pr11-node-bootstrap-receipt.txt"
receipt_durable="${DURABLE_RESULT_ROOT}/pr11-node-bootstrap-${SLURM_JOB_ID}-${SLURMD_NODENAME}.txt"
{
  echo "node=${SLURMD_NODENAME}"
  echo "expected_head=${EXPECTED_HEAD}"
  echo "actual_head=${actual_head}"
  echo "expected_uv_lock_sha=${EXPECTED_UV_LOCK_SHA}"
  echo "actual_uv_lock_sha=${actual_lock_sha}"
  echo "venv=${venv_dir}"
  echo "build_marker=${build_marker}"
  echo "cudnn_lib=${cudnn_lib}"
  echo 'uv_sync_locked=complete'
  echo 'transformer_engine_import=PASS'
  echo "installed_distributions_path=${installed_durable}"
  echo "installed_distributions_sha256=${installed_sha}"
  echo "installed_distributions_count=${installed_count}"
  echo 'result=PASS'
} > "${receipt_tmp}"
cp "${receipt_tmp}" "${receipt_durable}.tmp.${SLURM_JOB_ID}"
mv "${receipt_durable}.tmp.${SLURM_JOB_ID}" "${receipt_durable}"

echo "PR11_NODE_BOOTSTRAP_PASS node=${SLURMD_NODENAME} distributions_sha256=${installed_sha}"
