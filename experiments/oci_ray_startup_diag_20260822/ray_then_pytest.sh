#!/usr/bin/env bash

set -euo pipefail

job_tag="${SLURM_JOB_ID:-$$}"
RAY_NODE_TMP="/tmp/nrr${job_tag}"
failure_root=""
receipt_path=""
ray_tmp_safe=no
stage=init

finish() {
  local rc=$?
  trap - EXIT
  set +e

  if [[ -n "${receipt_path}" ]]; then
    {
      echo "final_stage=${stage}"
      echo "result=$([[ ${rc} -eq 0 ]] && echo PASS || echo FAIL)"
      echo "exit_code=${rc}"
    } >> "${receipt_path}"
  fi

  if ((rc != 0)) && [[ -n "${failure_root}" ]]; then
    mkdir -p "${failure_root}"
    {
      echo "exit_code=${rc}"
      echo "stage=${stage}"
      echo "source_root=${SOURCE_ROOT:-unset}"
      echo "venv=${UV_PROJECT_ENVIRONMENT:-unset}"
      echo "expected_head=${EXPECTED_HEAD:-unset}"
      echo "actual_head=${actual_head:-unset}"
      echo "expected_uv_lock_sha=${EXPECTED_UV_LOCK_SHA:-unset}"
      echo "actual_uv_lock_sha=${actual_lock_sha256:-unset}"
      echo "build_marker=${build_marker:-unset}"
      echo "tmpdir=${TMPDIR:-unset}"
      echo "ray_tmpdir=${RAY_TMPDIR:-unset}"
      echo "pwd=$(pwd)"
    } > "${failure_root}/environment.txt"

    local log_name log_prefix ray_log_root
    for log_prefix in core dashboard pytest; do
      case "${log_prefix}" in
        core) ray_log_root="${RAY_NODE_TMP}/core/session_latest/logs" ;;
        dashboard) ray_log_root="${RAY_NODE_TMP}/dashboard/session_latest/logs" ;;
        pytest) ray_log_root="${RAY_NODE_TMP}/ray/session_latest/logs" ;;
      esac
      for log_name in \
        dashboard.log \
        dashboard.err \
        dashboard_MetricsHead.log \
        dashboard_MetricsHead.out \
        dashboard_MetricsHead.err \
        raylet.err \
        gcs_server.err \
        monitor.err \
        monitor.log; do
        if [[ -f "${ray_log_root}/${log_name}" ]]; then
          cp -L "${ray_log_root}/${log_name}" \
            "${failure_root}/${log_prefix}-${log_name}"
        fi
      done
    done

    local listed=0 session_file
    : > "${failure_root}/session-files.txt"
    if [[ -d "${RAY_NODE_TMP}" ]]; then
      while IFS= read -r session_file; do
        printf '%s\n' "${session_file}" >> "${failure_root}/session-files.txt"
        listed=$((listed + 1))
        if ((listed >= 200)); then
          echo "LISTING_TRUNCATED_AT=200" >> "${failure_root}/session-files.txt"
          break
        fi
      done < <(find "${RAY_NODE_TMP}" -maxdepth 5 -type f)
    fi
  fi

  if [[ "${ray_tmp_safe}" == yes ]]; then
    rm -rf -- "${RAY_NODE_TMP}"
  fi

  exit "${rc}"
}

trap finish EXIT

case "${job_tag}" in
  ''|*[!0-9]*) echo "job tag must contain decimal digits only" >&2; exit 2 ;;
  *) ray_tmp_safe=yes ;;
esac

: "${SOURCE_ROOT:?set SOURCE_ROOT to the exact /home source checkout}"
: "${EXPECTED_HEAD:?set EXPECTED_HEAD to the exact 40-character source commit}"
: "${EXPECTED_UV_LOCK_SHA:?set EXPECTED_UV_LOCK_SHA to the exact uv.lock SHA256}"
: "${UV_PROJECT_ENVIRONMENT:?set UV_PROJECT_ENVIRONMENT to the rebuilt /raid/scratch venv}"
: "${DURABLE_RESULT_ROOT:?set DURABLE_RESULT_ROOT to a pre-existing /lustre result directory}"

if (($# == 0)); then
  echo "pass at least one focused pytest target" >&2
  exit 2
fi

unset RAY_ADDRESS
export TMPDIR="${RAY_NODE_TMP}"
export RAY_TMPDIR="${RAY_NODE_TMP}"

test -d "${DURABLE_RESULT_ROOT}"
DURABLE_RESULT_ROOT="$(realpath "${DURABLE_RESULT_ROOT}")"
case "${DURABLE_RESULT_ROOT}" in
  /lustre/*) ;;
  *) echo "DURABLE_RESULT_ROOT must resolve below /lustre" >&2; exit 2 ;;
esac
failure_root="${DURABLE_RESULT_ROOT}/ray-failure-${job_tag}"

test -d "${SOURCE_ROOT}"
SOURCE_ROOT="$(realpath "${SOURCE_ROOT}")"
case "${SOURCE_ROOT}" in
  /home/*) ;;
  *) echo "SOURCE_ROOT must resolve below /home" >&2; exit 2 ;;
esac

test -d "${UV_PROJECT_ENVIRONMENT}"
UV_PROJECT_ENVIRONMENT="$(realpath "${UV_PROJECT_ENVIRONMENT}")"
case "${UV_PROJECT_ENVIRONMENT}" in
  /raid/scratch/*) ;;
  *) echo "UV_PROJECT_ENVIRONMENT must resolve below /raid/scratch" >&2; exit 2 ;;
esac

mkdir -p "${RAY_NODE_TMP}"
python_bin="${UV_PROJECT_ENVIRONMENT}/bin/python"
test -x "${python_bin}"

stage=source-receipt
test "${#EXPECTED_HEAD}" -eq 40
test "${#EXPECTED_UV_LOCK_SHA}" -eq 64
actual_head="$(git -C "${SOURCE_ROOT}" rev-parse HEAD)"
test "${actual_head}" = "${EXPECTED_HEAD}"
source_status="$(git -C "${SOURCE_ROOT}" status --porcelain --untracked-files=all --ignore-submodules=none)"
test -z "${source_status}"
submodule_status="$(git -C "${SOURCE_ROOT}" submodule status --recursive)"
printf '%s\n' "${submodule_status}" > "${RAY_NODE_TMP}/submodules.txt"
if ! awk 'substr($0, 1, 1) ~ /[-+U]/ { bad=1 } END { exit bad }' \
  "${RAY_NODE_TMP}/submodules.txt"; then
  echo "recursive submodule checkout is not clean" >&2
  exit 1
fi
actual_lock_sha256="$(sha256sum "${SOURCE_ROOT}/uv.lock" | awk '{print $1}')"
test "${actual_lock_sha256}" = "${EXPECTED_UV_LOCK_SHA}"

stage=venv-build-marker
build_marker="${UV_PROJECT_ENVIRONMENT}/.nemo-rl-build-marker"
test -f "${build_marker}"
test "$(realpath "${build_marker}")" = "${build_marker}"
expected_build_marker="$(printf '%s\n' \
  "expected_head=${EXPECTED_HEAD}" \
  "expected_uv_lock_sha=${EXPECTED_UV_LOCK_SHA}" \
  "creation_slurm_job_id=${job_tag}" \
  "uv_sync_locked=complete")"
actual_build_marker="$(cat "${build_marker}")"
test "${actual_build_marker}" = "${expected_build_marker}"

stage=python-receipt
python_receipt="$("${python_bin}" - <<'PY'
import platform
import sys

import ray

print(sys.executable)
print(platform.python_version())
print(ray.__version__)
print(sys.prefix)
PY
)"
python_executable="$(printf '%s\n' "${python_receipt}" | sed -n '1p')"
python_version="$(printf '%s\n' "${python_receipt}" | sed -n '2p')"
ray_version="$(printf '%s\n' "${python_receipt}" | sed -n '3p')"
venv_prefix="$(printf '%s\n' "${python_receipt}" | sed -n '4p')"
test "$(realpath "${venv_prefix}")" = "${UV_PROJECT_ENVIRONMENT}"
harness_shebang="$(head -n 1 "${BASH_SOURCE[0]}")"
test "${harness_shebang}" = '#!/usr/bin/env bash'

installed_distributions_tmp="${RAY_NODE_TMP}/installed-distributions.txt"
"${python_bin}" - <<'PY' > "${installed_distributions_tmp}"
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
installed_distributions_sha256="$(sha256sum "${installed_distributions_tmp}" | awk '{print $1}')"
installed_distributions_count="$(wc -l < "${installed_distributions_tmp}" | awk '{print $1}')"
installed_distributions_path="${DURABLE_RESULT_ROOT}/installed-distributions-${job_tag}.txt"
cp "${installed_distributions_tmp}" "${installed_distributions_path}"

receipt_path="${DURABLE_RESULT_ROOT}/ray-diagnostic-receipt-${job_tag}.txt"
{
  echo "expected_head=${EXPECTED_HEAD}"
  echo "actual_head=${actual_head}"
  echo "recursive_clean=yes"
  echo "expected_uv_lock_sha=${EXPECTED_UV_LOCK_SHA}"
  echo "actual_uv_lock_sha=${actual_lock_sha256}"
  echo "build_marker=${build_marker}"
  echo "build_marker_expected_head=${EXPECTED_HEAD}"
  echo "build_marker_expected_uv_lock_sha=${EXPECTED_UV_LOCK_SHA}"
  echo "build_marker_creation_slurm_job_id=${job_tag}"
  echo "build_marker_uv_sync_locked=complete"
  echo "python_bin=${python_bin}"
  echo "python_executable=${python_executable}"
  echo "python_version=${python_version}"
  echo "ray_version=${ray_version}"
  echo "venv_prefix=${venv_prefix}"
  echo "harness_shebang=${harness_shebang}"
  echo "installed_distributions_path=${installed_distributions_path}"
  echo "installed_distributions_sha256=${installed_distributions_sha256}"
  echo "installed_distributions_count=${installed_distributions_count}"
  echo "ray_address_unset=yes"
  echo "ray_node_tmp=${RAY_NODE_TMP}"
} > "${receipt_path}"

cd "${SOURCE_ROOT}"

run_ray_smoke() {
  local include_dashboard=$1 smoke_name=$2
  export RAY_TMPDIR="${RAY_NODE_TMP}/${smoke_name}"
  mkdir -p "${RAY_TMPDIR}"
  RAY_SMOKE_INCLUDE_DASHBOARD="${include_dashboard}" "${python_bin}" - <<'PY'
import os

import ray


@ray.remote
def ping() -> str:
    return "pong"


include_dashboard = os.environ["RAY_SMOKE_INCLUDE_DASHBOARD"] == "true"
context = ray.init(
    address="local",
    num_cpus=1,
    include_dashboard=include_dashboard,
    _temp_dir=os.environ["RAY_TMPDIR"],
)
try:
    assert ray.get(ping.remote()) == "pong"
    print(
        "RAY_LOCAL_SMOKE_PASS",
        f"dashboard={include_dashboard}",
        f"ray={ray.__version__}",
        f"session_dir={context.address_info.get('session_dir')}",
        flush=True,
    )
finally:
    ray.shutdown()
PY
}

stage=ray-core-smoke
run_ray_smoke false core
echo "ray_core_smoke=PASS" >> "${receipt_path}"

stage=ray-dashboard-smoke
run_ray_smoke true dashboard
echo "ray_dashboard_smoke=PASS" >> "${receipt_path}"

stage=focused-pytest
export TMPDIR="${RAY_NODE_TMP}"
export RAY_TMPDIR="${RAY_NODE_TMP}"
"${python_bin}" -m pytest -q "$@"
echo "focused_pytest=PASS" >> "${receipt_path}"

stage=complete
