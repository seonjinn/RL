#!/usr/bin/env bash

set -euo pipefail

: "${SLURM_JOB_ID:?set SLURM_JOB_ID}"
: "${TASK5A_STAGE:?set TASK5A_STAGE}"
: "${DURABLE_RESULT_ROOT:?set DURABLE_RESULT_ROOT}"

case "${SLURM_JOB_ID}" in
  ''|*[!0-9]*) echo 'SLURM_JOB_ID must contain decimal digits only' >&2; exit 2 ;;
esac
case "${TASK5A_STAGE}" in
  ''|*[!a-zA-Z0-9_.-]*) echo 'TASK5A_STAGE contains unsafe characters' >&2; exit 2 ;;
esac

test -d "${DURABLE_RESULT_ROOT}"
DURABLE_RESULT_ROOT="$(realpath "${DURABLE_RESULT_ROOT}")"
if [[ "${TASK5A_RAY_ENV_ALLOW_TEST_PATHS:-0}" != 1 ]]; then
  case "${DURABLE_RESULT_ROOT}" in
    /lustre/*) ;;
    *) echo 'DURABLE_RESULT_ROOT must resolve below /lustre' >&2; exit 2 ;;
  esac
fi

TASK5A_RAY_ROOT="/tmp/nr${SLURM_JOB_ID}"
test ! -e "${TASK5A_RAY_ROOT}"
mkdir -p "${TASK5A_RAY_ROOT}"
TASK5A_RAY_ROOT_REAL="$(realpath "${TASK5A_RAY_ROOT}")"
test "${TASK5A_RAY_ROOT_REAL}" = "$(realpath /tmp)/nr${SLURM_JOB_ID}"
socket_budget_sample="${TASK5A_RAY_ROOT}/ray/session_2026-08-22_21-57-37_290492_1294011/sockets/plasma_store"
if ((${#socket_budget_sample} >= 108)); then
  echo "Ray AF_UNIX socket budget exceeded: ${socket_budget_sample}" >&2
  exit 2
fi

unset RAY_ADDRESS
export TMPDIR="${TASK5A_RAY_ROOT}"
export RAY_TMPDIR="${TASK5A_RAY_ROOT}"
task5a_ray_tmp_safe=yes

task5a_ray_env_finish() {
  local rc=$?
  local bundle listed log_name log_path scope session_file
  trap - EXIT
  set +e

  if ((rc != 0)); then
    bundle="${DURABLE_RESULT_ROOT}/ray-failure-${SLURM_JOB_ID}-${TASK5A_STAGE}"
    mkdir -p "${bundle}"
    {
      echo "exit_code=${rc}"
      echo "stage=${TASK5A_STAGE}"
      echo "tmpdir=${TMPDIR}"
      echo "ray_tmpdir=${RAY_TMPDIR}"
      echo "ray_tmpdir_real=${TASK5A_RAY_ROOT_REAL}"
      echo 'ray_address_unset=yes'
      echo "socket_budget_sample=${socket_budget_sample}"
      echo "socket_budget_length=${#socket_budget_sample}"
      echo "pwd=$(pwd)"
    } > "${bundle}/environment.txt"

    for scope in core dashboard pytest ray; do
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
        log_path="$(find "${TASK5A_RAY_ROOT}/${scope}" -maxdepth 5 -type f -name "${log_name}" -print -quit 2>/dev/null)"
        if [[ -n "${log_path}" ]]; then
          cp -L "${log_path}" "${bundle}/${scope}-${log_name}"
        fi
      done
    done

    listed=0
    : > "${bundle}/session-files.txt"
    while IFS= read -r session_file; do
      printf '%s\n' "${session_file}" >> "${bundle}/session-files.txt"
      listed=$((listed + 1))
      if ((listed >= 200)); then
        echo 'LISTING_TRUNCATED_AT=200' >> "${bundle}/session-files.txt"
        break
      fi
    done < <(find "${TASK5A_RAY_ROOT}" -maxdepth 6 -type f 2>/dev/null)
  fi

  if [[ "${task5a_ray_tmp_safe}" == yes ]]; then
    rm -rf -- "${TASK5A_RAY_ROOT}"
  fi
  exit "${rc}"
}
trap task5a_ray_env_finish EXIT

task5a_ray_local_smoke() {
  local python_bin=$1
  test -x "${python_bin}"
  "${python_bin}" - <<'PY'
import os

import ray


@ray.remote
def ping() -> str:
    return "pong"


context = ray.init(
    address="local",
    include_dashboard=False,
    num_cpus=1,
    _temp_dir=os.environ["RAY_TMPDIR"],
)
try:
    assert ray.get(ping.remote()) == "pong"
    print(
        "TASK5A_RAY_LOCAL_SMOKE_PASS",
        f"ray={ray.__version__}",
        f"session_dir={context.address_info.get('session_dir')}",
    )
finally:
    ray.shutdown()
PY
}
