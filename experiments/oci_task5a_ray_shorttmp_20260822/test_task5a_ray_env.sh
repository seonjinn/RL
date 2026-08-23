#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELPER="${SCRIPT_DIR}/task5a_ray_env.sh"
fixture_root="$(mktemp -d)"

cleanup() {
  rm -rf -- "${fixture_root}"
}
trap cleanup EXIT

old_socket='/raid/scratch/root/6459723/adaptive-task5a/tmp/ray/session_2026-08-22_21-57-37_290492_1294011/sockets/plasma_store'
if ((${#old_socket} <= 107)); then
  echo 'fixture must reproduce the AF_UNIX socket overflow' >&2
  exit 1
fi

durable_root="${fixture_root}/lustre/results"
mkdir -p "${durable_root}"
failure_job_id=991234
failure_ray_root="/tmp/nr${failure_job_id}"
test ! -e "${failure_ray_root}"

set +e
HELPER="${HELPER}" \
DURABLE_RESULT_ROOT="${durable_root}" \
TASK5A_RAY_ENV_ALLOW_TEST_PATHS=1 \
TASK5A_STAGE=focused-mcore-only \
SLURM_JOB_ID="${failure_job_id}" \
RAY_ADDRESS=auto \
  bash -c '
    set -euo pipefail
    source "${HELPER}"
    test "${TMPDIR}" = "/tmp/nr${SLURM_JOB_ID}"
    test "${RAY_TMPDIR}" = "/tmp/nr${SLURM_JOB_ID}"
    test "$(realpath "${RAY_TMPDIR}")" = "$(realpath /tmp)/nr${SLURM_JOB_ID}"
    test -z "${RAY_ADDRESS+x}"
    socket_path="${RAY_TMPDIR}/ray/session_2026-08-22_21-57-37_290492_1294011/sockets/plasma_store"
    test "${#socket_path}" -lt 108
    logs="${RAY_TMPDIR}/ray/session_fixture/logs"
    mkdir -p "${logs}"
    printf "%s\n" metrics-head-fixture > "${logs}/dashboard_MetricsHead.log"
    false
  '
failure_rc=$?
set -e

test "${failure_rc}" -ne 0
test ! -e "${failure_ray_root}"
failure_bundle="${durable_root}/ray-failure-${failure_job_id}-focused-mcore-only"
test -f "${failure_bundle}/environment.txt"
grep -Fxq "exit_code=${failure_rc}" "${failure_bundle}/environment.txt"
grep -Fxq 'stage=focused-mcore-only' "${failure_bundle}/environment.txt"
grep -Fxq "ray_tmpdir=${failure_ray_root}" "${failure_bundle}/environment.txt"
grep -Fxq 'ray_address_unset=yes' "${failure_bundle}/environment.txt"
test -f "${failure_bundle}/ray-dashboard_MetricsHead.log"
grep -Fxq metrics-head-fixture "${failure_bundle}/ray-dashboard_MetricsHead.log"
grep -Fq 'dashboard_MetricsHead.log' "${failure_bundle}/session-files.txt"

success_job_id=991235
success_ray_root="/tmp/nr${success_job_id}"
test ! -e "${success_ray_root}"
HELPER="${HELPER}" \
DURABLE_RESULT_ROOT="${durable_root}" \
TASK5A_RAY_ENV_ALLOW_TEST_PATHS=1 \
TASK5A_STAGE=bootstrap \
SLURM_JOB_ID="${success_job_id}" \
RAY_ADDRESS=auto \
  bash -c '
    set -euo pipefail
    source "${HELPER}"
    test -z "${RAY_ADDRESS+x}"
    test "${TMPDIR}" = "/tmp/nr${SLURM_JOB_ID}"
  '
test ! -e "${success_ray_root}"
test ! -e "${durable_root}/ray-failure-${success_job_id}-bootstrap"

echo TASK5A_RAY_ENV_CONTRACT_PASS
