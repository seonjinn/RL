#!/usr/bin/env bash
# shellcheck disable=SC2016

set -euo pipefail

HARNESS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ray_then_pytest.sh"

test -f "${HARNESS}"
bash -n "${HARNESS}"

required_contracts=(
  ': "${EXPECTED_HEAD:?'
  ': "${EXPECTED_UV_LOCK_SHA:?'
  'unset RAY_ADDRESS'
  'RAY_NODE_TMP="/tmp/nrr${job_tag}"'
  'job tag must contain decimal digits only'
  'address="local"'
  'run_ray_smoke false core'
  'run_ray_smoke true dashboard'
  '_temp_dir=os.environ["RAY_TMPDIR"]'
  'dashboard_MetricsHead.log'
  'dashboard_MetricsHead.out'
  'dashboard_MetricsHead.err'
  'recursive_clean=yes'
  'python_executable='
  'python_version='
  'ray_version='
  'venv_prefix='
  'harness_shebang='
  'DURABLE_RESULT_ROOT must resolve below /lustre'
  'SOURCE_ROOT must resolve below /home'
  'UV_PROJECT_ENVIRONMENT must resolve below /raid/scratch'
  'status --porcelain --untracked-files=all --ignore-submodules=none'
  'source_status="$(git -C'
  'submodule_status="$(git -C'
  'sha256sum "${SOURCE_ROOT}/uv.lock"'
  '.nemo-rl-build-marker'
  'uv_sync_locked=complete'
  'creation_slurm_job_id='
  'installed-distributions-'
  'installed_distributions_sha256='
  'installed_distributions_count='
  'LISTING_TRUNCATED_AT=200'
  '"${python_bin}" -m pytest -q "$@"'
)

for contract in "${required_contracts[@]}"; do
  grep -Fq "${contract}" "${HARNESS}"
done

trap_line="$(grep -n '^trap finish EXIT$' "${HARNESS}" | cut -d: -f1)"
first_required_env_line="$(grep -n '^: "${SOURCE_ROOT:?' "${HARNESS}" | cut -d: -f1)"
test "${trap_line}" -lt "${first_required_env_line}"

if grep -Eq '(^|[[:space:]])(sbatch|srun|scancel)([[:space:]]|$)' "${HARNESS}"; then
  echo "scheduler mutation command found in diagnostic harness" >&2
  exit 1
fi

fixture_root="$(mktemp -d)"
fixture_root="$(realpath "${fixture_root}")"
fixture_job_id="$((9000000 + $$))"
fixture_tmp="/tmp/nrr${fixture_job_id}"
fixture_harness="${fixture_root}/ray_then_pytest.sh"
fixture_source="${fixture_root}/home/source"
fixture_durable="${fixture_root}/lustre/results"
fixture_venv="${fixture_root}/raid/scratch/venv"
fixture_bin="${fixture_root}/bin"

cleanup_fixture() {
  if [[ -n "${marker_tmp:-}" ]]; then
    rm -rf -- "${marker_tmp}"
  fi
  rm -rf -- "${fixture_tmp}" "${fixture_root}"
}
trap cleanup_fixture EXIT

mkdir -p \
  "${fixture_source}" \
  "${fixture_durable}" \
  "${fixture_venv}/bin" \
  "${fixture_bin}"
printf 'fixture-lock\n' > "${fixture_source}/uv.lock"
fixture_lock_sha="$(shasum -a 256 "${fixture_source}/uv.lock" | awk '{print $1}')"

cat > "${fixture_bin}/git" <<'SH'
#!/usr/bin/env bash
set -u
case "${3:-}" in
  rev-parse)
    echo "${FAKE_GIT_HEAD}"
    ;;
  status)
    printf '%s' "${FAKE_GIT_STATUS_OUTPUT:-}"
    exit "${FAKE_GIT_STATUS_RC:-0}"
    ;;
  submodule)
    printf '%s' "${FAKE_GIT_SUBMODULE_OUTPUT:-}"
    exit "${FAKE_GIT_SUBMODULE_RC:-0}"
    ;;
  *)
    echo "unexpected fake git invocation: $*" >&2
    exit 91
    ;;
esac
SH
cat > "${fixture_bin}/sha256sum" <<'SH'
#!/usr/bin/env bash
exec shasum -a 256 "$@"
SH
cat > "${fixture_venv}/bin/python" <<'SH'
#!/usr/bin/env bash
echo "fixture python must not run in provenance-failure contracts" >&2
exit 92
SH
chmod +x \
  "${fixture_bin}/git" \
  "${fixture_bin}/sha256sum" \
  "${fixture_venv}/bin/python"

sed \
  -e "s|/lustre/|${fixture_root}/lustre/|g" \
  -e "s|/home/|${fixture_root}/home/|g" \
  -e "s|/raid/scratch/|${fixture_root}/raid/scratch/|g" \
  "${HARNESS}" > "${fixture_harness}"

prepare_failure_logs() {
  local job_id=$1
  local ray_tmp="/tmp/nrr${job_id}"
  mkdir -p "${ray_tmp}/dashboard/session_latest/logs"
  printf 'metrics-head-failure\n' \
    > "${ray_tmp}/dashboard/session_latest/logs/dashboard_MetricsHead.err"
  printf 'dashboard-failure\n' \
    > "${ray_tmp}/dashboard/session_latest/logs/dashboard.err"
}

status_job_id="${fixture_job_id}"
status_tmp="/tmp/nrr${status_job_id}"
status_failure_root="${fixture_durable}/ray-failure-${status_job_id}"
prepare_failure_logs "${status_job_id}"

set +e
PATH="${fixture_bin}:${PATH}" \
FAKE_GIT_HEAD=0000000000000000000000000000000000000000 \
FAKE_GIT_STATUS_RC=7 \
SLURM_JOB_ID="${status_job_id}" \
SOURCE_ROOT="${fixture_source}" \
UV_PROJECT_ENVIRONMENT="${fixture_venv}" \
DURABLE_RESULT_ROOT="${fixture_durable}" \
EXPECTED_HEAD=0000000000000000000000000000000000000000 \
EXPECTED_UV_LOCK_SHA="${fixture_lock_sha}" \
  bash "${fixture_harness}" tests/unit/example.py >/dev/null 2>&1
status_rc=$?
set -e

test "${status_rc}" -eq 7
test -f "${status_failure_root}/dashboard-dashboard_MetricsHead.err"
grep -Fq 'metrics-head-failure' \
  "${status_failure_root}/dashboard-dashboard_MetricsHead.err"
test -f "${status_failure_root}/session-files.txt"
grep -Fq 'dashboard_MetricsHead.err' "${status_failure_root}/session-files.txt"
test ! -e "${status_tmp}"

marker_job_id="$((fixture_job_id + 1))"
marker_tmp="/tmp/nrr${marker_job_id}"
marker_failure_root="${fixture_durable}/ray-failure-${marker_job_id}"
prepare_failure_logs "${marker_job_id}"
cat > "${fixture_venv}/.nemo-rl-build-marker" <<EOF
expected_head=1111111111111111111111111111111111111111
expected_uv_lock_sha=${fixture_lock_sha}
creation_slurm_job_id=${marker_job_id}
uv_sync_locked=complete
EOF

set +e
PATH="${fixture_bin}:${PATH}" \
FAKE_GIT_HEAD=0000000000000000000000000000000000000000 \
FAKE_GIT_STATUS_RC=0 \
FAKE_GIT_SUBMODULE_RC=0 \
SLURM_JOB_ID="${marker_job_id}" \
SOURCE_ROOT="${fixture_source}" \
UV_PROJECT_ENVIRONMENT="${fixture_venv}" \
DURABLE_RESULT_ROOT="${fixture_durable}" \
EXPECTED_HEAD=0000000000000000000000000000000000000000 \
EXPECTED_UV_LOCK_SHA="${fixture_lock_sha}" \
  bash "${fixture_harness}" tests/unit/example.py >/dev/null 2>&1
marker_rc=$?
set -e

test "${marker_rc}" -ne 0
test -f "${marker_failure_root}/environment.txt"
grep -Fq 'stage=venv-build-marker' "${marker_failure_root}/environment.txt"
test ! -e "${marker_tmp}"

producer_marker_job_id="$((fixture_job_id + 2))"
marker_tmp="/tmp/nrr${producer_marker_job_id}"
producer_marker_failure_root="${fixture_durable}/ray-failure-${producer_marker_job_id}"
producer_cudnn_lib="${fixture_venv}/lib/python3.13/site-packages/nvidia/cudnn/lib"
prepare_failure_logs "${producer_marker_job_id}"
mkdir -p "${producer_cudnn_lib}"
touch "${producer_cudnn_lib}/libcudnn.so.9"
cat > "${fixture_venv}/.nemo-rl-build-marker" <<EOF
expected_head=0000000000000000000000000000000000000000
expected_uv_lock_sha=${fixture_lock_sha}
creation_slurm_job_id=${producer_marker_job_id}
cudnn_lib=${producer_cudnn_lib}
uv_sync_locked=complete
EOF

set +e
PATH="${fixture_bin}:${PATH}" \
FAKE_GIT_HEAD=0000000000000000000000000000000000000000 \
FAKE_GIT_STATUS_RC=0 \
FAKE_GIT_SUBMODULE_RC=0 \
SLURM_JOB_ID="${producer_marker_job_id}" \
SOURCE_ROOT="${fixture_source}" \
UV_PROJECT_ENVIRONMENT="${fixture_venv}" \
DURABLE_RESULT_ROOT="${fixture_durable}" \
EXPECTED_HEAD=0000000000000000000000000000000000000000 \
EXPECTED_UV_LOCK_SHA="${fixture_lock_sha}" \
  bash "${fixture_harness}" tests/unit/example.py >/dev/null 2>&1
producer_marker_rc=$?
set -e

test "${producer_marker_rc}" -eq 92
test -f "${producer_marker_failure_root}/environment.txt"
grep -Fq 'stage=python-receipt' \
  "${producer_marker_failure_root}/environment.txt"
test ! -e "${marker_tmp}"

echo "RAY_THEN_PYTEST_HARNESS_CONTRACT_PASS"
