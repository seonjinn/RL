#!/usr/bin/env bash
# shellcheck disable=SC2016

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRAPPER="${SCRIPT_DIR}/task5a-linux-gate-shorttmp.sbatch"

test -f "${WRAPPER}"
bash -n "${WRAPPER}"

required_contracts=(
  '#SBATCH --account=nemotron_n3_post'
  '#SBATCH --partition=batch'
  '#SBATCH --qos=normal'
  '#SBATCH --nodes=1'
  '#SBATCH --gpus-per-node=4'
  '#SBATCH --time=04:00:00'
  'EXPECTED_SHA=1b9ec8b3b12f57d7b691a68ba220a7ceb22c7af1'
  'EXPECTED_RAY_ENV_SHA=2b31f7bde7cc0dbe2a6e97849212071e7da17e89c89b27ca6a72a0926a651d87'
  'RAY_ENV_HELPER=/home/sna/task5a-ray-shorttmp-harness/experiments/oci_task5a_ray_shorttmp_20260822/task5a_ray_env.sh'
  'test "$(sha256sum "${RAY_ENV_HELPER}"'
  'export DURABLE_RESULT_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/adaptive-task5a-1b9ec8b3-gate'
  'source /home/sna/task5a-ray-shorttmp-harness/experiments/oci_task5a_ray_shorttmp_20260822/task5a_ray_env.sh'
  'export TASK5A_STAGE=bootstrap-mcore-transformer-engine'
  'export TASK5A_STAGE=static-10-files'
  'export TASK5A_STAGE=focused-mcore-only'
  'export TASK5A_STAGE=related-policy-wrappers'
  'export TASK5A_STAGE=two-rank-consensus'
  'task5a_ray_local_smoke \"\${UV_PROJECT_ENVIRONMENT}/bin/python\"'
  'tests/unit/models/megatron/test_draft_optimizer.py'
  'tests/unit/models/megatron/test_draft_optimizer_suspension.py'
  'tests/unit/models/megatron/test_megatron_worker.py'
  'tests/unit/models/megatron/test_megatron_split_state.py'
  'tests/unit/models/policy/test_split_api_wrappers.py'
  'tests/unit/distributed/test_draft_cadence_consensus.py'
)

for contract in "${required_contracts[@]}"; do
  grep -Fq -- "${contract}" "${WRAPPER}"
done

test "$(grep -Fc 'source /home/sna/task5a-ray-shorttmp-harness/experiments/oci_task5a_ray_shorttmp_20260822/task5a_ray_env.sh' "${WRAPPER}")" -eq 1
test "$(grep -Fc 'export TASK5A_STAGE=' "${WRAPPER}")" -eq 5
test "$(grep -Fc 'task5a_ray_local_smoke \"\${UV_PROJECT_ENVIRONMENT}/bin/python\"' "${WRAPPER}")" -eq 1

if grep -Eq 'export (TMPDIR|RAY_TMPDIR)=/raid/scratch' "${WRAPPER}"; then
  echo 'long Ray/TMP root reintroduced into Task5A wrapper' >&2
  exit 1
fi

if grep -Eq -- '(^|[[:space:]])(scancel)([[:space:]]|$)' "${WRAPPER}"; then
  echo 'scheduler cancellation command found in wrapper' >&2
  exit 1
fi

echo TASK5A_SHORTTMP_WRAPPER_CONTRACT_PASS
