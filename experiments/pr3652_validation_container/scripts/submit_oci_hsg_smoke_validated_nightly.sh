#!/bin/bash

set -euo pipefail

if (( $# > 1 )); then
  echo "Usage: ACTION=test-only|submit $0 [ACTION]" >&2
  exit 2
fi

readonly ACTION=${1:-test-only}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
SCRIPT_ROOT=$(git -C "${script_dir}" rev-parse --show-toplevel)
readonly SCRIPT_ROOT
EXPECTED_TOOLING_SHA=$(git -C "${SCRIPT_ROOT}" rev-parse HEAD)
readonly EXPECTED_TOOLING_SHA
readonly BATCH_SCRIPT=${SCRIPT_ROOT}/experiments/pr3652_validation_container/scripts/oci_hsg_smoke_validated_nightly.sbatch
readonly BATCH_RELATIVE_PATH=experiments/pr3652_validation_container/scripts/oci_hsg_smoke_validated_nightly.sbatch
readonly LOG_DIRECTORY=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/container-transfer/nemo_rl_nightly_20260904_c6edc455/oci-smoke

[[ "${SCRIPT_ROOT}" = /* ]]
worktree_status=$(git -C "${SCRIPT_ROOT}" status --porcelain)
test -z "${worktree_status}"
test -x "${BATCH_SCRIPT}"
mkdir -p "${LOG_DIRECTORY}"

case ${ACTION} in
  test-only)
    git -C "${SCRIPT_ROOT}" show "${EXPECTED_TOOLING_SHA}:${BATCH_RELATIVE_PATH}" | sbatch \
      --test-only \
      --chdir="${SCRIPT_ROOT}" \
      --export="ALL,SCRIPT_ROOT=${SCRIPT_ROOT},EXPECTED_TOOLING_SHA=${EXPECTED_TOOLING_SHA}"
    ;;
  submit)
    git -C "${SCRIPT_ROOT}" show "${EXPECTED_TOOLING_SHA}:${BATCH_RELATIVE_PATH}" | sbatch \
      --chdir="${SCRIPT_ROOT}" \
      --export="ALL,SCRIPT_ROOT=${SCRIPT_ROOT},EXPECTED_TOOLING_SHA=${EXPECTED_TOOLING_SHA}"
    ;;
  *)
    echo "Unsupported ACTION: ${ACTION}" >&2
    exit 2
    ;;
esac
