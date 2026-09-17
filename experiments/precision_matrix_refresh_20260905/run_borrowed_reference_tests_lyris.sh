#!/usr/bin/env bash
set -euo pipefail

ACTION=${ACTION:-test-only}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-gb200}
SOURCE_ROOT=${SOURCE_ROOT:-/home/${USER}/RL-mxfp8-full-performance-20260917-v3}
CONTAINER=${CONTAINER:-/lustre/fsw/coreai_dlalgo_llm/users/${USER}/containers/nemo_rl_nightly.sqsh}
LOG_ROOT=${LOG_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/${USER}/precision-matrix-refresh-20260905/borrowed-reference-tests}
TEST_SCRIPT=${SOURCE_ROOT}/experiments/precision_matrix_refresh_20260905/run_borrowed_reference_tests_in_container.sh

case "${ACTION}" in
  test-only | submit) ;;
  *) echo "ACTION must be test-only or submit" >&2; exit 2 ;;
esac

for path in "${SOURCE_ROOT}" "${CONTAINER}" "${TEST_SCRIPT}"; do
  [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 2; }
done

mkdir -p "${LOG_ROOT}"

SBATCH_MODE=()
if [[ "${ACTION}" == test-only ]]; then
  SBATCH_MODE=(--test-only)
fi

exec sbatch "${SBATCH_MODE[@]}" \
  --nodes=1 \
  --exclusive \
  --account="${SLURM_ACCOUNT}" \
  --partition="${PARTITION}" \
  --time=00:30:00 \
  --job-name="${SLURM_ACCOUNT}-borrowed-ref.gb200-tests" \
  --output="${LOG_ROOT}/slurm-%j.out" \
  --wrap="srun --container-image=${CONTAINER} --container-mounts=/home:/home,/lustre:/lustre --container-remap-root bash ${TEST_SCRIPT} ${SOURCE_ROOT}"
