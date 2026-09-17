#!/usr/bin/env bash
set -euo pipefail

ACTION=${ACTION:-test-only}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-batch}
SOURCE_ROOT=${SOURCE_ROOT:-/home/${USER}/RL-mxfp8-full-performance-20260917-v3}
CONTAINER=${CONTAINER:-/lustre/fsw/coreai_dlalgo_llm/users/${USER}/containers/nemo_rl_nightly.sqsh}
LOG_ROOT=${LOG_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/${USER}/precision-matrix-refresh-20260905/borrowed-reference-tests}
WORKER_PYTHON=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python

case "${ACTION}" in
  test-only | submit) ;;
  *) echo "ACTION must be test-only or submit" >&2; exit 2 ;;
esac

for path in "${SOURCE_ROOT}" "${CONTAINER}"; do
  [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 2; }
done

mkdir -p "${LOG_ROOT}"

read -r -d '' TEST_COMMAND <<EOF || true
set -euo pipefail
export PYTHONPATH=${SOURCE_ROOT}:${SOURCE_ROOT}/tests/unit/models/policy:${SOURCE_ROOT}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${SOURCE_ROOT}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM
cd ${SOURCE_ROOT}
${WORKER_PYTHON} tests/unit/models/policy/test_reference_snapshot.py
${WORKER_PYTHON} tests/unit/models/policy/test_reference_swap_lifecycle.py
${WORKER_PYTHON} tests/unit/models/policy/test_reference_swap_ddp_gpu.py
cd ${SOURCE_ROOT}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM
${WORKER_PYTHON} tests/unit_tests/distributed/test_borrowed_cpu_snapshot.py
EOF

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
  --wrap="srun --container-image=${CONTAINER} --container-mounts=/home:/home,/lustre:/lustre --container-remap-root bash -lc $(printf '%q' "${TEST_COMMAND}")"
