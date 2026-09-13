#!/bin/bash
set -euo pipefail

: "${TRACE:?Set the completed nsys-rep path}"
: "${OUTPUT:?Set durable summary directory}"
ROOT="/raid/scratch/${USER}/policy-trace-summary/${SLURM_JOB_ID}"
mkdir -p "${ROOT}" "${OUTPUT}"
export TMPDIR="${ROOT}"
nsys --version
nsys export --type sqlite --output "${ROOT}/policy.sqlite" "${TRACE}"
nsys stats --report cuda_gpu_kern_sum,nvtx_sum,nvtx_gpu_proj_sum,cuda_api_sum --format csv \
  --output "${ROOT}/summary" "${ROOT}/policy.sqlite"
for report in cuda_gpu_kern_sum nvtx_sum nvtx_gpu_proj_sum cuda_api_sum; do
  test -s "${ROOT}/summary_${report}.csv"
  cp "${ROOT}/summary_${report}.csv" "${OUTPUT}/${report}.csv"
done
printf 'trace=%s\njob=%s\n' "${TRACE}" "${SLURM_JOB_ID}" > "${OUTPUT}/provenance.txt"
# SQLite and temporary export artifacts intentionally remain node-local.
