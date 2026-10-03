#!/usr/bin/env bash

set -euo pipefail
: "${REPO:?Set REPO to the committed integration checkout}"
: "${EXPECTED_SOURCE_SHA:?Pin the diagnostic source}"
: "${CONTAINER:?Set the GPU-smoke-validated image}"
: "${RESULT_ROOT:?Set the durable diagnostic output directory}"
SRUN_BIN=${SRUN_BIN:-/cm/local/apps/slurm/25.11/bin/srun}
[[ "$(git -C "${REPO}" rev-parse HEAD)" == "${EXPECTED_SOURCE_SHA}" ]]
mkdir -p "${RESULT_ROOT}"

"${SRUN_BIN}" -N1 -n1 --container-image="${CONTAINER}" \
  --container-mounts="${REPO}:${REPO}:ro,/raid/scratch:/raid/scratch,${RESULT_ROOT}:${RESULT_ROOT}" \
  bash -s -- "${REPO}" "${RESULT_ROOT}" <<'BASH'
set -euo pipefail
repo=$1
output=$2
scratch="/raid/scratch/${USER}/ray-dashboard-${SLURM_JOB_ID}"
ray_root="/tmp/rd${SLURM_JOB_ID}"
export HOME="${scratch}/home" TMPDIR="${ray_root}"
export UV_CACHE_DIR="${scratch}/uv"
unset WANDB_API_KEY HF_TOKEN GH_TOKEN GITHUB_TOKEN SSH_AUTH_SOCK
mkdir -p "${HOME}" "${TMPDIR}"
save_logs() {
  tar -cf "${output}/ray-dashboard-${SLURM_JOB_ID}.tar" -C "${ray_root}" .
}
trap save_logs EXIT

script="${repo}/experiments/mxfp8_vllm029_fullperf_20260930/diagnose_ray_dashboard.py"
for role in driver vllm; do
  if [[ "${role}" == driver ]]; then
    python=/opt/nemo_rl_venv/bin/python
  else
    python=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python
  fi
  status=0
  timeout --kill-after=30s 240s "${python}" "${script}" \
    --role "${role}" --temp-dir "${ray_root}/${role}" || status=$?
  echo "${role}: exit ${status}"
done
BASH
