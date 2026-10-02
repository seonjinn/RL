#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0

set -euo pipefail

: "${CONTAINER:?Set CONTAINER to the tested NeMo-RL image}"
: "${SOURCE_ROOT:?Set SOURCE_ROOT to the NeMo-RL checkout under /home}"
: "${FLASHINFER_SOURCE_ROOT:?Set FLASHINFER_SOURCE_ROOT to FlashInfer v0.6.18 source under /home}"
: "${SRUN_BIN:=/cm/local/apps/slurm/25.11/bin/srun}"

if [[ ! -x "${SRUN_BIN}" ]]; then
  echo "srun executable not found: ${SRUN_BIN}" >&2
  exit 1
fi

"${SRUN_BIN}" \
  --ntasks=1 \
  --container-image="${CONTAINER}" \
  --container-mounts=/home:/home,/lustre:/lustre \
  --container-remap-root \
  bash -lc \
  "cd '${SOURCE_ROOT}' && \
  FLASHINFER_SOURCE_ROOT='${FLASHINFER_SOURCE_ROOT}' \
  /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python \
  experiments/mxfp8_vllm029_fullperf_20260930/test_flashinfer_trtllm_mxfp8_qwen3_shapes.py"
