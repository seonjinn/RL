#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0

set -euo pipefail

: "${CONTAINER:?Set CONTAINER to the tested NeMo-RL image}"
: "${SOURCE_ROOT:?Set SOURCE_ROOT to the NeMo-RL checkout under /home}"

srun \
  --ntasks=1 \
  --container-image="${CONTAINER}" \
  --container-mounts=/home:/home,/lustre:/lustre \
  --container-remap-root \
  bash -lc \
  "cd '${SOURCE_ROOT}' && \
  /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python \
  -m pytest -q \
  experiments/mxfp8_vllm029_fullperf_20260930/test_flashinfer_mxfp8_prompt_shapes.py"
