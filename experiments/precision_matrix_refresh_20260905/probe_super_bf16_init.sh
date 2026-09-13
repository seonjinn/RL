#!/usr/bin/env bash
set -euo pipefail
: "${REPO:?}"
: "${CONTAINER:?}"
: "${RESULT_DIR:?}"
: "${HF_HOME_SOURCE:?}"
srun --ntasks=1 --kill-on-bad-exit=1 --wait=30 \
  --no-container-mount-home --container-image="${CONTAINER}" \
  --container-mounts="${REPO}:/source:ro,${RESULT_DIR}:/results,${HF_HOME_SOURCE}:/hf:ro,/raid/scratch:/raid/scratch" \
  --output="${RESULT_DIR}/init-%N.log" \
  bash -c '
set -euo pipefail
cd /source
ROOT=/raid/scratch/${SLURM_JOB_USER}/super-init-${SLURM_JOB_ID}
mkdir -p "$ROOT"
export PYTHONPATH=/source HF_HOME=/hf HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=$ROOT/pycache
export XDG_CACHE_HOME=$ROOT/cache VLLM_CACHE_ROOT=$ROOT/vllm
export TORCHINDUCTOR_CACHE_DIR=$ROOT/inductor TRITON_CACHE_DIR=$ROOT/triton
export HF_MODULES_CACHE=$ROOT/hf_modules RAY_TMPDIR=/raid/scratch/${SLURM_JOB_USER}/r${SLURM_JOB_ID}
export VLLM_LOGGING_LEVEL=DEBUG OMP_NUM_THREADS=4
PYTHON=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python
test -x "$PYTHON"
trap '\''tar -czf /results/ray-logs.tar.gz --ignore-failed-read "$RAY_TMPDIR"/session_*/logs 2>/dev/null || true'\'' EXIT
timeout --signal=TERM --kill-after=30s 25m "$PYTHON" \
  experiments/precision_matrix_refresh_20260905/probe_super_bf16_init.py
'
