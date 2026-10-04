#!/usr/bin/env bash
set -euo pipefail
: "${REPO:?Set the committed diagnostic worktree}"
: "${CONTAINER:?Set the certified runtime-aligned image}"
SRUN_BIN=${SRUN_BIN:-/cm/local/apps/slurm/current/bin/srun}
SCRATCH=/raid/scratch/${USER}/storage-unit-${SLURM_JOB_ID}
mkdir -p "${SCRATCH}"
"${SRUN_BIN}" --nodes=1 --ntasks=1 --container-image="${CONTAINER}" \
  --container-mounts="${REPO}:${REPO}:ro,/raid/scratch:/raid/scratch" \
  bash -lc "set -euo pipefail; cd '${REPO}'; \
    export TMPDIR='${SCRATCH}' PYTHONPYCACHEPREFIX='${SCRATCH}/pycache'; \
    export PYTHONDONTWRITEBYTECODE=1; \
    /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python \
      -c 'import sys; sys.path.append(\"/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/lib/python3.13/site-packages\"); import pytest; raise SystemExit(pytest.main(sys.argv[1:]))' \
      --noconftest -q --cache-clear -o cache_dir='${SCRATCH}/pytest' \
      experiments/mxfp8_vllm029_fullperf_20260930/test_storage_inventory.py"
