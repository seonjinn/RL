#!/usr/bin/env bash
set -euo pipefail
: "${REPO:?Set the committed experiment checkout}"
: "${CONTAINER:?Set the certified vLLM 0.29 image}"
: "${LIGHTNING_SNAPSHOT:?Set the existing complete local model snapshot}"
SRUN_BIN=${SRUN_BIN:-/cm/local/apps/slurm/current/bin/srun}
SCRATCH=/raid/scratch/${USER}/lightning-preflight-${SLURM_JOB_ID}
"${SRUN_BIN}" --nodes=1 --ntasks=1 --container-image="${CONTAINER}" \
  --container-mounts="${REPO}:${REPO}:ro,/raid/scratch:/raid/scratch,/lustre:/lustre:ro" \
  bash -c "set -euo pipefail; mkdir -p '${SCRATCH}'; cd '${REPO}'; \
    export HF_HOME='${SCRATCH}/hf' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; \
    export TMPDIR='${SCRATCH}' PYTHONPYCACHEPREFIX='${SCRATCH}/pycache'; \
    export PERFORMANCE_PROFILE=runtime-aligned LIGHTNING_SNAPSHOT='${LIGHTNING_SNAPSHOT}'; \
    /opt/nemo_rl_venv/bin/python experiments/mxfp8_vllm029_fullperf_20260930/test_lightning_performance.py"
