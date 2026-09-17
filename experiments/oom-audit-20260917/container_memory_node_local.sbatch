#!/bin/bash
# One-node matched-image probe; only Enroot storage placement changes.
set -euo pipefail
: "${CONTAINER:?immutable image required}"
: "${AUDIT_SOURCE:?source checkout required}"
: "${SLURM_JOB_ID:?submit through sbatch}"
export ENROOT_DATA_PATH="/raid/scratch/${USER}/oom-container-${SLURM_JOB_ID}/data"
export ENROOT_CACHE_PATH="/raid/scratch/${USER}/oom-container-${SLURM_JOB_ID}/cache"
export ENROOT_TEMP_PATH="/raid/scratch/${USER}/oom-container-${SLURM_JOB_ID}/tmp"
export ENROOT_RUNTIME_PATH="/raid/scratch/${USER}/oom-container-${SLURM_JOB_ID}/runtime"
mkdir -p "$ENROOT_DATA_PATH" "$ENROOT_CACHE_PATH" "$ENROOT_TEMP_PATH" "$ENROOT_RUNTIME_PATH"
PROBE="$AUDIT_SOURCE/experiments/oom-audit-20260917/container_memory_probe.py"
python3 "$PROBE" host_before_node_local
srun --nodes=1 --ntasks=1 --container-image="$CONTAINER" \
  --container-mounts=/home:/home,/raid:/raid \
  --container-workdir=/opt/nemo-rl --no-container-mount-home \
  /opt/nemo_rl_venv/bin/python "$PROBE" new_node_local
python3 "$PROBE" host_after_node_local
