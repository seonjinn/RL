#!/bin/bash
# 1 node x4 GB200; no policy, no HybridEP, no inference quality claims.
set -euo pipefail
: "${CONTAINER:?immutable nightly required}"
: "${AUDIT_SOURCE:?source checkout required}"
: "${HF_HOME:?cached model config required}"
: "${AUDIT_SLEEP_MODE:?true or false required}"
export VLLM_LOGGING_LEVEL=DEBUG PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/opt/nemo-rl:/opt/nemo-rl/experiments/oom-audit-20260917
export XDG_CACHE_HOME="/raid/scratch/${USER}/vllm-init-${SLURM_JOB_ID}/cache"
export TMPDIR="/raid/scratch/${USER}/vllm-init-${SLURM_JOB_ID}/tmp"
export HF_MODULES_CACHE="/raid/scratch/${USER}/vllm-init-${SLURM_JOB_ID}/hf-modules"
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR" "$HF_MODULES_CACHE"
PROBE_ARGS=()
if [[ "${AUDIT_SLEEP_CYCLES:-0}" != 0 ]]; then
  PROBE_ARGS+=(--sleep-cycles "$AUDIT_SLEEP_CYCLES")
fi
if [[ "${AUDIT_INSPECT_CUMEM:-0}" == 1 ]]; then
  PROBE_ARGS+=(--inspect-cumem)
fi
if [[ "${AUDIT_PROFILE_BEFORE_GRAPHS:-0}" == 1 ]]; then
  PROBE_ARGS+=(--profile-before-graphs)
fi
srun --nodes=1 --ntasks=1 --container-image="$CONTAINER" \
  --container-mounts="/home:/home,/lustre:/lustre,/raid:/raid,${AUDIT_SOURCE}:/opt/nemo-rl" \
  --container-workdir=/opt/nemo-rl --no-container-mount-home \
  /usr/local/bin/python-VllmAsyncGenerationWorker \
  /opt/nemo-rl/experiments/oom-audit-20260917/vllm_init_probe.py \
  --sleep-mode "$AUDIT_SLEEP_MODE" "${PROBE_ARGS[@]}"
