#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
set -euo pipefail
: "${CONTAINER:?immutable image required}"
: "${AUDIT_SOURCE:?host checkout required}"
unset PYTHONPATH UV_NO_SYNC NRL_FORCE_REBUILD_VENVS PYTHONOPTIMIZE
export AUDIT_SOURCE
srun --container-image="${CONTAINER}" \
  --container-mounts=/home:/home,/lustre:/lustre,/raid:/raid \
  --container-workdir=/opt/nemo-rl --no-container-mount-home \
  bash -c '
    set -euo pipefail
    git -C /opt/nemo-rl rev-parse HEAD
    /opt/nemo_rl_venv/bin/python "$AUDIT_SOURCE/experiments/oom-audit-20260917/smoke.py" driver
    /usr/local/bin/python-MegatronPolicyWorker "$AUDIT_SOURCE/experiments/oom-audit-20260917/smoke.py" policy
    /usr/local/bin/python-VllmAsyncGenerationWorker "$AUDIT_SOURCE/experiments/oom-audit-20260917/smoke.py" generation
  '
