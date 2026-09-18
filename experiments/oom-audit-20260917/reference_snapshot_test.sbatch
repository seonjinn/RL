#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
set -euo pipefail
: "${CONTAINER:?immutable image required}"
: "${AUDIT_SOURCE:?source checkout required}"
srun --container-image="${CONTAINER}" \
  --container-mounts=/home:/home,/lustre:/lustre,/raid:/raid \
  --container-workdir=/opt/nemo-rl --no-container-mount-home \
  /usr/local/bin/python-MegatronPolicyWorker \
  "${AUDIT_SOURCE}/experiments/oom-audit-20260917/test_reference_snapshot.py"
