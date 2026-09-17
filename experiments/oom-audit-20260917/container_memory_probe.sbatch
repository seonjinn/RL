#!/bin/bash
# One node, two sequential image-only probes; no model loading.
# Required exports: OLD_CONTAINER, CONTAINER, AUDIT_SOURCE.
set -euo pipefail
: "${OLD_CONTAINER:?older immutable image required}"
: "${CONTAINER:?new immutable image required}"
: "${AUDIT_SOURCE:?source checkout required}"
PROBE="$AUDIT_SOURCE/experiments/oom-audit-20260917/container_memory_probe.py"
python3 "$PROBE" host_before
for IMAGE_ARM in old new; do
  IMAGE_PATH="$OLD_CONTAINER"
  if [[ "$IMAGE_ARM" == new ]]; then IMAGE_PATH="$CONTAINER"; fi
  srun --nodes=1 --ntasks=1 --container-image="$IMAGE_PATH" \
    --container-mounts=/home:/home,/raid:/raid \
    --container-workdir=/opt/nemo-rl --no-container-mount-home \
    /opt/nemo_rl_venv/bin/python "$PROBE" "$IMAGE_ARM"
  python3 "$PROBE" "host_after_${IMAGE_ARM}"
done
