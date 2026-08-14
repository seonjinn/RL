#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --job-name=nightly-ray-permissions

set -euo pipefail

: "${CONTAINER:?Set CONTAINER to an immutable nightly .sqsh path}"
: "${PROBE_DIR:?Set PROBE_DIR to a shared-storage output path}"

mkdir -p "${PROBE_DIR}"

srun \
  --nodes=1 \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --gpus-per-task=1 \
  --gpu-bind=single:1 \
  --no-container-mount-home \
  --container-writable \
  --container-image="${CONTAINER}" \
  --container-mounts=/lustre:/lustre \
  bash -lc '
    set -euo pipefail
    stat -c "root_mode_before=%a" /root
    if /opt/nemo_rl_venv/bin/python -c "import ray"; then
      echo "ray_import_before=pass"
    else
      echo "ray_import_before=fail"
    fi
    chmod o+x /root
    stat -c "root_mode_after=%a" /root
    /opt/nemo_rl_venv/bin/python - <<"PY"
import ray
import requests
import urllib3

print(f"ray={ray.__version__}")
print(f"requests={requests.__version__}")
print(f"urllib3={urllib3.__version__}")
print("NIGHTLY_RAY_PERMISSION_PROBE_PASS")
PY
  ' | tee "${PROBE_DIR}/probe.log"
