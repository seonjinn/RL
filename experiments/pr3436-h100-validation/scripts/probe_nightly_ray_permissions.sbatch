#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --job-name=nightly-ray-permissions

set -euo pipefail

: "${CONTAINER:?Set CONTAINER to an immutable nightly .sqsh path}"
: "${PROBE_DIR:?Set PROBE_DIR to a shared-storage output path}"
CONTAINER_MODE=${CONTAINER_MODE:-readonly}
export CONTAINER_MODE

case "${CONTAINER_MODE}" in
  readonly)
    container_mode_args=()
    ;;
  writable)
    container_mode_args=(--container-writable)
    ;;
  *)
    echo "CONTAINER_MODE must be readonly or writable" >&2
    exit 2
    ;;
esac

mkdir -p "${PROBE_DIR}"

srun \
  --nodes=1 \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --gpus-per-task=1 \
  --gpu-bind=single:1 \
  --no-container-mount-home \
  "${container_mode_args[@]}" \
  --container-image="${CONTAINER}" \
  --container-mounts=/lustre:/lustre \
  bash -lc '
    set -euo pipefail
    echo "container_mode=${CONTAINER_MODE}"
    id
    stat -c "root_mode_before=%a" /root
    if /opt/nemo_rl_venv/bin/python -c "import ray"; then
      echo "ray_import_before=pass"
    else
      echo "ray_import_before=fail"
      if [[ "${CONTAINER_MODE}" == "readonly" ]]; then
        exit 10
      fi
    fi
    if [[ "${CONTAINER_MODE}" == "writable" ]]; then
      chmod o+x /root
      stat -c "root_mode_after=%a" /root
    fi
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
