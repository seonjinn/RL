#!/usr/bin/env bash
# Source only in the transport experiment, before launching fresh Python workers.
set -euo pipefail

: "${LOCAL_ROOT:?}"
: "${PYTHON_BIN:?}"
abi=$("${PYTHON_BIN}" -c 'import sys, torch; assert torch.version.cuda.startswith("13."); print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
export M2N_OVERLAY_DIR=${LOCAL_ROOT}/m2n-0.1.0-core-0.5.0-nccl-2.30.7-cu13-${abi}
(
  flock 9
  if [[ ! -f "${M2N_OVERLAY_DIR}/.complete" ]]; then
    uv --no-config pip install --python "${PYTHON_BIN}" --target "${M2N_OVERLAY_DIR}" \
      --no-deps --link-mode copy \
      nccl4py==0.5.0 nccl-extensions==0.1.0 nvidia-nccl-cu13==2.30.7 \
      cuda-core==1.0.0 cuda-pathfinder==1.5.4 cuda-bindings==13.0.3
    touch "${M2N_OVERLAY_DIR}/.complete"
  fi
) 9>"${M2N_OVERLAY_DIR}.lock"
export PYTHONPATH=${M2N_OVERLAY_DIR}:${PYTHONPATH}
nccl_library=$("${PYTHON_BIN}" -c '
import importlib.metadata, os
from pathlib import Path
dist = importlib.metadata.distribution("nvidia-nccl-cu13")
paths = [Path(dist.locate_file(p)).resolve() for p in dist.files if p.name == "libnccl.so.2"]
assert len(paths) == 1, paths
assert paths[0].is_relative_to(Path(os.environ["M2N_OVERLAY_DIR"]).resolve())
print(paths[0])
')
if [[ ! -f "${nccl_library}" ]]; then
  echo "Expected isolated NCCL library is missing: ${nccl_library}" >&2
  return 1
fi
export LD_PRELOAD=${nccl_library}${LD_PRELOAD:+:${LD_PRELOAD}}
