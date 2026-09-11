#!/usr/bin/env bash
# Source in a disposable job container before launching fresh Python workers.
set -euo pipefail

: "${SLURM_JOB_ID:?Run only in an allocated job container}"
: "${PYTHON_BIN:?}"
case "${PYTHON_BIN}" in
  /opt/ray_venvs/*/bin/python) ;;
  *) echo "Refusing to modify a non-container interpreter" >&2; return 1 ;;
esac
"${PYTHON_BIN}" -c 'import platform, torch; assert platform.machine() == "aarch64"; assert torch.version.cuda.startswith("13.")'

# Remove the old regular nccl package so it cannot hide the new namespace.
# This changes only the disposable container, not the immutable image or Torch/vLLM.
uv --no-config pip install --python "${PYTHON_BIN}" --no-deps --reinstall \
  --link-mode copy nccl4py==0.5.0 nccl-extensions==0.1.0 \
  cuda-core==1.0.0 cuda-pathfinder==1.5.4 cuda-bindings==13.0.3
M2N_BINDINGS_ROOT=$("${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
export M2N_BINDINGS_ROOT
nccl_library=$("${PYTHON_BIN}" -c '
import importlib.metadata
from pathlib import Path
dist = importlib.metadata.distribution("nvidia-nccl-cu13")
assert dist.version == "2.30.7", dist.version
paths = [Path(dist.locate_file(p)).resolve() for p in dist.files if p.name == "libnccl.so.2"]
assert len(paths) == 1, paths
print(paths[0])
')
[[ -f "${nccl_library}" ]] || return 1
export LD_PRELOAD=${nccl_library}${LD_PRELOAD:+:${LD_PRELOAD}}
