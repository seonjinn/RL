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
wheel_options=()
if [[ -n "${M2N_WHEELHOUSE:-}" ]]; then
  LOCAL_WHEELS=${LOCAL_ROOT:?}/m2n-wheels-cp313-0.1.0-0.5.0
  mkdir -p "${LOCAL_WHEELS}"
  while read -r _hash wheel; do
    cp "${M2N_WHEELHOUSE}/${wheel}" "${LOCAL_WHEELS}/${wheel}"
  done < experiments/native_mxfp8_m2n/wheels.sha256
  (
    cd "${LOCAL_WHEELS}"
    sha256sum --check "${SOURCE_DIR}/experiments/native_mxfp8_m2n/wheels.sha256"
  )
  wheel_options=(--offline --no-index --find-links "${LOCAL_WHEELS}")
fi
timeout --kill-after=15s 180s uv --no-config pip install \
  --python "${PYTHON_BIN}" --no-deps --reinstall --link-mode copy \
  "${wheel_options[@]}" nccl4py==0.5.0 nccl-extensions==0.1.0
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
export M2N_NCCL_LIBRARY=${nccl_library}
export LD_PRELOAD=${nccl_library}${LD_PRELOAD:+:${LD_PRELOAD}}
