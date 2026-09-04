#!/bin/bash

set -euo pipefail

if (( $# != 3 )); then
  echo "Usage: $0 SEMANTIC_WORKTREE EXPECTED_REPO_SHA SCRATCH_DIRECTORY" >&2
  exit 2
fi

readonly SEMANTIC_WORKTREE=$1
readonly EXPECTED_REPO_SHA=$2
readonly SCRATCH_DIRECTORY=$3
readonly MAIN_PYTHON=/opt/nemo_rl_venv/bin/python
readonly VLLM_WORKER_PYTHON=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python

test -d "${SEMANTIC_WORKTREE}"
test "$(git -C "${SEMANTIC_WORKTREE}" rev-parse HEAD)" = "${EXPECTED_REPO_SHA}"
semantic_worktree_status=$(git -C "${SEMANTIC_WORKTREE}" status --porcelain)
test -z "${semantic_worktree_status}"
test -x "${MAIN_PYTHON}"
test -x "${VLLM_WORKER_PYTHON}"

case ${SCRATCH_DIRECTORY} in
  /raid/scratch/nemo-rl-validated-nightly/oci-smoke-[0-9]*) ;;
  *)
    echo 'SCRATCH_DIRECTORY must be the job-local OCI smoke scratch directory' >&2
    exit 2
    ;;
esac

test "${TMPDIR:-}" = "${SCRATCH_DIRECTORY}"
test "${PYTHONPYCACHEPREFIX:-}" = "${SCRATCH_DIRECTORY}/pycache"
test "${XDG_CACHE_HOME:-}" = "${SCRATCH_DIRECTORY}/xdg-cache"
test "${UV_CACHE_DIR:-}" = "${SCRATCH_DIRECTORY}/uv-cache"
test "${TORCHINDUCTOR_CACHE_DIR:-}" = "${SCRATCH_DIRECTORY}/torchinductor-cache"
test "${TRITON_CACHE_DIR:-}" = "${SCRATCH_DIRECTORY}/triton-cache"

"${MAIN_PYTHON}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")
if torch.version.cuda is None:
    raise RuntimeError("Torch does not report a CUDA build")
if torch.cuda.device_count() != 4:
    raise RuntimeError(f"Expected 4 visible CUDA devices, got {torch.cuda.device_count()}")

print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"cuda_device_count={torch.cuda.device_count()}")
PY

test "$("${VLLM_WORKER_PYTHON}" -c 'import vllm; print(vllm.__version__)')" = 0.25.1

cd "${SEMANTIC_WORKTREE}"
"${MAIN_PYTHON}" -m pytest \
  --basetemp="${TMPDIR}/pytest" \
  -p no:cacheprovider \
  tests/unit/precision_policy
test "$(git -C "${SEMANTIC_WORKTREE}" rev-parse HEAD)" = "${EXPECTED_REPO_SHA}"
semantic_worktree_status=$(git -C "${SEMANTIC_WORKTREE}" status --porcelain)
test -z "${semantic_worktree_status}"
