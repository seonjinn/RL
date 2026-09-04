#!/bin/bash

set -euo pipefail

if (( $# != 2 )); then
  echo "Usage: $0 SEMANTIC_WORKTREE EXPECTED_REPO_SHA" >&2
  exit 2
fi

readonly SEMANTIC_WORKTREE=$1
readonly EXPECTED_REPO_SHA=$2
readonly MAIN_PYTHON=${MAIN_PYTHON:-/opt/nemo_rl_venv/bin/python}
readonly VLLM_WORKER_PYTHON=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python

test -d "${SEMANTIC_WORKTREE}"
test "$(git -C "${SEMANTIC_WORKTREE}" rev-parse HEAD)" = "${EXPECTED_REPO_SHA}"
test -z "$(git -C "${SEMANTIC_WORKTREE}" status --porcelain)"
test -x "${MAIN_PYTHON}"
test -x "${VLLM_WORKER_PYTHON}"

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
"${MAIN_PYTHON}" -m pytest tests/unit/precision_policy
