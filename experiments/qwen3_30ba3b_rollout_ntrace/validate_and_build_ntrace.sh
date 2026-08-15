#!/usr/bin/env bash

set -euo pipefail

: "${NEMO_SOURCE:?set NEMO_SOURCE to the NeMo-RL source tree}"
: "${NTRACE_SOURCE:?set NTRACE_SOURCE to the pinned ntrace source tree}"
: "${NTRACE_RUNTIME:?set NTRACE_RUNTIME to a new shared install target}"

cd "${NEMO_SOURCE}"
export PYTHONPATH="${NEMO_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"

/opt/nemo_rl_venv/bin/python - <<'PY'
import flashinfer
import torch
import vllm

print(
    f"torch={torch.__version__} vllm={vllm.__version__} "
    f"flashinfer={flashinfer.__version__}",
    flush=True,
)
PY

/opt/nemo_rl_venv/bin/python -m pytest -q \
  tests/unit/models/generation/test_rollout_profiler.py \
  tests/unit/models/generation/test_vllm_rollout_profiler.py \
  tests/unit/models/generation/test_vllm_fp8_quantization.py

NTRACE_INSTALL_SOURCE="${NTRACE_SOURCE}" \
NTRACE_INSTALL_TARGET="${NTRACE_RUNTIME}" \
NTRACE_INSTALL_PYTHON=/opt/nemo_rl_venv/bin/python \
  bash "${NTRACE_SOURCE}/scripts/ntrace_nemo_rl_install_target.sh"

PYTHONPATH="${NTRACE_RUNTIME}:${PYTHONPATH}" \
  /opt/nemo_rl_venv/bin/python - <<'PY'
import ntrace
from ntrace.backends import get_backend, selected_backend_name

assert hasattr(ntrace.NemoRLRolloutTraceController, "close")
print(
    f"ntrace={ntrace.__version__} backend={selected_backend_name()} "
    f"backend_type={type(get_backend()).__name__} close_hook=present",
    flush=True,
)
PY
