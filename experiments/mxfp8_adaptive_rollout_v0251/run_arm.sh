#!/usr/bin/env bash
set -euo pipefail

ARM=${1:?usage: run_arm.sh baseline|adaptive}
ROOT=${NEMO_RL_REPO_ROOT:?set NEMO_RL_REPO_ROOT}
VLLM_SOURCE=${CUSTOM_VLLM_SOURCE:?set CUSTOM_VLLM_SOURCE}
RESULT_ROOT=${CANARY_RESULT_ROOT:?set CANARY_RESULT_ROOT}
CONFIG=${CANARY_CONFIG:-$ROOT/experiments/mxfp8_adaptive_rollout_v0251/configs/eval_ultra_tp8.yaml}
mkdir -p "$RESULT_ROOT/$ARM"
export CANARY_OUTPUT_DIR="$RESULT_ROOT/$ARM/eval"

contract=(
  python3 -m experiments.mxfp8_adaptive_rollout_v0251.contract
  --arm "$ARM"
  --source "$VLLM_SOURCE"
  --shell
)
if [[ "$ARM" == adaptive ]]; then
  contract+=(
    --tactic-file "${TACTIC_FILE:?set TACTIC_FILE}"
    --tactic-sha256 "${TACTIC_SHA256:?set TACTIC_SHA256}"
    --layer-allowlist-b64 "${LAYER_ALLOWLIST_B64:?set LAYER_ALLOWLIST_B64}"
    --switch-m "${SWITCH_M:-256}"
  )
fi
"${contract[@]}" > "$RESULT_ROOT/$ARM/arm.env"
source "$RESULT_ROOT/$ARM/arm.env"
export PYTHONPATH="$ROOT:$PYTHONPATH"

VLLM_PYTHON_BIN=${VLLM_PYTHON_BIN:-python-VllmGenerationWorker}
if ! command -v "$VLLM_PYTHON_BIN" >/dev/null 2>&1; then
  echo "missing NeMo-RL vLLM actor interpreter: $VLLM_PYTHON_BIN" >&2
  exit 2
fi
"$VLLM_PYTHON_BIN" - <<'PY' | tee "$RESULT_ROOT/$ARM/runtime.txt"
import hashlib
import os
from pathlib import Path
import flashinfer
import torch
import vllm
import vllm._C

source = Path(os.environ["CUSTOM_VLLM_SOURCE"]).resolve()
loaded = Path(vllm.__file__).resolve()
if source not in loaded.parents:
    raise SystemExit(f"vLLM source mismatch: expected {source}, loaded {loaded}")
print(f"vllm_version={vllm.__version__}")
print(f"vllm_file={loaded}")
print(f"vllm_extension={Path(vllm._C.__file__).resolve()}")
print(f"flashinfer_version={flashinfer.__version__}")
print(f"cuda_version={torch.version.cuda}")
if os.environ.get("TACTIC_FILE") and os.environ.get("NEMORL_MXFP8_LINEAR_BACKEND") == "flashinfer_trtllm":
    path = Path(os.environ["TACTIC_FILE"])
    print(f"tactic_sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")
PY

python3 "$ROOT/experiments/mxfp8_adaptive_rollout_v0251/run_eval_canary.py" \
  --config "$CONFIG" \
  --arm "$ARM" \
  2>&1 | tee "$RESULT_ROOT/$ARM/run.log"
