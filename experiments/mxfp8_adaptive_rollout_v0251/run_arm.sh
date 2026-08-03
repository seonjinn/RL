#!/usr/bin/env bash
set -euo pipefail

ARM=${1:?usage: run_arm.sh baseline|trace|trtllm_default|adaptive}
ROOT=${NEMO_RL_REPO_ROOT:?set NEMO_RL_REPO_ROOT}
VLLM_SOURCE=${CUSTOM_VLLM_SOURCE:?set CUSTOM_VLLM_SOURCE}
VLLM_RUNTIME_ROOT=${CUSTOM_VLLM_RUNTIME_ROOT:?set CUSTOM_VLLM_RUNTIME_ROOT}
RESULT_ROOT=${CANARY_RESULT_ROOT:?set CANARY_RESULT_ROOT}
CONFIG=${CANARY_CONFIG:-$ROOT/experiments/mxfp8_adaptive_rollout_v0251/configs/eval_ultra_tp8.yaml}
mkdir -p "$RESULT_ROOT/$ARM"
export CANARY_OUTPUT_DIR="$RESULT_ROOT/$ARM/eval"

for name in \
  VLLM_MXFP8_DENSE_SHAPE_TRACE \
  VLLM_MXFP8_DENSE_SHAPE_TRACE_DIR \
  VLLM_MXFP8_DENSE_SHAPE_TRACE_MAX \
  VLLM_MXFP8_DENSE_TRTLLM_LAYOUT \
  VLLM_MXFP8_DENSE_TRTLLM_SWITCH_M \
  VLLM_MXFP8_DENSE_TRTLLM_EXACT_TACTIC_FILE \
  VLLM_MXFP8_DENSE_TRTLLM_EXACT_TACTIC_SHA256 \
  VLLM_MXFP8_DENSE_TRTLLM_LAYER_ALLOWLIST \
  VLLM_MXFP8_DENSE_TRTLLM_LAYER_ALLOWLIST_B64 \
  VLLM_MXFP8_DENSE_TRTLLM_TACTIC \
  VLLM_MXFP8_DENSE_TRTLLM_TACTIC_HINTS_128X4; do
  unset "$name"
done

contract=(
  python3 -m experiments.mxfp8_adaptive_rollout_v0251.contract
  --arm "$ARM"
  --runtime-root "$VLLM_RUNTIME_ROOT"
  --shell
)
if [[ "$ARM" == adaptive ]]; then
  contract+=(
    --tactic-file "${TACTIC_FILE:?set TACTIC_FILE}"
    --tactic-sha256 "${TACTIC_SHA256:?set TACTIC_SHA256}"
    --layer-allowlist-b64 "${LAYER_ALLOWLIST_B64:?set LAYER_ALLOWLIST_B64}"
    --switch-m "${SWITCH_M:-256}"
  )
elif [[ "$ARM" == trtllm_default ]]; then
  contract+=(
    --layer-allowlist-b64 "${LAYER_ALLOWLIST_B64:?set LAYER_ALLOWLIST_B64}"
    --switch-m "${SWITCH_M:-256}"
  )
elif [[ "$ARM" == trace ]]; then
  contract+=(
    --trace-dir "${SHAPE_TRACE_DIR:?set SHAPE_TRACE_DIR}"
    --trace-max "${SHAPE_TRACE_MAX:-8192}"
  )
fi
"${contract[@]}" > "$RESULT_ROOT/$ARM/arm.env"
source "$RESULT_ROOT/$ARM/arm.env"
export PYTHONPATH="$ROOT:$PYTHONPATH"

actual_vllm_commit=$(git -C "$VLLM_SOURCE" rev-parse HEAD)
if [[ "$actual_vllm_commit" != "${EXPECTED_VLLM_COMMIT:?set EXPECTED_VLLM_COMMIT}" ]]; then
  echo "custom vLLM commit mismatch: expected $EXPECTED_VLLM_COMMIT, got $actual_vllm_commit" >&2
  exit 2
fi
git -C "$VLLM_SOURCE" diff --quiet
git -C "$VLLM_SOURCE" diff --cached --quiet
printf 'vllm_source=%s\nvllm_runtime_root=%s\nvllm_commit=%s\n' \
  "$VLLM_SOURCE" "$VLLM_RUNTIME_ROOT" "$actual_vllm_commit" \
  | tee "$RESULT_ROOT/$ARM/runtime.txt"

export NEMO_RL_PY_EXECUTABLES_SYSTEM=1
driver_python="${NEMO_RL_DRIVER_VENV_DIR:?set NEMO_RL_DRIVER_VENV_DIR}/bin/python"
"$driver_python" \
  "$ROOT/experiments/mxfp8_adaptive_rollout_v0251/run_eval_canary.py" \
  --config "$CONFIG" \
  --arm "$ARM" \
  2>&1 | tee "$RESULT_ROOT/$ARM/run.log"

if [[ "$ARM" == trace ]]; then
  "$driver_python" \
    -m experiments.mxfp8_adaptive_rollout_v0251.shape_trace \
    "$SHAPE_TRACE_DIR" \
    --output "$RESULT_ROOT/$ARM/shape_summary.json"
else
  grep -q 'enforce_eager.*False' "$RESULT_ROOT/$ARM/run.log"
fi
