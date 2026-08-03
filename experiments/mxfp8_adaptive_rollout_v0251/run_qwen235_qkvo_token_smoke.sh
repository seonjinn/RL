#!/usr/bin/env bash
set -euo pipefail

if [[ "${NEMORL_ENABLE_QWEN235_QKVO_TOKEN_SMOKE:-0}" != 1 ]]; then
  echo "Qwen235 QKVO token smoke requires explicit opt-in" >&2
  exit 2
fi

ROOT=${NEMO_RL_REPO_ROOT:?set NEMO_RL_REPO_ROOT}
RESULT_ROOT=${CANARY_RESULT_ROOT:?set CANARY_RESULT_ROOT}
scope=${NEMORL_QWEN235_TOKEN_SMOKE_SCOPE:-qkvo}
case "$scope" in
  qkvo)
    config_name=eval_qwen3_235ba22b_qkvo_token_smoke.yaml
    ;;
  moe)
    config_name=eval_qwen3_235ba22b_moe_token_smoke.yaml
    ;;
  bf16)
    config_name=eval_qwen3_235ba22b_bf16_token_smoke.yaml
    ;;
  *)
    echo "unsupported Qwen235 token smoke scope: $scope" >&2
    exit 2
    ;;
esac
export CANARY_CONFIG="$ROOT/experiments/mxfp8_adaptive_rollout_v0251/configs/$config_name"

bash "$ROOT/experiments/mxfp8_adaptive_rollout_v0251/run_ab.sh" baseline

driver_python="${NEMO_RL_DRIVER_VENV_DIR:?set NEMO_RL_DRIVER_VENV_DIR}/bin/python"
PYTHONPATH="$ROOT:${CUSTOM_VLLM_RUNTIME_ROOT:?set CUSTOM_VLLM_RUNTIME_ROOT}" \
  "$driver_python" \
  -m experiments.mxfp8_adaptive_rollout_v0251.response_validity_gate \
  --evaluation "$RESULT_ROOT/baseline/eval/evaluation_data.json" \
  --expected-rows 64 \
  --max-repetitive-fraction 0.1 \
  --output "$RESULT_ROOT/response_validity_gate.json"
