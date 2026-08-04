#!/usr/bin/env bash
set -euo pipefail

scope=${NEMORL_QWEN235_REFIT_SCOPE:-moe}
case "$scope" in
  moe)
    config_name=grpo_qwen3_235ba22b_moe_refit_token_smoke.yaml
    ;;
  qkvo)
    config_name=grpo_qwen3_235ba22b_qkvo_refit_token_smoke.yaml
    ;;
  *)
    echo "unsupported Qwen235 refit scope: $scope" >&2
    exit 2
    ;;
esac

root=${NEMO_RL_REPO_ROOT:?set NEMO_RL_REPO_ROOT}
result_root=${CANARY_RESULT_ROOT:?set CANARY_RESULT_ROOT}
vllm_source=${CUSTOM_VLLM_SOURCE:?set CUSTOM_VLLM_SOURCE}
driver_python=${NEMO_RL_DRIVER_VENV_DIR:?set NEMO_RL_DRIVER_VENV_DIR}/bin/python

actual_nemo_rl_commit=$(git -C "$root" rev-parse HEAD)
if [[ "$actual_nemo_rl_commit" != "${EXPECTED_NEMO_RL_COMMIT:?set EXPECTED_NEMO_RL_COMMIT}" ]]; then
  echo "NeMo-RL commit mismatch: expected $EXPECTED_NEMO_RL_COMMIT, got $actual_nemo_rl_commit" >&2
  exit 2
fi
if [[ -n "$(git -C "$root" status --porcelain --untracked-files=all)" ]]; then
  echo "NeMo-RL repository is not clean: $root" >&2
  exit 2
fi

actual_vllm_commit=$(git -C "$vllm_source" rev-parse HEAD)
if [[ "$actual_vllm_commit" != "${EXPECTED_VLLM_COMMIT:?set EXPECTED_VLLM_COMMIT}" ]]; then
  echo "custom vLLM commit mismatch: expected $EXPECTED_VLLM_COMMIT, got $actual_vllm_commit" >&2
  exit 2
fi
if [[ -n "$(git -C "$vllm_source" status --porcelain --untracked-files=all)" ]]; then
  echo "custom vLLM repository is not clean: $vllm_source" >&2
  exit 2
fi

runtime_root=$(env -u PYTHONPATH -u VLLM_SUBPROCESS_PYTHONPATH "$driver_python" \
  "$root/experiments/mxfp8_adaptive_rollout_v0251/runtime_overlay.py" \
  --source-root "$vllm_source" \
  --destination-base "${CUSTOM_VLLM_RUNTIME_BASE:?set CUSTOM_VLLM_RUNTIME_BASE}" \
  --source-revision "$actual_vllm_commit")
export PYTHONPATH="$root:$runtime_root"
export VLLM_SUBPROCESS_PYTHONPATH="$PYTHONPATH"
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1
export NEMORL_MXFP8_REFIT_AUDIT=1

"$driver_python" \
  "$root/experiments/mxfp8_adaptive_rollout_v0251/flashinfer_preflight.py"
"$driver_python" "$root/examples/run_grpo.py" \
  --config "$root/experiments/mxfp8_adaptive_rollout_v0251/configs/$config_name" \
  2>&1 | tee "$result_root/run.log"

grep -q "NEMORL_MXFP8_REFIT event=load_weights" "$result_root/run.log"
grep -q "enforce_eager.*False" "$result_root/run.log"
"$driver_python" \
  -m experiments.mxfp8_adaptive_rollout_v0251.refit_validation_gate \
  --log-root "$result_root/validation" \
  --expected-rows 64 \
  --max-repetitive-fraction 0.1 \
  --output "$result_root/response_validity_gate.json"
