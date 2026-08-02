#!/usr/bin/env bash
set -euo pipefail

ROOT=${NEMO_RL_REPO_ROOT:?set NEMO_RL_REPO_ROOT}
VLLM_SOURCE=${CUSTOM_VLLM_SOURCE:?set CUSTOM_VLLM_SOURCE}
VLLM_RUNTIME_BASE=${CUSTOM_VLLM_RUNTIME_BASE:?set CUSTOM_VLLM_RUNTIME_BASE}

actual_vllm_commit=$(git -C "$VLLM_SOURCE" rev-parse HEAD)
if [[ "$actual_vllm_commit" != "${EXPECTED_VLLM_COMMIT:?set EXPECTED_VLLM_COMMIT}" ]]; then
  echo "custom vLLM commit mismatch: expected $EXPECTED_VLLM_COMMIT, got $actual_vllm_commit" >&2
  exit 2
fi
git -C "$VLLM_SOURCE" diff --quiet
git -C "$VLLM_SOURCE" diff --cached --quiet

driver_python="${NEMO_RL_DRIVER_VENV_DIR:?set NEMO_RL_DRIVER_VENV_DIR}/bin/python"
VLLM_RUNTIME_ROOT=$(env -u PYTHONPATH -u VLLM_SUBPROCESS_PYTHONPATH \
  "$driver_python" \
  "$ROOT/experiments/mxfp8_adaptive_rollout_v0251/runtime_overlay.py" \
  --source-root "$VLLM_SOURCE" \
  --destination-base "$VLLM_RUNTIME_BASE" \
  --source-revision "$actual_vllm_commit")
export CUSTOM_VLLM_RUNTIME_ROOT="$VLLM_RUNTIME_ROOT"

PYTHONPATH="$ROOT:$VLLM_RUNTIME_ROOT" "$driver_python" - "$VLLM_RUNTIME_ROOT" <<'PY'
import importlib.util
import pathlib
import sys

runtime_root = pathlib.Path(sys.argv[1]).resolve()
import vllm
import vllm._C_stable_libtorch

package_path = pathlib.Path(vllm.__file__).resolve()
extension = importlib.util.find_spec("vllm._C_stable_libtorch")
if runtime_root not in package_path.parents:
    raise SystemExit(f"vLLM loaded outside runtime overlay: {package_path}")
if extension is None or extension.origin is None:
    raise SystemExit("vLLM stable extension has no import origin")
extension_path = pathlib.Path(extension.origin).resolve()
if runtime_root not in extension_path.parents:
    raise SystemExit(f"vLLM stable extension loaded outside overlay: {extension_path}")
print(f"vllm_package={package_path}")
print(f"vllm_stable_extension={extension_path}")
PY

PYTHONPATH="$ROOT:$VLLM_RUNTIME_ROOT" "$driver_python" \
  -m experiments.mxfp8_adaptive_rollout_v0251.flashinfer_preflight

bash "$ROOT/experiments/mxfp8_adaptive_rollout_v0251/run_arm.sh" trace
cat "$CANARY_RESULT_ROOT/trace/shape_summary.json"
