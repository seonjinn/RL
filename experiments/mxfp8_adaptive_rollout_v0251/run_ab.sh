#!/usr/bin/env bash
set -euo pipefail

ACTION=${1:-run}
if [[ "$ACTION" != run && "$ACTION" != smoke ]]; then
  echo "usage: run_ab.sh [run|smoke]" >&2
  exit 2
fi

ROOT=${NEMO_RL_REPO_ROOT:?set NEMO_RL_REPO_ROOT}
RESULT_ROOT=${CANARY_RESULT_ROOT:?set CANARY_RESULT_ROOT}
VLLM_SOURCE=${CUSTOM_VLLM_SOURCE:?set CUSTOM_VLLM_SOURCE}
VLLM_RUNTIME_BASE=${CUSTOM_VLLM_RUNTIME_BASE:?set CUSTOM_VLLM_RUNTIME_BASE}
export NEMO_RL_REPO_ROOT="$ROOT"
export CANARY_RESULT_ROOT="$RESULT_ROOT"

actual_vllm_commit=$(git -C "$VLLM_SOURCE" rev-parse HEAD)
if [[ "$actual_vllm_commit" != "${EXPECTED_VLLM_COMMIT:?set EXPECTED_VLLM_COMMIT}" ]]; then
  echo "custom vLLM commit mismatch: expected $EXPECTED_VLLM_COMMIT, got $actual_vllm_commit" >&2
  exit 2
fi
git -C "$VLLM_SOURCE" diff --quiet
git -C "$VLLM_SOURCE" diff --cached --quiet

runtime_python=(
  env -u PYTHONPATH -u VLLM_SUBPROCESS_PYTHONPATH
  "UV_PROJECT_ENVIRONMENT=${NEMO_RL_DRIVER_VENV_DIR:?set NEMO_RL_DRIVER_VENV_DIR}"
  uv run --locked --extra vllm --directory "$ROOT" python
)
VLLM_RUNTIME_ROOT=$("${runtime_python[@]}" \
  "$ROOT/experiments/mxfp8_adaptive_rollout_v0251/runtime_overlay.py" \
  --source-root "$VLLM_SOURCE" \
  --destination-base "$VLLM_RUNTIME_BASE" \
  --source-revision "$actual_vllm_commit")
export CUSTOM_VLLM_RUNTIME_ROOT="$VLLM_RUNTIME_ROOT"

PYTHONPATH="$ROOT:$VLLM_RUNTIME_ROOT" "${runtime_python[@]}" - "$VLLM_RUNTIME_ROOT" <<'PY'
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

if [[ "$ACTION" == smoke ]]; then
  exit 0
fi

bash "$ROOT/experiments/mxfp8_adaptive_rollout_v0251/run_arm.sh" baseline
bash "$ROOT/experiments/mxfp8_adaptive_rollout_v0251/run_arm.sh" adaptive
python3 -m experiments.mxfp8_adaptive_rollout_v0251.summarize \
  "$RESULT_ROOT/baseline/run.log" \
  "$RESULT_ROOT/adaptive/run.log" \
  --output "$RESULT_ROOT/summary.json"
cat "$RESULT_ROOT/summary.json"
