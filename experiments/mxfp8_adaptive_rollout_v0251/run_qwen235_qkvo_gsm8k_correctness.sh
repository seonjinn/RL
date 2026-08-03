#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

if [[ "${NEMORL_ENABLE_QWEN235_GSM8K_CORRECTNESS:-0}" != 1 ]]; then
  echo "Qwen235 GSM8K correctness requires explicit opt-in: set NEMORL_ENABLE_QWEN235_GSM8K_CORRECTNESS=1" >&2
  exit 2
fi

ROOT=${NEMO_RL_REPO_ROOT:?set NEMO_RL_REPO_ROOT}
RESULT_ROOT=${CANARY_RESULT_ROOT:?set CANARY_RESULT_ROOT}
VLLM_SOURCE=${CUSTOM_VLLM_SOURCE:?set CUSTOM_VLLM_SOURCE}
VLLM_RUNTIME_BASE=${CUSTOM_VLLM_RUNTIME_BASE:?set CUSTOM_VLLM_RUNTIME_BASE}
EXPECTED_DATASET_SHA256=${GSM8K_EXPECTED_SHA256:-a6851a2bc0207bf1eb8b448a3605d53bed61936686a0e12ab2ce70be08d48e77}
export NEMO_RL_REPO_ROOT="$ROOT"
export CANARY_RESULT_ROOT="$RESULT_ROOT"
export CANARY_CONFIG="$ROOT/experiments/mxfp8_adaptive_rollout_v0251/configs/eval_qwen3_235ba22b_qkvo_gsm8k_correctness.yaml"

actual_vllm_commit=$(git -C "$VLLM_SOURCE" rev-parse HEAD)
if [[ "$actual_vllm_commit" != "${EXPECTED_VLLM_COMMIT:?set EXPECTED_VLLM_COMMIT}" ]]; then
  echo "custom vLLM commit mismatch: expected $EXPECTED_VLLM_COMMIT, got $actual_vllm_commit" >&2
  exit 2
fi
git -C "$VLLM_SOURCE" diff --quiet
git -C "$VLLM_SOURCE" diff --cached --quiet

driver_python="${NEMO_RL_DRIVER_VENV_DIR:?set NEMO_RL_DRIVER_VENV_DIR}/bin/python"
builder_python=(env -u PYTHONPATH -u VLLM_SUBPROCESS_PYTHONPATH "$driver_python")
VLLM_RUNTIME_ROOT=$("${builder_python[@]}" \
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
PY

PYTHONPATH="$ROOT:$VLLM_RUNTIME_ROOT" "$driver_python" \
  -m experiments.mxfp8_adaptive_rollout_v0251.flashinfer_preflight

input_dir="$RESULT_ROOT/input"
export GSM8K_JSONL="$input_dir/gsm8k_test.jsonl"
gsm8k_manifest="$input_dir/gsm8k_test.manifest.json"
PYTHONPATH="$ROOT:$VLLM_RUNTIME_ROOT" "$driver_python" \
  -m experiments.mxfp8_adaptive_rollout_v0251.materialize_gsm8k \
  --output "$GSM8K_JSONL" \
  --manifest "$gsm8k_manifest" \
  --expected-rows 1319 \
  --expected-sha256 "$EXPECTED_DATASET_SHA256"

bash "$ROOT/experiments/mxfp8_adaptive_rollout_v0251/run_arm.sh" baseline
bash "$ROOT/experiments/mxfp8_adaptive_rollout_v0251/run_arm.sh" adaptive

PYTHONPATH="$ROOT:$VLLM_RUNTIME_ROOT" "$driver_python" \
  -m experiments.mxfp8_adaptive_rollout_v0251.gsm8k_correctness_gate \
  --baseline-dir "$RESULT_ROOT/baseline/eval" \
  --adaptive-dir "$RESULT_ROOT/adaptive/eval" \
  --dataset "$GSM8K_JSONL" \
  --manifest "$gsm8k_manifest" \
  --expected-rows 1319 \
  --alpha 0.05 \
  --min-baseline-accuracy 0.01 \
  --output "$RESULT_ROOT/gsm8k_correctness_gate.json"
