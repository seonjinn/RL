#!/usr/bin/env bash
set -euo pipefail

# This script is sourced by run_grpo_qwen3_235b_swe.sh inside the Slurm
# container before the NeMo-RL driver starts. It installs vLLM into a shared
# Lustre target so Ray actors on worker nodes can import it through PYTHONPATH.

INSTALL_VLLM_IN_SYSTEM="${INSTALL_VLLM_IN_SYSTEM:-true}"
if [[ "$INSTALL_VLLM_IN_SYSTEM" != "true" && "$INSTALL_VLLM_IN_SYSTEM" != "True" ]]; then
  return 0 2>/dev/null || exit 0
fi

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3}"
VLLM_PIP_SPEC="${VLLM_PIP_SPEC:-vllm==0.10.2}"
SHARED_VLLM_SITE="${SHARED_VLLM_SITE:-$ARTIFACT_ROOT/python_site/vllm_0_10_2_nodeps_py312}"
PYTHON_BIN="${DRIVER_LAUNCHER:-/opt/venv/bin/python}"
VLLM_EXTRA_PIP_SPECS=(
  msgspec
  blake3
  partial-json-parser
  gguf
  compressed-tensors==0.11.0
  depyf==0.19.0
  diskcache==5.6.3
  lark==1.2.2
  lm-format-enforcer==0.11.3
  "llguidance>=0.7.11,<0.8.0"
  outlines_core==0.2.11
  xgrammar==0.1.23
  cbor2
  ijson
  setproctitle
  "openai-harmony>=0.0.3"
  pydantic==2.13.4
  pydantic-core==2.46.4
  annotated-types==0.7.0
  typing-inspection==0.4.2
  transformers==4.55.2
  tokenizers==0.21.4
  "huggingface-hub>=0.34.0,<1.0"
  hf-xet
  "mistral_common[audio,image]>=1.8.2"
)
BOOTSTRAP_SPEC_MARKER="$SHARED_VLLM_SITE/.vllm_bootstrap_spec"
BOOTSTRAP_SPEC_TEXT="$VLLM_PIP_SPEC|${VLLM_EXTRA_PIP_SPECS[*]}"

marker_is_source_build() {
  [[ -f "$BOOTSTRAP_SPEC_MARKER" ]] && grep -q "source-build" "$BOOTSTRAP_SPEC_MARKER"
}

if [[ -f "$BOOTSTRAP_SPEC_MARKER" ]] && [[ "$(cat "$BOOTSTRAP_SPEC_MARKER")" != "$BOOTSTRAP_SPEC_TEXT" ]]; then
  if marker_is_source_build; then
    echo "[VLLM BOOTSTRAP] preserving source-built vLLM target despite pip spec mismatch: $SHARED_VLLM_SITE"
  elif [[ "$SHARED_VLLM_SITE" == "$ARTIFACT_ROOT"/python_site/* ]]; then
    echo "[VLLM BOOTSTRAP] shared target spec changed; cleaning $SHARED_VLLM_SITE"
    rm -rf "$SHARED_VLLM_SITE"
  else
    echo "[VLLM BOOTSTRAP] shared target spec changed but path is outside ARTIFACT_ROOT/python_site; reinstalling in place"
  fi
fi

mkdir -p "$SHARED_VLLM_SITE"
export PYTHONPATH="$SHARED_VLLM_SITE:${PYTHONPATH:-}"

if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
from vllm import SamplingParams  # noqa: F401
import vllm._C  # noqa: F401
from vllm.config import CompilationConfig  # noqa: F401
from vllm.logger import init_logger  # noqa: F401
PY
then
  echo "[VLLM BOOTSTRAP] vLLM native runtime imports already work with PYTHONPATH=$SHARED_VLLM_SITE"
  if marker_is_source_build; then
    echo "[VLLM BOOTSTRAP] keeping existing source-build marker at $BOOTSTRAP_SPEC_MARKER"
  else
    printf '%s\n' "$BOOTSTRAP_SPEC_TEXT" > "$BOOTSTRAP_SPEC_MARKER"
  fi
else
  if marker_is_source_build; then
    echo "[VLLM BOOTSTRAP] source-built vLLM target failed native import; refusing to overwrite it with a wheel: $SHARED_VLLM_SITE" >&2
    return 1 2>/dev/null || exit 1
  fi
  echo "[VLLM BOOTSTRAP] installing vLLM wheel into shared PYTHONPATH target: $SHARED_VLLM_SITE"
  echo "[VLLM BOOTSTRAP] pip spec: $VLLM_PIP_SPEC"
  "$PYTHON_BIN" -m pip install --no-deps --upgrade --target "$SHARED_VLLM_SITE" "$VLLM_PIP_SPEC"
  echo "[VLLM BOOTSTRAP] installing selected vLLM runtime dependencies into shared target"
  "$PYTHON_BIN" -m pip install --no-deps --upgrade --target "$SHARED_VLLM_SITE" "${VLLM_EXTRA_PIP_SPECS[@]}"
  "$PYTHON_BIN" - <<'PY'
from vllm import SamplingParams  # noqa: F401
import vllm._C  # noqa: F401
from vllm.config import CompilationConfig  # noqa: F401
from vllm.logger import init_logger  # noqa: F401
print("[VLLM BOOTSTRAP] vLLM native runtime import check passed")
PY
  printf '%s\n' "$BOOTSTRAP_SPEC_TEXT" > "$BOOTSTRAP_SPEC_MARKER"
fi
