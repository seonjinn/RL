#!/bin/bash
#SBATCH --job-name=q235b-vllm-build
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --mem=0
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3}"
REPORT_DIR="${REPORT_DIR:-$ARTIFACT_ROOT/reports}"
CONTAINER="${CONTAINER:?set CONTAINER}"
MOUNTS="${MOUNTS:-/lustre:/lustre,$REPO_ROOT:$REPO_ROOT,$ARTIFACT_ROOT:$ARTIFACT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
VLLM_SOURCE_SPEC="${VLLM_SOURCE_SPEC:-https://files.pythonhosted.org/packages/7d/0a/278d7bbf454f7de5322a5007427eed3e8b34ed6c2802491b56bbdfd7bbb4/vllm-0.10.2.tar.gz}"
OUTPUT_SITE="${OUTPUT_SITE:-$ARTIFACT_ROOT/python_site/vllm_0_10_2_cu129_torch28nv_source_py312}"
TMP_SITE="${TMP_SITE:-$OUTPUT_SITE.tmp.$SLURM_JOB_ID}"
BUILD_PAYLOAD="${BUILD_PAYLOAD:-$REPORT_DIR/vllm_native_source_build_payload.sh}"
JSON_OUT="${JSON_OUT:-$REPORT_DIR/vllm_native_source_build.json}"
MARKDOWN_OUT="${MARKDOWN_OUT:-$REPORT_DIR/vllm_native_source_build.md}"

mkdir -p logs "$REPORT_DIR" "$(dirname "$OUTPUT_SITE")"

cat > "$BUILD_PAYLOAD" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

deps=(
  "${SETUPTOOLS_SPEC:-setuptools>=77.0.3,<81.0.0}"
  wheel
  setuptools_scm
  "${SETUPTOOLS_RUST_SPEC:-setuptools-rust>=1.9.0}"
  vcs_versioning
  cmake
  ninja
  packaging
  pybind11
  jinja2
  msgspec
  blake3
  partial-json-parser
  gguf
  "${COMPRESSED_TENSORS_SPEC:-compressed-tensors==0.11.0}"
  "${DEPYF_SPEC:-depyf==0.19.0}"
  "${DISKCACHE_SPEC:-diskcache==5.6.3}"
  "${LARK_SPEC:-lark==1.2.2}"
  "${LM_FORMAT_ENFORCER_SPEC:-lm-format-enforcer==0.11.3}"
  "${LLGUIDANCE_SPEC:-llguidance>=0.7.11,<0.8.0}"
  "${OUTLINES_CORE_SPEC:-outlines_core==0.2.11}"
  "${XGRAMMAR_SPEC:-xgrammar==0.1.23}"
  cbor2
  ijson
  setproctitle
  pybase64
  py-cpuinfo
  frozendict
  "${OPENAI_HARMONY_SPEC:-openai-harmony>=0.0.3}"
  "${PYDANTIC_SPEC:-pydantic==2.13.4}"
  "${PYDANTIC_CORE_SPEC:-pydantic-core==2.46.4}"
  "${ANNOTATED_TYPES_SPEC:-annotated-types==0.7.0}"
  "${TYPING_INSPECTION_SPEC:-typing-inspection==0.4.2}"
  "${TRANSFORMERS_SPEC:-transformers==4.55.2}"
  "${TOKENIZERS_SPEC:-tokenizers==0.21.4}"
  "${HUGGINGFACE_HUB_SPEC:-huggingface-hub>=0.34.0,<1.0}"
  hf-xet
  "${MISTRAL_COMMON_SPEC:-mistral_common[audio,image]>=1.8.2}"
)

echo "[VLLM SOURCE BUILD] python=$PYTHON_BIN"
"$PYTHON_BIN" - <<'PY'
import sys
import torch
print("python", sys.version)
print("torch", torch.__version__, "cuda", torch.version.cuda, "file", torch.__file__)
PY

rm -rf "$TMP_SITE"
mkdir -p "$TMP_SITE"

export MAX_JOBS="${MAX_JOBS:-16}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-16}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0 10.0}"
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cuda}"
export VLLM_USE_PRECOMPILED=0
export VLLM_PRECOMPILED_WHEEL_LOCATION=
export PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"

echo "[VLLM SOURCE BUILD] installing python runtime deps into $TMP_SITE"
"$PYTHON_BIN" -m pip install --no-deps --upgrade --target "$TMP_SITE" "${deps[@]}"
export PYTHONPATH="$TMP_SITE:${PYTHONPATH:-}"

echo "[VLLM SOURCE BUILD] building/installing $VLLM_SOURCE_SPEC from source into $TMP_SITE"
"$PYTHON_BIN" -m pip install \
  --no-deps \
  --no-build-isolation \
  --no-binary=vllm \
  --target "$TMP_SITE" \
  "$VLLM_SOURCE_SPEC"

echo "[VLLM SOURCE BUILD] probing native imports from $TMP_SITE"
PYTHONPATH="$TMP_SITE:${PYTHONPATH:-}" "$PYTHON_BIN" - <<'PY'
import json
import os
import sys
out = {"python": sys.executable, "site": os.environ.get("TMP_SITE")}
import torch
out["torch_version"] = torch.__version__
out["torch_cuda"] = torch.version.cuda
import vllm
out["vllm_version"] = getattr(vllm, "__version__", None)
out["vllm_file"] = vllm.__file__
import vllm._C
out["vllm_c_file"] = vllm._C.__file__
from vllm.config import CompilationConfig
out["compilation_config_ok"] = True
from vllm.v1.engine.async_llm import AsyncLLM
out["async_llm_import_ok"] = True
from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM
out["qwen3_moe_import_ok"] = True
try:
    from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (  # noqa: E501
        ExampleHiddenStatesConnector,
    )
    out["example_hidden_states_connector_ok"] = True
except Exception as exc:
    out["example_hidden_states_connector_ok"] = False
    out["example_hidden_states_connector_error"] = repr(exc)
out["extract_hidden_states_method_ok"] = False
try:
    import inspect
    from vllm.config.speculative import SpeculativeConfig
    out["extract_hidden_states_method_ok"] = "extract_hidden_states" in inspect.getsource(SpeculativeConfig)
except Exception as exc:
    out["extract_hidden_states_method_error"] = repr(exc)
print(json.dumps(out, sort_keys=True))
PY

rm -rf "$OUTPUT_SITE"
mv "$TMP_SITE" "$OUTPUT_SITE"
printf "%s\n" "$VLLM_SOURCE_SPEC|source-build|torch28nv" > "$OUTPUT_SITE/.vllm_bootstrap_spec"

"$PYTHON_BIN" - "$JSON_OUT" "$MARKDOWN_OUT" "$OUTPUT_SITE" "$VLLM_SOURCE_SPEC" <<'PY'
import json
import sys
import time
from pathlib import Path

json_out = Path(sys.argv[1])
markdown_out = Path(sys.argv[2])
output_site = sys.argv[3]
spec = sys.argv[4]
payload = {
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    "overall_status": "pass",
    "output_site": output_site,
    "vllm_source_spec": spec,
}
json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
markdown_out.write_text(
    "# vLLM Native Source Build\n\n"
    "Overall: **PASS**\n\n"
    f"Output site: `{output_site}`\n\n"
    f"Source spec: `{spec}`\n",
    encoding="utf-8",
)
PY

echo "[VLLM SOURCE BUILD] complete: $OUTPUT_SITE"
SH
chmod +x "$BUILD_PAYLOAD"

SRUN_CONTAINER_ARGS=(
  --container-image="$CONTAINER"
  --container-mounts="$MOUNTS"
  --container-workdir="$REPO_ROOT"
  --no-container-mount-home
)

if sinfo -p "$SLURM_JOB_PARTITION" -h -o "%G" | grep -q "gpu:"; then
  SRUN_CONTAINER_ARGS+=(--gres="gpu:$GPUS_PER_NODE")
fi

srun "${SRUN_CONTAINER_ARGS[@]}" --ntasks=1 env \
  PYTHON_BIN="$PYTHON_BIN" \
  VLLM_SOURCE_SPEC="$VLLM_SOURCE_SPEC" \
  OUTPUT_SITE="$OUTPUT_SITE" \
  TMP_SITE="$TMP_SITE" \
  JSON_OUT="$JSON_OUT" \
  MARKDOWN_OUT="$MARKDOWN_OUT" \
  "$BUILD_PAYLOAD"
