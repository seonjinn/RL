#!/bin/bash
#SBATCH --job-name=q235b-vllm-finalize
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=batch
#SBATCH --time=01:00:00
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
SOURCE_TMP_SITE="${SOURCE_TMP_SITE:?set SOURCE_TMP_SITE}"
OUTPUT_SITE="${OUTPUT_SITE:-$ARTIFACT_ROOT/python_site/vllm_0_10_2_cu129_torch28nv_source_py312}"
VLLM_SOURCE_SPEC="${VLLM_SOURCE_SPEC:-https://files.pythonhosted.org/packages/7d/0a/278d7bbf454f7de5322a5007427eed3e8b34ed6c2802491b56bbdfd7bbb4/vllm-0.10.2.tar.gz}"
BUILD_PAYLOAD="${BUILD_PAYLOAD:-$REPORT_DIR/vllm_finalize_source_site_payload.sh}"
JSON_OUT="${JSON_OUT:-$REPORT_DIR/vllm_native_source_build.json}"
MARKDOWN_OUT="${MARKDOWN_OUT:-$REPORT_DIR/vllm_native_source_build.md}"
FINALIZE_DEPS="${FINALIZE_DEPS:-pybase64}"
IN_PLACE="${IN_PLACE:-false}"

mkdir -p logs "$REPORT_DIR" "$(dirname "$OUTPUT_SITE")"

cat > "$BUILD_PAYLOAD" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d "$SOURCE_TMP_SITE" ]]; then
  echo "SOURCE_TMP_SITE does not exist: $SOURCE_TMP_SITE" >&2
  exit 1
fi

read -r -a finalize_deps <<< "$FINALIZE_DEPS"

echo "[VLLM SOURCE FINALIZE] python=$PYTHON_BIN"
"$PYTHON_BIN" - <<'PY'
import sys
import torch
print("python", sys.version)
print("torch", torch.__version__, "cuda", torch.version.cuda, "file", torch.__file__)
PY

echo "[VLLM SOURCE FINALIZE] installing missing runtime deps into $SOURCE_TMP_SITE: ${finalize_deps[*]}"
"$PYTHON_BIN" -m pip install --no-deps --upgrade --target "$SOURCE_TMP_SITE" "${finalize_deps[@]}"

echo "[VLLM SOURCE FINALIZE] probing native imports from $SOURCE_TMP_SITE"
PYTHONPATH="$SOURCE_TMP_SITE:${PYTHONPATH:-}" "$PYTHON_BIN" - <<'PY'
import json
import os
import sys
out = {"python": sys.executable, "site": os.environ.get("SOURCE_TMP_SITE")}
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
print(json.dumps(out, sort_keys=True))
PY

if [[ "$IN_PLACE" == "true" || "$IN_PLACE" == "True" || "$IN_PLACE" == "1" ]]; then
  if [[ "$SOURCE_TMP_SITE" != "$OUTPUT_SITE" ]]; then
    echo "IN_PLACE=true requires SOURCE_TMP_SITE to equal OUTPUT_SITE" >&2
    exit 1
  fi
else
  rm -rf "$OUTPUT_SITE"
  mv "$SOURCE_TMP_SITE" "$OUTPUT_SITE"
fi
printf "%s\n" "$VLLM_SOURCE_SPEC|source-build|torch28nv|finalized" > "$OUTPUT_SITE/.vllm_bootstrap_spec"

"$PYTHON_BIN" - "$JSON_OUT" "$MARKDOWN_OUT" "$OUTPUT_SITE" "$VLLM_SOURCE_SPEC" "$FINALIZE_DEPS" "$IN_PLACE" <<'PY'
import json
import sys
import time
from pathlib import Path

json_out = Path(sys.argv[1])
markdown_out = Path(sys.argv[2])
output_site = sys.argv[3]
spec = sys.argv[4]
finalize_deps = sys.argv[5]
in_place = sys.argv[6]
payload = {
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    "overall_status": "pass",
    "output_site": output_site,
    "vllm_source_spec": spec,
    "finalized_from_tmp_site": True,
    "finalized_in_place": in_place.lower() in {"true", "1"},
    "finalize_deps": finalize_deps.split(),
}
json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
markdown_out.write_text(
    "# vLLM Native Source Build\n\n"
    "Overall: **PASS**\n\n"
    f"Output site: `{output_site}`\n\n"
    f"Source spec: `{spec}`\n\n"
    f"Finalized from tmp site with deps: `{finalize_deps}`\n",
    encoding="utf-8",
)
PY

echo "[VLLM SOURCE FINALIZE] complete: $OUTPUT_SITE"
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
  SOURCE_TMP_SITE="$SOURCE_TMP_SITE" \
  OUTPUT_SITE="$OUTPUT_SITE" \
  VLLM_SOURCE_SPEC="$VLLM_SOURCE_SPEC" \
  JSON_OUT="$JSON_OUT" \
  MARKDOWN_OUT="$MARKDOWN_OUT" \
  FINALIZE_DEPS="$FINALIZE_DEPS" \
  IN_PLACE="$IN_PLACE" \
  "$BUILD_PAYLOAD"
