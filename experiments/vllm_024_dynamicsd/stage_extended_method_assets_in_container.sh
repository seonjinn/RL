#!/usr/bin/env bash
set -euo pipefail

: "${HF_HOME:?HF_HOME is required}"
: "${ASSET_ROOT:?ASSET_ROOT is required}"
: "${VLLM_COMMIT:?VLLM_COMMIT is required}"
: "${ANGELSLIM_COMMIT:?ANGELSLIM_COMMIT is required}"
: "${PARD_COMMIT:?PARD_COMMIT is required}"
: "${QWEN8_REVISION:?QWEN8_REVISION is required}"
: "${PARD_REVISION:?PARD_REVISION is required}"
: "${PARD2_REVISION:?PARD2_REVISION is required}"
: "${DFLASH_REVISION:?DFLASH_REVISION is required}"
: "${DFLARE_REVISION:?DFLARE_REVISION is required}"

export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PYTHONUNBUFFERED=1

python3 - <<'PY'
import os

from huggingface_hub import snapshot_download

assets = (
    ("Qwen/Qwen3-8B", os.environ["QWEN8_REVISION"]),
    ("amd/PARD-Qwen3-0.6B", os.environ["PARD_REVISION"]),
    ("amd/PARD2-Qwen3-8B", os.environ["PARD2_REVISION"]),
    ("z-lab/Qwen3-8B-DFlash-b16", os.environ["DFLASH_REVISION"]),
    ("AngelSlim/Qwen3-8b-dflare", os.environ["DFLARE_REVISION"]),
)
for repo_id, revision in assets:
    path = snapshot_download(repo_id=repo_id, revision=revision)
    print(f"staged={repo_id}@{revision} path={path}", flush=True)
PY

clone_pinned() {
  local repository="$1"
  local commit="$2"
  local destination="$3"
  if [[ ! -d "${destination}/.git" ]]; then
    git clone --filter=blob:none "${repository}" "${destination}"
  fi
  git -C "${destination}" fetch origin "${commit}" --depth=1
  git -C "${destination}" checkout --detach "${commit}"
  test "$(git -C "${destination}" rev-parse HEAD)" = "${commit}"
}

vllm_source="${ASSET_ROOT}/src/vllm-${VLLM_COMMIT}"
angelslim_source="${ASSET_ROOT}/src/angelslim-${ANGELSLIM_COMMIT}"
pard_source="${ASSET_ROOT}/src/pard-${PARD_COMMIT}"
clone_pinned https://github.com/vllm-project/vllm.git "${VLLM_COMMIT}" "${vllm_source}"
clone_pinned https://github.com/Tencent/AngelSlim.git "${ANGELSLIM_COMMIT}" "${angelslim_source}"
clone_pinned https://github.com/AMD-AGI/PARD.git "${PARD_COMMIT}" "${pard_source}"

if ! grep -q 'method in ("draft_model", "pard2")' "${vllm_source}/vllm/config/speculative.py"; then
  patch -p1 -d "${vllm_source}" \
    < /workspace/experiment/patches/vllm024_pard2_target_features.patch
fi
if ! grep -q 'output-json' "${angelslim_source}/tools/dflash_benchmark.py"; then
  patch -p1 -d "${angelslim_source}" \
    < /workspace/experiment/patches/angelslim_benchmark_json.patch
fi

common_site_versioned="${ASSET_ROOT}/python/common-py312-arctic-0.1.1"
if [[ ! -f "${common_site_versioned}/.complete" ]]; then
  mkdir -p "${common_site_versioned}"
  CMAKE_BUILD_PARALLEL_LEVEL=16 python3 -m pip install \
    --no-build-isolation --no-deps --no-cache-dir \
    --target "${common_site_versioned}" 'arctic-inference==0.1.1'
  PYTHONPATH="${common_site_versioned}" python3 -c 'import arctic_inference'
  touch "${common_site_versioned}/.complete"
fi
ln -sfn "${common_site_versioned}" "${ASSET_ROOT}/python/common"

pard2_overlay_versioned="${ASSET_ROOT}/python/pard2-overlay-${VLLM_COMMIT}"
if [[ ! -f "${pard2_overlay_versioned}/.complete" ]]; then
  pard2_overlay_tmp="${pard2_overlay_versioned}.partial.${SLURM_JOB_ID}"
  rm -rf "${pard2_overlay_tmp}"
  mkdir -p "${pard2_overlay_tmp}"
  OVERLAY_TMP="${pard2_overlay_tmp}" python3 - <<'PY'
import os
import shutil
from pathlib import Path

import vllm

source = Path(vllm.__file__).resolve().parent
destination = Path(os.environ["OVERLAY_TMP"]) / "vllm"
shutil.copytree(source, destination)
PY
  for relative_path in \
    vllm/config/speculative.py \
    vllm/model_executor/models/qwen3.py \
    vllm/v1/spec_decode/draft_model.py \
    vllm/v1/spec_decode/llm_base_proposer.py \
    vllm/v1/worker/gpu_model_runner.py; do
    install -D -m 0644 \
      "${vllm_source}/${relative_path}" \
      "${pard2_overlay_tmp}/${relative_path}"
  done
  python3 -m compileall -q "${pard2_overlay_tmp}/vllm"
  touch "${pard2_overlay_tmp}/.complete"
  mv "${pard2_overlay_tmp}" "${pard2_overlay_versioned}"
fi
ln -sfn "${pard2_overlay_versioned}" "${ASSET_ROOT}/python/pard2_overlay"

PYTHONPATH="${ASSET_ROOT}/python/pard2_overlay:${ASSET_ROOT}/python/common" \
  python3 - <<'PY'
import inspect

import vllm
from vllm.config.speculative import SpeculativeConfig

assert vllm.__version__ == "0.24.0", vllm.__version__
assert "pard2" in inspect.getsource(SpeculativeConfig.uses_draft_model)
print(f"validated_vllm={vllm.__version__}")
PY
PYTHONPATH="${angelslim_source}" python3 - <<'PY'
from angelslim.compressor.speculative.train.models.draft.qwen_dflare import (
    QwenDFlareDraftModel,
)
from angelslim.compressor.speculative.train.models.draft.qwen_dflash import (
    QwenDFlashDraftModel,
)

print(QwenDFlashDraftModel.__name__, QwenDFlareDraftModel.__name__)
PY
