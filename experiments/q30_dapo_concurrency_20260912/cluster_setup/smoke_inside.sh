#!/usr/bin/env bash
set -euo pipefail
readonly SCRATCH=/tmp/q30-parity-smoke-${SLURM_JOB_ID}
mkdir -p "${SCRATCH}"
export HF_HOME="${SCRATCH}/hf" UV_CACHE_DIR="${SCRATCH}/uv" TRITON_CACHE_DIR="${SCRATCH}/triton"
readonly PY=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python
test -x "${PY}"
"${PY}" "${Q30_SETUP_REPO}/experiments/qwen3_30ba3b_bf16_flashinfer_specdec_latest_main_20260909/prepare_vllm_dspark_fap_overlay.py" --overlay-root "${SCRATCH}/vllm-overlay"
export PYTHONPATH="${SCRATCH}/vllm-overlay:${Q30_SETUP_REPO}:${Q30_SETUP_REPO}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
"${PY}" - <<'PY'
import importlib.metadata
import json
import os
from pathlib import Path

import torch
import transformer_engine.pytorch
import megatron.core
import nemo_rl
import vllm

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 4, torch.cuda.device_count()
assert vllm.__version__.startswith("0.25.1"), vllm.__version__
for device in range(4):
    value = torch.ones(16, device=f"cuda:{device}")
    assert value.sum().item() == 16
print("GPU_SMOKE_OK", torch.cuda.get_device_name(), torch.__version__, vllm.__version__)
print("FLASHINFER", importlib.metadata.version("flashinfer-python"))
assets = Path(os.environ["Q30_SETUP_ASSETS"])
for name in ("target", "dflash", "dspark"):
    config = json.loads((assets / name / "config.json").read_text())
    print("MODEL_CONFIG", name, config.get("architectures"), config.get("model_type"))
print("IMPORT_SMOKE_COMPLETE; model execution and multi-node GRPO still require a canary")
PY
