#!/bin/bash

set -euo pipefail

EXPECTED_VLLM_COMMIT=${EXPECTED_VLLM_COMMIT:-77d5e10eec8f5cc217d16a9230f2955cf8553cee}
EXPECTED_VLLM_REPOSITORY=${EXPECTED_VLLM_REPOSITORY:-https://github.com/seonjinn/vllm.git}
EXPECTED_VLLM_VERSION=${EXPECTED_VLLM_VERSION:-0.20.2}
EXPECTED_FLASHINFER_VERSION=${EXPECTED_FLASHINFER_VERSION:-0.6.8.post1}
EXPECTED_MODEL=${EXPECTED_MODEL:-Qwen/Qwen3-30B-A3B}
EXPECTED_TP=${EXPECTED_TP:-1}
VLLM_PYTHON_BIN=${VLLM_PYTHON_BIN:-/usr/local/bin/python-VllmGenerationWorker}
MCORE_PYTHON_BIN=${MCORE_PYTHON_BIN:-/usr/local/bin/python-MegatronPolicyWorker}

: "${EXPECTED_NEMO_RL_COMMIT:?Set the immutable NeMo-RL commit}"
: "${EXPECTED_CONFIG_NAME:?Set the package-relative MXFP8 JSON name}"
: "${EXPECTED_CONFIG_SHA256:?Set the expected raw JSON SHA256}"
: "${CONTAINER_IMAGE:?Set the immutable .sqsh path}"
: "${EXPECTED_CONTAINER_SHA256:?Set the immutable .sqsh SHA256}"

if [[ "${SLURM_JOB_NUM_NODES:-0}" != "1" ]]; then
  echo "container smoke requires exactly one SLURM node" >&2
  exit 2
fi
if [[ -L "$CONTAINER_IMAGE" ]]; then
  echo "container smoke rejects convenience symlinks: $CONTAINER_IMAGE" >&2
  exit 2
fi
actual_container_sha256=$(sha256sum "$CONTAINER_IMAGE" | awk '{print $1}')
if [[ "$actual_container_sha256" != "$EXPECTED_CONTAINER_SHA256" ]]; then
  echo "container SHA256 mismatch" >&2
  exit 2
fi

export EXPECTED_VLLM_COMMIT
export EXPECTED_VLLM_REPOSITORY
export EXPECTED_VLLM_VERSION
export EXPECTED_FLASHINFER_VERSION
export EXPECTED_MODEL
export EXPECTED_TP
export EXPECTED_NEMO_RL_COMMIT
export EXPECTED_CONFIG_NAME
export EXPECTED_CONFIG_SHA256

"$VLLM_PYTHON_BIN" - <<'PY'
import hashlib
import os
import subprocess
from pathlib import Path

import flashinfer
import nemo_rl
import torch
import vllm
from vllm.model_executor.kernels.linear.mxfp8.tactic_config import (
    load_mxfp8_dense_runtime_config,
)

expected_gpu_count = 4
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == expected_gpu_count, (
    f"expected {expected_gpu_count} visible GPUs, got {torch.cuda.device_count()}"
)
assert vllm.__version__ == os.environ["EXPECTED_VLLM_VERSION"]
assert flashinfer.__version__ == os.environ["EXPECTED_FLASHINFER_VERSION"]

vllm_root = Path(vllm.__file__).resolve().parents[1]
vllm_commit = subprocess.check_output(
    ["git", "-C", str(vllm_root), "rev-parse", "HEAD"], text=True
).strip()
assert vllm_commit == os.environ["EXPECTED_VLLM_COMMIT"]
vllm_remotes = subprocess.check_output(
    ["git", "-C", str(vllm_root), "remote", "-v"], text=True
)
assert os.environ["EXPECTED_VLLM_REPOSITORY"] in vllm_remotes

nemo_root = Path(nemo_rl.__file__).resolve().parents[1]
nemo_commit = subprocess.check_output(
    ["git", "-C", str(nemo_root), "rev-parse", "HEAD"], text=True
).strip()
assert nemo_commit == os.environ["EXPECTED_NEMO_RL_COMMIT"]

major, minor = torch.cuda.get_device_capability()
runtime_config = load_mxfp8_dense_runtime_config(
    os.environ["EXPECTED_CONFIG_NAME"],
    actual_vllm_version=vllm.__version__,
    actual_flashinfer_version=flashinfer.__version__,
    actual_compute_capability=(major, minor),
    actual_model=os.environ["EXPECTED_MODEL"],
    actual_tensor_parallel_size=int(os.environ["EXPECTED_TP"]),
)
config_bytes = runtime_config.source_path.read_bytes()
assert hashlib.sha256(config_bytes).hexdigest() == os.environ[
    "EXPECTED_CONFIG_SHA256"
]
assert runtime_config.source_sha256 == os.environ["EXPECTED_CONFIG_SHA256"]
assert runtime_config.compatibility["model"] == os.environ["EXPECTED_MODEL"]
assert runtime_config.compatibility["tensor_parallel_size"] == int(
    os.environ["EXPECTED_TP"]
)
assert runtime_config.provenance["container_sha256"] == os.environ[
    "EXPECTED_CONTAINER_SHA256"
]

print(f"cuda_visible={torch.cuda.device_count()}")
print(f"nemo_rl_source={nemo_rl.__file__}")
print(f"nemo_rl_commit={nemo_commit}")
print(f"vllm_source={vllm.__file__}")
print(f"vllm_commit={vllm_commit}")
print(f"vllm_version={vllm.__version__}")
print(f"flashinfer_version={flashinfer.__version__}")
print(f"config_path={runtime_config.source_path}")
print(f"config_sha256={runtime_config.source_sha256}")
print("loader_ok=true")
PY

"$MCORE_PYTHON_BIN" - <<'PY'
from pathlib import Path

import megatron.core
import torch
import transformer_engine.pytorch

expected_gpu_count = 4
assert torch.cuda.is_available(), "CUDA is unavailable in the mcore environment"
assert torch.cuda.device_count() == expected_gpu_count, (
    f"expected {expected_gpu_count} visible GPUs in the mcore environment, "
    f"got {torch.cuda.device_count()}"
)

print(f"megatron_core_source={Path(megatron.core.__file__).resolve()}")
print(
    "transformer_engine_source="
    f"{Path(transformer_engine.pytorch.__file__).resolve()}"
)
print("mcore_loader_ok=true")
PY
