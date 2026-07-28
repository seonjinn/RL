#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=00:20:00

# Callers own the cluster-specific GPU request: Ptyche requires
# --gpus-per-node=4, while OCI-HSG requires --gres=gpu:4.
set -euo pipefail

: "${CONTAINER:?Set CONTAINER to an immutable staged NeMo-RL nightly image.}"
: "${MOUNTS:?Set MOUNTS for the source and cache filesystems.}"
CONFIG=${CONFIG:-examples/configs/recipes/llm/performance/grpo-nanov3-30BA3B-2n8g-megatron-pack-cp-cudagraph-matrix.yaml}
export CONFIG

srun \
  --ntasks=1 \
  --container-image="${CONTAINER}" \
  --container-mounts="${MOUNTS}" \
  --no-container-mount-home \
  --container-workdir="${SLURM_SUBMIT_DIR}" \
  bash -lc '
    set -euo pipefail
    cd "${SLURM_SUBMIT_DIR}"
    python - <<"PY"
import importlib
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

assert sys.version_info >= (3, 13, 13), sys.version
assert torch.cuda.is_available(), "CUDA is not available in the staged image"
assert torch.cuda.device_count() >= 1, "No GPU is visible in the staged image"
print(f"python_version={sys.version}")
print(f"gpu_count={torch.cuda.device_count()}")

for module_name in (
    "nemo_rl",
    "transformer_engine.pytorch",
    "megatron.core",
    "mamba_ssm",
):
    module = importlib.import_module(module_name)
    print(f"{module_name}_path={Path(module.__file__).resolve()}")

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)

register_omegaconf_resolvers()
config = load_config(os.environ["CONFIG"])
config = parse_hydra_overrides(
    config,
    [
        "+policy.megatron_cfg.cuda_graph_scope=[attn,mamba]",
        "+policy.megatron_cfg.cuda_graph_impl=transformer_engine",
        "+policy.megatron_cfg.cuda_graph_packed_seq=true",
        "+policy.megatron_cfg.cuda_graph_warmup_steps=3",
    ],
)
MasterConfig(**OmegaConf.to_container(config, resolve=True))
print("current_uv_config_parse=passed")
print("runtime_package_probe=passed")
PY
    uv lock --check
    echo "full_runtime_probe=passed"
  '
