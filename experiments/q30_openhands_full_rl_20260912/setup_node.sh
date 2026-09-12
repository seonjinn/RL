#!/usr/bin/env bash
set -euo pipefail
: "${SWE_NODE_ROOT:?}" "${SWE_SOURCE_ROOT:?}"
mkdir -p "${SWE_NODE_ROOT}"/{tmp,cache,uv-cache,gym-venvs,mcore-overlay,triton,torch-extensions,vllm-cache,wandb-cache}
export UV_CACHE_DIR="${SWE_NODE_ROOT}/uv-cache"
cp -a "${SWE_SOURCE_ROOT}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron" "${SWE_NODE_ROOT}/mcore-overlay/"
test -f "${SWE_NODE_ROOT}/mcore-overlay/megatron/core/datasets/helpers.cpp"
command -v apptainer
apptainer --version
readonly vllm_python=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python
test -x "${vllm_python}"
"${vllm_python}" -c 'import torch, vllm, flashinfer; assert torch.cuda.is_available(); print("runtime", torch.__version__, vllm.__version__, flashinfer.__version__)'
printf '[SWE-GATE] node prerequisites passed on %s\n' "$(hostname)"
