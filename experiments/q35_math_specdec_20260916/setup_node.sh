#!/usr/bin/env bash
set -euo pipefail
mkdir -p "${RAY_TMPDIR}" "${Q35_NODE_ROOT}/tmp" "${Q35_NODE_ROOT}/cache" "${Q35_NODE_ROOT}/hf"
test ! -e "${Q35_NODE_ROOT}/target"
cp -aL "${Q35_TARGET}" "${Q35_NODE_ROOT}/target"
mkdir -p "${Q35_NODE_ROOT}/mcore-overlay"
cp -a "${Q35_SOURCE}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron" "${Q35_NODE_ROOT}/mcore-overlay/"
readonly VLLM_PYTHON=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python
test -x "${VLLM_PYTHON}"
if [[ "${Q35_ARM}" != baseline ]]; then
  readonly DRAFT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/specdec_ptv23/ptv3_swa/sd2p3rp-q35-a3b-ptv3rp25-${Q35_ARM}-b8-16n/exported-checkpoint-44000
  cp -aL "${DRAFT}" "${Q35_NODE_ROOT}/draft"
fi
if [[ "${Q35_ARM}" == dspark ]]; then
  "${VLLM_PYTHON}" "${Q35_SOURCE}/experiments/qwen3_30ba3b_bf16_flashinfer_specdec_latest_main_20260909/prepare_vllm_dspark_fap_overlay.py" --overlay-root "${Q35_NODE_ROOT}/vllm-overlay"
fi
"${VLLM_PYTHON}" -c 'import vllm, torch, flashinfer; print("RUNTIME", vllm.__version__, torch.__version__, flashinfer.__version__)'
df -h "${Q35_NODE_ROOT}"
