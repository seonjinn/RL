#!/bin/bash
set -euo pipefail

# Run inside the nightly container in a dedicated GB200 allocation.
: "${SOURCE_REPO:?Set the frozen source repository}"
: "${SOURCE_SHA:?Set the tested source revision}"
: "${BRIDGE_SHA:?Set the tested Bridge revision}"
ROOT="/raid/scratch/${USER}/native-mixed-views/${SLURM_JOB_ID}"
BRIDGE="${SOURCE_REPO}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
MCORE="${BRIDGE}/3rdparty/Megatron-LM"
PYTHON=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
mkdir -p "${ROOT}/source" "${ROOT}/bridge" "${ROOT}/mcore"
git -C "${SOURCE_REPO}" archive "${SOURCE_SHA}" nemo_rl tests/unit/models/megatron/test_group_experts.py pyproject.toml | tar -xf - -C "${ROOT}/source"
git -C "${BRIDGE}" archive "${BRIDGE_SHA}" src | tar -xf - -C "${ROOT}/bridge"
MCORE_SHA="${MCORE_SOURCE_SHA:-$(git -C "${BRIDGE}" ls-tree "${BRIDGE_SHA}" 3rdparty/Megatron-LM | awk '{print $3}')}"
git -C "${MCORE}" archive "${MCORE_SHA}" megatron | tar -xf - -C "${ROOT}/mcore"
if [[ "${TEST_MCORE_NAMES:-0}" == 1 ]]; then
  MCORE_TEST_SHA="${MCORE_TEST_SHA:-${MCORE_SHA}}"
  git -C "${MCORE}" archive "${MCORE_TEST_SHA}" tests | tar -xf - -C "${ROOT}/mcore"
  printf 'MCORE_TEST_SHA=%s\n' "${MCORE_TEST_SHA}"
fi
export UV_CACHE_DIR="/raid/scratch/${USER}/uv-cache"
export PYTHONPYCACHEPREFIX="${ROOT}/pycache"
export TRITON_CACHE_DIR="${ROOT}/triton"
export TORCH_EXTENSIONS_DIR="${ROOT}/extensions"
export PYTHONPATH="${ROOT}/deps:${ROOT}/source:${ROOT}/bridge/src:${ROOT}/mcore"
uv pip install --python "${PYTHON}" --target "${ROOT}/deps" pytest
cd "${ROOT}/source"
printf 'SOURCE_SHA=%s\nBRIDGE_SHA=%s\nMCORE_SHA=%s\n' "${SOURCE_SHA}" "${BRIDGE_SHA}" "${MCORE_SHA}"
"${PYTHON}" -P -c 'import torch, megatron.bridge; print(torch.cuda.get_device_name()); print(megatron.bridge.__file__)'
"${PYTHON}" -P -m pytest -q --tb=short -o addopts= tests/unit/models/megatron/test_group_experts.py
if [[ "${TEST_MCORE_NAMES:-0}" == 1 ]]; then
  cd "${ROOT}/mcore"
  export PYTHONPATH="${ROOT}/mcore:${PYTHONPATH}"
  # These self-contained tests need no datasets or upstream autouse fixtures.
  "${PYTHON}" -P -m pytest --noconftest -q --tb=short -o addopts= \
    tests/unit_tests/models/test_gpt_model_module_names.py \
    tests/unit_tests/transformer/test_transformer_block.py::test_transformer_block_propagates_local_module_names \
    tests/unit_tests/transformer/test_transformer_block.py::test_transformer_block_keeps_legacy_layer_constructor_compatible \
    tests/unit_tests/ssm/test_hybrid_block.py::test_all_layer_configs_route_to_matching_specs \
    tests/unit_tests/transformer/test_multi_token_prediction.py::TestMultiTokenPredictionLayer::test_construction_names_match_zero_based_mtp_registration
fi
