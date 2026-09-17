#!/usr/bin/env bash

set -euo pipefail

SOURCE_ROOT=${1:?Pass the NeMo-RL source root}
WORKER_PYTHON=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python

export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/tests/unit/models/policy:${SOURCE_ROOT}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${SOURCE_ROOT}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"

cd "${SOURCE_ROOT}"
"${WORKER_PYTHON}" tests/unit/models/policy/test_reference_snapshot.py
"${WORKER_PYTHON}" tests/unit/models/policy/test_reference_swap_lifecycle.py
"${WORKER_PYTHON}" tests/unit/models/policy/test_reference_swap_ddp_gpu.py

cd "${SOURCE_ROOT}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
"${WORKER_PYTHON}" tests/unit_tests/distributed/test_borrowed_cpu_snapshot.py
