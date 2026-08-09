#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SUBMIT_SCRIPT=${SCRIPT_DIR}/submit_gcp_nrt.sh

grep -Fq 'nemo-rl-nightly-cw-fallback-20260808' "${SUBMIT_SCRIPT}"
grep -Fq 'nemo_rl_nightly_20260805_15171871.sqsh' "${SUBMIT_SCRIPT}"
grep -Fq 'NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS:-false}' "${SUBMIT_SCRIPT}"
grep -Fq 'WORKER_VENV_ROOT=${WORKER_VENV_ROOT:-${CACHE_ROOT}/worker-venvs}' "${SUBMIT_SCRIPT}"
grep -Fq 'NEMO_RL_VENV_DIR=${WORKER_VENV_ROOT}' "${SUBMIT_SCRIPT}"
grep -Fq 'UV_CACHE_DIR=${CACHE_ROOT}/uv-cache' "${SUBMIT_SCRIPT}"
grep -Fq 'TOTAL_NODES=${TOTAL_NODES:-16}' "${SUBMIT_SCRIPT}"
grep -Fq 'GPUS_PER_NODE=${GPUS_PER_NODE:-4}' "${SUBMIT_SCRIPT}"
grep -Fq 'GEN_NODES=${GEN_NODES:-8}' "${SUBMIT_SCRIPT}"
grep -Fq 'TRAIN_EP=${TRAIN_EP:-8}' "${SUBMIT_SCRIPT}"
grep -Fq 'VLLM_TP=${VLLM_TP:-4}' "${SUBMIT_SCRIPT}"
grep -Fq 'The reportable A/B requires 16 nodes, 4 GPUs/node, and 8 generation nodes' "${SUBMIT_SCRIPT}"
grep -Fq 'generation_tensor_parallel_size=${VLLM_TP}' "${SUBMIT_SCRIPT}"
grep -Fq 'policy.generation.vllm_cfg.tensor_parallel_size=${VLLM_TP}' "${SUBMIT_SCRIPT}"
grep -Fq 'policy.megatron_cfg.expert_model_parallel_size=${TRAIN_EP}' "${SUBMIT_SCRIPT}"
grep -Fq 'VLLM_ALLREDUCE_USE_SYMM_MEM=${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}' "${SUBMIT_SCRIPT}"
grep -Fq 'export VLLM_ALLREDUCE_USE_SYMM_MEM=${VLLM_ALLREDUCE_USE_SYMM_MEM}' "${SUBMIT_SCRIPT}"
grep -Fq 'vllm_allreduce_use_symm_mem=${VLLM_ALLREDUCE_USE_SYMM_MEM}' "${SUBMIT_SCRIPT}"
grep -Fq 'NRL_ALLOW_PARTIAL_GPU_NODES=${NRL_ALLOW_PARTIAL_GPU_NODES:-1}' "${SUBMIT_SCRIPT}"
grep -Fq 'export NRL_ALLOW_PARTIAL_GPU_NODES' "${SUBMIT_SCRIPT}"
grep -Fq -- '--oversubscribe' "${SUBMIT_SCRIPT}"
grep -Fq 'NRL_ALLOW_PARTIAL_GPU_NODES' "${SCRIPT_DIR}/../../ray.sub"
grep -Fq 'uv run --frozen examples/run_grpo.py' "${SUBMIT_SCRIPT}"

if grep -Fq 'UV_PYTHON_INSTALL_DIR=' "${SUBMIT_SCRIPT}"; then
  echo "submit script must not override the container Python installation" >&2
  exit 1
fi
