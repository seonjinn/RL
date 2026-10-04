#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?Set the committed diagnostic checkout}"
: "${EXPECTED_SOURCE_SHA:?Pin the full diagnostic SHA}"
: "${CONTAINER:?Set the certified vLLM 0.29 image}"
: "${RESULT_ROOT:?Set the existing comparison result directory}"
: "${SOURCE_ARCHIVE_OVERRIDE:?Set the immutable base-plus-probe archive}"
: "${SOURCE_ARCHIVE_SHA256:?Pin the probe archive SHA256}"
: "${UNIT_TEST_LOG:?Set the successful GB200 unit-test log}"
grep -q '16 passed' "${UNIT_TEST_LOG}"
[[ "$(git -C "${REPO}" rev-parse HEAD)" == "${EXPECTED_SOURCE_SHA}" ]]
export SOURCE_PAYLOAD_SHA=${EXPECTED_SOURCE_SHA}
export CLUSTER=oci MODEL=qwen235 MODE=sync TOPOLOGY=default QUANT_SCOPE=moe
export PERFORMANCE_RECIPE=1 PERFORMANCE_PROFILE=runtime-aligned
export MAX_STEPS=3 MOE_BACKEND=flashinfer_trtllm MOE_ROUTER_DTYPE=fp32
export NRL_STORAGE_INVENTORY=1 USE_SHARED_MODEL=1
export NRL_FORCE_REBUILD_VENVS=false ACTOR_VENV_ROOT=/opt/ray_venvs
export NEMO_RL_PY_EXECUTABLES_SYSTEM=0 NRL_DISABLE_NUMA_MEMBIND=1
export PARTITION=batch WALLTIME=04:00:00
export WANDB_HOME=${WANDB_HOME:-/home/${USER}}
export SLURM_ACCOUNT=${SLURM_ACCOUNT:-nemotron_sw_post}
export RUN_GROUP=${RUN_GROUP:-qwen235-storage-20261005}
export SOURCE_ARCHIVE_OVERRIDE SOURCE_ARCHIVE_SHA256

# Only instrumentation and update count differ from the failed Sync cells.
unset DIAGNOSTIC_NUM_PROMPTS_PER_STEP DIAGNOSTIC_NUM_GENERATIONS_PER_PROMPT
unset DIAGNOSTIC_TRAIN_GLOBAL_BATCH_SIZE DIAGNOSTIC_MAX_NEW_TOKENS
unset GPU_MEMORY_UTILIZATION SUPER_GPU_MEMORY_UTILIZATION KV_CACHE_MEMORY_BYTES
unset VLLM_ATTENTION_BACKEND VLLM_BLOCK_SIZE VLLM_ENABLE_PREFIX_CACHING VLLM_ENFORCE_EAGER
unset ASYNC_IN_FLIGHT_WEIGHT_UPDATES ASYNC_RECOMPUTE_KV_CACHE AFTEROK_JOB_ID
unset REFIT_PREQUANTIZE_OVERRIDE REFIT_TRANSPORT_OVERRIDE MODEL_SNAPSHOT_OVERRIDE

mkdir -p "${RESULT_ROOT}/submission"
for arm in mxfp8-false-mxfp8 mxfp8-true-mxfp8; do
  for action in test-only submit; do
    ARM=${arm} ACTION=${action} \
      bash "${REPO}/experiments/mxfp8_vllm029_fullperf_20260930/submit.sh" \
      2>&1 | tee "${RESULT_ROOT}/submission/${RUN_GROUP}-${arm}-${action}.log"
  done
done
