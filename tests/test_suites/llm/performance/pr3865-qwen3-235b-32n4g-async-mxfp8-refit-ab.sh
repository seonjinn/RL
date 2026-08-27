#!/bin/bash

# ===== BEGIN CONFIG =====
NUM_NODES=32
GPUS_PER_NODE=4
SEGMENT_SIZE=16
STEPS_PER_RUN=20
MAX_STEPS=20
NUM_RUNS=1
NUM_MINUTES=240
JOB_REAPER_COMMENT='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"disproportionate_resource_requirement","description":"Async GRPO has long GPU-idle phases during Ray init and model loading"}}'
# ===== END CONFIG =====

export NCCL_NVLS_ENABLE=0
export RAY_CGRAPH_get_timeout=2400
CONFIG_REL=examples/configs/recipes/llm/performance/grpo-qwen3-235b-32n4g-async-1off-mxfp8-rollout.yaml
MODEL_OVERRIDES=(
  +policy.generation.vllm_kwargs.distributed_timeout_seconds=2400
  +policy.generation.vllm_cfg.max_num_seqs=32
)

source "$(dirname -- "${BASH_SOURCE[0]}")/pr3865_refit_ab_common.sh"
