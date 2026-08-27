#!/bin/bash

# ===== BEGIN CONFIG =====
NUM_NODES=32
GPUS_PER_NODE=4
SEGMENT_SIZE=8
STEPS_PER_RUN=20
MAX_STEPS=20
NUM_RUNS=1
NUM_MINUTES=300
JOB_REAPER_COMMENT='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"disproportionate_resource_requirement","description":"Async GRPO has long GPU-idle phases during Ray init and model loading"}}'
# ===== END CONFIG =====

export NCCL_NVLS_ENABLE=0
CONFIG_REL=examples/configs/recipes/llm/performance/grpo-nemotron3-super-120BA12B-32n4g-async-1off-mxfp8-rollout.yaml
MODEL_OVERRIDES=()

source "$(dirname -- "${BASH_SOURCE[0]}")/pr3865_refit_ab_common.sh"
