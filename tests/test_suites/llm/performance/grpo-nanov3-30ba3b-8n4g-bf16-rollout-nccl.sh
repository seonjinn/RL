#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
source "$SCRIPT_DIR/common.env"

export NCCL_NVLS_ENABLE=0
export RAY_CGRAPH_get_timeout=2400

# ===== BEGIN CONFIG =====
NUM_NODES=8
GPUS_PER_NODE=4
SEGMENT_SIZE=4
STEPS_PER_RUN=20
MAX_STEPS=20
NUM_RUNS=$(((MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN))
NUM_MINUTES=240
# ===== END CONFIG =====

cd "$PROJECT_ROOT"
uv run examples/run_grpo.py \
    --config "$CONFIG_PATH" \
    grpo.max_num_steps=$MAX_STEPS \
    grpo.seed=42 \
    grpo.val_at_start=false \
    grpo.val_at_end=false \
    checkpointing.enabled=false \
    logger.log_dir="$LOG_DIR" \
    logger.wandb_enabled=true \
    logger.wandb.project=nemo-rl \
    logger.wandb.name="$EXP_NAME" \
    logger.monitor_gpus=true \
    logger.tensorboard_enabled=true \
    +policy.generation.vllm_kwargs.distributed_timeout_seconds=2400 \
    "$@" \
    2>&1 | tee "$RUN_LOG"

uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"
