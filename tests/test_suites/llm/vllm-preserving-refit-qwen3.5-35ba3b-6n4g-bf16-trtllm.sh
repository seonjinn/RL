#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
export NRL_REFIT_SLEEP_CACHE_ROOT=${NRL_REFIT_SLEEP_CACHE_ROOT:-/raid/scratch/${USER}/refit-sleep-${SLURM_JOB_ID:-manual}}
export NRL_TEST_RUN_DIR="$NRL_REFIT_SLEEP_CACHE_ROOT/qwen35"
source "$SCRIPT_DIR/common.env"

# ===== BEGIN CONFIG =====
NUM_NODES=6
GPUS_PER_NODE=4
SEGMENT_SIZE=2
NUM_RUNS=1
NUM_MINUTES=240
# ===== END CONFIG =====

source "$SCRIPT_DIR/refit_sleep.env"
# Keep the existing BF16 runtime wrapper's logs distinct from outer pytest logs.
export NRL_TEST_RUN_DIR="$EXP_DIR/bf16-control"
timeout --signal=TERM --kill-after=30 13800 uv run --no-sync python -m pytest \
    -o addopts='' -v -s --timeout=13200 --timeout-method=thread \
    --junitxml="$EXP_DIR/junit.xml" \
    tests/functional/test_vllm_refit_sleep.py::test_qwen35_bf16_nccl_reshard_preserving_control \
    2>&1 | tee "$RUN_LOG"
