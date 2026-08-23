#!/usr/bin/env bash

set -euo pipefail

# ===== BEGIN CONFIG =====
NUM_NODES=16
GPUS_PER_NODE=4
SEGMENT_SIZE=16
STEPS_PER_RUN=4
MAX_STEPS=4
NUM_RUNS=1
NUM_MINUTES=240
# ===== END CONFIG =====

export NTRACE_ARM=mxfp8
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${SCRIPT_DIR}/run_capture.sh" "$@"
