#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export RUN_MATRIX=${RUN_MATRIX:-"mxfp8-legacy-receiver-quant:baseline mxfp8-legacy-prequant:baseline mxfp8-legacy-prequant:batched-shuffle mxfp8-legacy-prequant:optimized mxfp8-nccl-prequant:optimized"}
export WANDB_PROJECT=${WANDB_PROJECT:-sna-pr3294-cumulative-ablation}

exec "${SCRIPT_DIR}/submit_prequant_ab.sh"
