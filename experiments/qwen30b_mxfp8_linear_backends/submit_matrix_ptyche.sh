#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}

for backend in \
    flashinfer_cutedsl \
    flashinfer_cutlass \
    flashinfer_trtllm \
    flashinfer_trtllm_adaptive; do
    BACKEND="${backend}" RUN_ID="${RUN_ID}" "${SCRIPT_DIR}/submit_ptyche.sh"
done
