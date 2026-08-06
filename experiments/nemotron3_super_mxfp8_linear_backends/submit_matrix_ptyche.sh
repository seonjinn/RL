#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ACTION=${ACTION:-dry-run}
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}

for backend in flashinfer_cutedsl flashinfer_cutlass; do
    experiment_root_args=()
    if [[ -n "${EXPERIMENT_ROOT:-}" ]]; then
        experiment_root_args=(EXPERIMENT_ROOT="${EXPERIMENT_ROOT%/}/${backend}")
    fi
    env "${experiment_root_args[@]}" ACTION="${ACTION}" BACKEND="${backend}" RUN_ID="${RUN_ID}" \
        "${SCRIPT_DIR}/submit_ptyche.sh"
done
