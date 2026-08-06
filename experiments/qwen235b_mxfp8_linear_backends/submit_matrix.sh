#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ACTION=${ACTION:-dry-run}
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}

for backend in flashinfer_cutedsl flashinfer_cutlass; do
    ACTION="${ACTION}" \
        BACKEND="${backend}" \
        DEPENDENCY_JOB_ID= \
        RUN_ID="${RUN_ID}" \
        "${SCRIPT_DIR}/submit_cluster.sh"
done
