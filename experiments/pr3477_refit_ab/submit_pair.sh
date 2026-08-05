#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_SUFFIX=${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}
export RUN_SUFFIX

for mode in legacy nccl; do
  MODE=${mode} "${SCRIPT_DIR}/submit_gcp_nrt.sh"
done
