#!/bin/bash

set -euo pipefail

if [[ "$#" -ne 6 ]]; then
  echo 'usage: pr11_per_node_launcher.sh SOURCE_ROOT NODE_SCRATCH_ROOT DURABLE_RESULT_ROOT EXPECTED_HEAD EXPECTED_UV_LOCK_SHA BOOTSTRAP' >&2
  exit 64
fi

SOURCE_ROOT=$1
NODE_SCRATCH_ROOT=$2
DURABLE_RESULT_ROOT=$3
EXPECTED_HEAD=$4
EXPECTED_UV_LOCK_SHA=$5
bootstrap=$6

test -x /bin/bash
test -f "${bootstrap}"

export SOURCE_ROOT
export NODE_SCRATCH_ROOT
export DURABLE_RESULT_ROOT
export EXPECTED_HEAD
export EXPECTED_UV_LOCK_SHA

exec /bin/bash "${bootstrap}"
