#!/usr/bin/env bash

set -euo pipefail

REPO=${REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
arguments=$(awk '/^COMMON_SRUN_ARGS="/ { emit=1 } /^# Number of CPUs per worker node/ { exit } emit { print }' "${REPO}/ray.sub")
export GRES_ARG=--gres=gpu:4 MOUNTS=/lustre:/lustre CONTAINER=test.sqsh
export SLURM_SUBMIT_DIR=/home/test SLURM_JOB_PARTITION=batch SLURM_JOB_ACCOUNT=test

for memory in 0 65536; do
  SLURM_MEM_PER_NODE=${memory}
  eval "${arguments}"
  if [[ " ${COMMON_SRUN_ARGS} " != *" --mem=${memory} "* ]]; then
    echo "Ray steps must inherit the explicit per-node allocation: ${memory}" >&2
    exit 1
  fi
done

unset SLURM_MEM_PER_NODE
RAY_STEP_MEM_PER_NODE=0
eval "${arguments}"
if [[ " ${COMMON_SRUN_ARGS} " != *' --mem=0 '* ]]; then
  echo "Slurm omits SLURM_MEM_PER_NODE for --mem=0; the explicit step budget is required" >&2
  exit 1
fi

unset RAY_STEP_MEM_PER_NODE
eval "${arguments}"
if [[ " ${COMMON_SRUN_ARGS} " == *' --mem='* ]]; then
  echo "A per-CPU/default memory allocation must not be replaced by all node memory" >&2
  exit 1
fi
printf 'Ray step memory inheritance passed.\n'
