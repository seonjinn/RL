#!/bin/bash

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)

cd "${PROJECT_ROOT}"

if [[ -n "$(git status --porcelain)" ]]; then
  printf 'Refusing to update a dirty checkout.\n' >&2
  git status --short >&2
  exit 2
fi

git pull --ff-only --recurse-submodules=no
git submodule sync --recursive
git submodule update --init --recursive

if [[ -n "$(git status --porcelain)" ]]; then
  printf 'Refusing to submit from a dirty checkout.\n' >&2
  git status --short >&2
  exit 2
fi

if [[ -n "$(git rev-list '@{upstream}..HEAD')" ]]; then
  printf 'Refusing to submit commits that have not been pushed upstream.\n' >&2
  exit 2
fi

: "${CONTAINER:?CONTAINER is required}"
: "${DRIVER_VENV:?DRIVER_VENV is required}"
: "${UV_CACHE_DIR:?UV_CACHE_DIR is required}"
: "${NEMO_RL_VENV_DIR:?NEMO_RL_VENV_DIR is required}"

for shared_path_name in DRIVER_VENV UV_CACHE_DIR NEMO_RL_VENV_DIR; do
  shared_path=${!shared_path_name}
  case "${shared_path}" in
    /lustre/*) ;;
    *)
      printf '%s must be on shared /lustre storage: %s\n' \
        "${shared_path_name}" "${shared_path}" >&2
      exit 2
      ;;
  esac
done

PARTITION=${PARTITION:-batch}
TIME_LIMIT=${TIME_LIMIT:-02:00:00}
VENV_LOG_DIR=${VENV_LOG_DIR:-"$(dirname -- "${DRIVER_VENV}")/logs"}

FAIRSHARE_ROWS=$(sshare -a --user="$(id -un)" -o Account,User,FairShare -n -P)
read -r AUTO_ACCOUNT AUTO_FAIRSHARE < <(
  awk -F'|' -v user="$(id -un)" '
    $2 == user {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
      if (!seen || $3 + 0 > best) {
        account = $1
        best = $3 + 0
        fairshare = $3
        seen = 1
      }
    }
    END {
      if (account != "") {
        print account, fairshare
      }
    }
  ' <<< "${FAIRSHARE_ROWS}"
) || true
if [[ -z "${AUTO_ACCOUNT:-}" ]]; then
  printf 'Could not resolve a user-level FairShare account.\n' >&2
  exit 2
fi

ACCOUNT=${ACCOUNT:-"${AUTO_ACCOUNT}"}
mkdir -p "${VENV_LOG_DIR}"

export CONTAINER
export DRIVER_VENV
export HYBRID_EP_PROJECT_ROOT="${PROJECT_ROOT}"
export UV_CACHE_DIR
export NEMO_RL_VENV_DIR

sbatch_args=(
  --export=ALL
  --nodes=1
  --ntasks=1
  --gres=gpu:1
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --time="${TIME_LIMIT}"
  --job-name="${ACCOUNT}.hybridep-driver-venv"
  --output="${VENV_LOG_DIR}/prepare-driver-venv-%j.out"
  "${SCRIPT_DIR}/prepare_driver_venv.sbatch"
)

printf 'Validating schedule with sbatch --test-only...\n'
sbatch --test-only "${sbatch_args[@]}"

job_id=$(sbatch --parsable "${sbatch_args[@]}")
printf 'Submitted %s with account %s (FairShare %s).\n' \
  "${job_id}" "${ACCOUNT}" "${AUTO_FAIRSHARE}"
