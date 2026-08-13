#!/bin/bash

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

: "${SOURCE_ROOT:?Set SOURCE_ROOT to an exact NeMo-RL checkout}"
: "${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the exact source SHA}"
: "${CONTAINER:?Set CONTAINER to an immutable NeMo-RL image}"
: "${VENV_ROOT:?Set VENV_ROOT to a source-specific Lustre path}"

TEST_EXPR=${TEST_EXPR:-ep_a2a or a2a_overlap or delay_wgrad or schedule_plan}

actual_commit=$(git -C "${SOURCE_ROOT}" rev-parse HEAD)
if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
    echo "Expected ${EXPECTED_COMMIT}, got ${actual_commit}" >&2
    exit 2
fi
if ! git -C "${SOURCE_ROOT}" diff-index --quiet --ignore-submodules=untracked HEAD --; then
    echo "Tracked source changes are not allowed" >&2
    exit 2
fi

printf -v test_command \
    'set -euo pipefail; cd %q; export NRL_IGNORE_VERSION_MISMATCH=1 NRL_FORCE_REBUILD_VENVS=false NEMO_RL_VENV_DIR=%q; uv run --frozen --extra mcore --group test pytest --mcore-only -q tests/unit/models/megatron/test_megatron_setup.py tests/unit/models/megatron/test_train.py -k %q' \
    "${SOURCE_ROOT}" \
    "${VENV_ROOT}" \
    "${TEST_EXPR}"

srun \
    --nodes=1 \
    --ntasks=1 \
    --no-container-mount-home \
    --container-image="${CONTAINER}" \
    --container-mounts=/lustre:/lustre \
    bash -lc "${test_command}"
