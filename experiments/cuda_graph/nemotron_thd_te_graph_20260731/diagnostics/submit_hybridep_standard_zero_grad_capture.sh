#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)

export MCORE_TEST_ROWS=dropless_hybridep_nano16
export MCORE_TEST_VARIANT=hybridep-standard-zero-grad-capture
export MCORE_TEST_CAPTURE_ONLY=1
export MCORE_TEST_ZERO_GRAD_BEFORE_CAPTURE=1
unset MCORE_TEST_DISABLE_NANO_SHARED_EXPERT
unset MCORE_TEST_NANO_CG_SUBMODULE
unset MCORE_TEST_FORWARD_ONLY_MODEL_WARMUP
unset MCORE_TEST_RELEASE_WARMUP_GRAPH
unset MCORE_TEST_HYBRIDEP_MODEL_WARMUP_STAGE
unset MCORE_TEST_LINEAR_MODEL_WARMUP

exec "${script_dir}/../submit_mcore_matrix.sh"
