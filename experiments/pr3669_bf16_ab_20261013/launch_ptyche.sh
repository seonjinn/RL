#!/usr/bin/env bash
set -euo pipefail

ROOT=/project/coreai_dlalgo_llm/sna/pr3669-ab-20261013
export BASE_SOURCE="${BASE_SOURCE:-${ROOT}/base}"
export PR_SOURCE="${PR_SOURCE:-${ROOT}/pr}"
export IMAGE="${IMAGE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/mxfp8_backends_20261007/nemo_rl_nightly_vllm029_20261007_2977681.sqsh}"
export HF_HOME="${HF_HOME:-/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home}"
export RESULT_ROOT="${RESULT_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/results/pr3669-bf16-ab-20261013-ptyche}"
export PARTITION="${PARTITION:-36x2-a01r}"

exec bash "${ROOT}/launch_ab.sh"
