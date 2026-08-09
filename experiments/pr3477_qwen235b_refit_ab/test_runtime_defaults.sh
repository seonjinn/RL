#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SUBMIT_SCRIPT=${SCRIPT_DIR}/submit_gcp_nrt.sh

grep -Fq 'nemo-rl-nightly-cw-fallback-20260808' "${SUBMIT_SCRIPT}"
grep -Fq 'nemo_rl_nightly_20260805_15171871.sqsh' "${SUBMIT_SCRIPT}"
grep -Fq 'NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS:-true}' "${SUBMIT_SCRIPT}"
grep -Fq 'WORKER_VENV_ROOT=${WORKER_VENV_ROOT:-/tmp/nemo-rl-pr3477-qwen235b-${RUN_SUFFIX}-${MODE}-workers}' "${SUBMIT_SCRIPT}"
grep -Fq 'NEMO_RL_VENV_DIR=${WORKER_VENV_ROOT}' "${SUBMIT_SCRIPT}"
grep -Fq 'uv run --frozen examples/run_grpo.py' "${SUBMIT_SCRIPT}"

if grep -Fq 'UV_PYTHON_INSTALL_DIR=' "${SUBMIT_SCRIPT}"; then
  echo "submit script must not override the container Python installation" >&2
  exit 1
fi
