#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SUBMIT_SCRIPT=${SCRIPT_DIR}/submit_gcp_nrt.sh

grep -Fq 'nemo-rl-nightly-cw-fallback-20260808' "${SUBMIT_SCRIPT}"
grep -Fq 'nemo_rl_nightly_20260805_15171871.sqsh' "${SUBMIT_SCRIPT}"
grep -Fq '/opt/nemo_rl_venv/bin/python examples/run_grpo.py' "${SUBMIT_SCRIPT}"

if grep -Fq 'UV_PYTHON_INSTALL_DIR=' "${SUBMIT_SCRIPT}"; then
  echo "submit script must not override the container Python installation" >&2
  exit 1
fi
