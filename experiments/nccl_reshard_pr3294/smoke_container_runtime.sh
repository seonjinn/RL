#!/usr/bin/env bash

set -euo pipefail

REPO=${REPO:?REPO is required}
RAY_BOOTSTRAP_ARCHIVE=${RAY_BOOTSTRAP_ARCHIVE:?RAY_BOOTSTRAP_ARCHIVE is required}
RUNTIME_ROOT=${RUNTIME_ROOT:-/tmp/pr3294-container-runtime}

if [[ ! -x "${RUNTIME_ROOT}/bin/ray" ]]; then
  STAGING_ROOT="${RUNTIME_ROOT}.tmp-$$"
  mkdir -p "${STAGING_ROOT}"
  tar -xzf "${RAY_BOOTSTRAP_ARCHIVE}" \
    -C "${STAGING_ROOT}" --strip-components=1
  mv "${STAGING_ROOT}" "${RUNTIME_ROOT}"
fi

export PYTHONPATH="${REPO}:${RUNTIME_ROOT}/lib/python3.13/site-packages"
/opt/nemo_rl_venv/bin/python -c '
import megatron.core
import modelopt
import ray
import requests
import torch
import urllib3
import vllm

print(
    f"python={__import__('sys').version.split()[0]} "
    f"ray={ray.__version__} torch={torch.__version__} "
    f"vllm={vllm.__version__} requests={requests.__version__} "
    f"urllib3={urllib3.__version__}"
)
'
