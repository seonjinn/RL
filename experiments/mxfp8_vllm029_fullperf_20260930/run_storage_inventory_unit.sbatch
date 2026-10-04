#!/usr/bin/env bash
set -euo pipefail
: "${REPO:?Set the committed diagnostic worktree}"
: "${CONTAINER:?Set the certified runtime-aligned image}"
SRUN_BIN=${SRUN_BIN:-/cm/local/apps/slurm/current/bin/srun}
SCRATCH=/raid/scratch/${USER}/storage-unit-${SLURM_JOB_ID}
EXPECT_RED=${EXPECT_RED:-0}
mkdir -p "${SCRATCH}"
"${SRUN_BIN}" --nodes=1 --ntasks=1 --container-image="${CONTAINER}" \
  --container-mounts="${REPO}:${REPO}:ro,/raid/scratch:/raid/scratch" \
  bash -c "set -euo pipefail; cd '${REPO}'; \
    export TMPDIR='${SCRATCH}' PYTHONPYCACHEPREFIX='${SCRATCH}/pycache'; \
    export PYTHONDONTWRITEBYTECODE=1; \
    run_tests() { /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python \
      -c 'import sys, site; site.addsitedir(\"/opt/nemo_rl_venv/lib/python3.13/site-packages\"); import pytest; raise SystemExit(pytest.main(sys.argv[1:]))' \
      --noconftest -q --maxfail=0 --cache-clear -o cache_dir='${SCRATCH}/pytest' \
      experiments/mxfp8_vllm029_fullperf_20260930/test_storage_inventory.py; }
    if [[ '${EXPECT_RED}' == 1 ]]; then
      if run_tests > '${SCRATCH}/red.log' 2>&1; then
        cat '${SCRATCH}/red.log'; exit 2
      fi
      cat '${SCRATCH}/red.log'
      grep -q '5 failed' '${SCRATCH}/red.log'
      grep -q 'Read-only storage inventory is not implemented' '${SCRATCH}/red.log'
      printf 'TEST_FIRST_RED_CONFIRMED\\n'
      for ((attempt=0; attempt<15; attempt++)); do
        [[ -f nemo_rl/utils/storage_inventory.py ]] && break
        sleep 60
      done
      [[ -f nemo_rl/utils/storage_inventory.py ]] || exit 3
    fi
    run_tests"
