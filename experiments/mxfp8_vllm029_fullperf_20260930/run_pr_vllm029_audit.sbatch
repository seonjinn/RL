#!/usr/bin/env bash

set -euo pipefail

: "${REPO:?Set REPO to a NeMo-RL repository containing the requested SHAs}"
: "${CONTAINER:?Set CONTAINER to the pinned vLLM 0.29 nightly image}"
: "${PR_CASES:?Set PR_CASES to newline-separated pr|sha|comma-separated-tests records}"
SRUN_BIN=${SRUN_BIN:-/cm/local/apps/slurm/current/bin/srun}
[[ -x "${SRUN_BIN}" ]] || { echo "Missing srun: ${SRUN_BIN}" >&2; exit 127; }

SCRATCH_ROOT="/raid/scratch/${USER}/pr-vllm029-audit/${SLURM_JOB_ID}"
RAY_ROOT="/tmp/r${SLURM_JOB_ID}"
mkdir -p "${SCRATCH_ROOT}"
COMMON_DIR=$(git -C "${REPO}" rev-parse --path-format=absolute --git-common-dir)
MOUNTS="${REPO}:${REPO}:ro,/raid/scratch:/raid/scratch"
if [[ "${COMMON_DIR}" != "${REPO}/"* ]]; then
  MOUNTS+=",${COMMON_DIR}:${COMMON_DIR}:ro"
fi

"${SRUN_BIN}" --nodes=1 --ntasks=1 \
  --container-image="${CONTAINER}" \
  --container-mounts="${MOUNTS}" \
  bash -lc "
    set -euo pipefail
    export HOME='${SCRATCH_ROOT}/home'
    export TMPDIR='${RAY_ROOT}'
    export RAY_TMPDIR='${RAY_ROOT}'
    export UV_CACHE_DIR='${SCRATCH_ROOT}/uv'
    export TORCHINDUCTOR_CACHE_DIR='${SCRATCH_ROOT}/inductor'
    export TRITON_CACHE_DIR='${SCRATCH_ROOT}/triton'
    export HF_HUB_OFFLINE=1
    export WANDB_MODE=disabled
    unset WANDB_API_KEY HF_TOKEN GH_TOKEN GITHUB_TOKEN SSH_AUTH_SOCK
    VLLM_PY=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python
    \"\${VLLM_PY}\" -c 'import importlib.metadata as m; print({p: m.version(p) for p in (\"torch\", \"vllm\", \"flashinfer-python\", \"ray\")}); assert m.version(\"vllm\") == \"0.29.0\"'
    mkdir -p \"\${HOME}\" \"\${TMPDIR}\" \"\${UV_CACHE_DIR}\"

    failures=()
    while IFS='|' read -r pr sha test_csv; do
      [[ -n \"\${pr}\" ]] || continue
      case_root='${SCRATCH_ROOT}/src-pr'\"\${pr}\"
      mkdir -p \"\${case_root}\"
      git -C '${REPO}' archive \"\${sha}\" | tar -x -C \"\${case_root}\"
      IFS=',' read -r -a tests <<< \"\${test_csv}\"
      echo \"===== PR #\${pr} @ \${sha} =====\"
      if (
        cd \"\${case_root}\"
        export PYTHONPATH=\"\${case_root}:${REPO}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${REPO}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM\"
        timeout --kill-after=30s 900s \"\${VLLM_PY}\" -c '
import site
import sys

# Use the actor interpreter so its .pth files and CuTeDSL bindings are active.
site.addsitedir(\"/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/lib/python3.13/site-packages\")
site.addsitedir(\"/opt/nemo_rl_venv/lib/python3.13/site-packages\")
import pytest

raise SystemExit(pytest.main(sys.argv[1:]))
' -q --maxfail=0 -m 'not hf_gated' \"\${tests[@]}\"
      ); then
        echo \"===== PR #\${pr}: PASS =====\"
      else
        failures+=(\"\${pr}\")
        echo \"===== PR #\${pr}: FAIL =====\" >&2
      fi
      rm -rf \"\${case_root}\"
    done <<< '${PR_CASES}'

    if (( \${#failures[@]} > 0 )); then
      printf 'Failed PRs: %s\\n' \"\${failures[*]}\" >&2
      exit 1
    fi
  "

rm -rf "${SCRATCH_ROOT}" "${RAY_ROOT}"
