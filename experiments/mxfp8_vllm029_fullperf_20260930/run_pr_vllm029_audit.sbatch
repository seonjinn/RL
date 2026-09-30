#!/usr/bin/env bash

set -euo pipefail

: "${REPO:?Set REPO to a NeMo-RL repository containing the requested SHAs}"
: "${CONTAINER:?Set CONTAINER to the pinned vLLM 0.29 nightly image}"
: "${PR_CASES:?Set PR_CASES to newline-separated pr|sha|comma-separated-tests records}"
SRUN_BIN=${SRUN_BIN:-/cm/local/apps/slurm/current/bin/srun}
[[ -x "${SRUN_BIN}" ]] || { echo "Missing srun: ${SRUN_BIN}" >&2; exit 127; }

SCRATCH_ROOT="/raid/scratch/${USER}/pr-vllm029-audit/${SLURM_JOB_ID}"
RAY_ROOT="/tmp/nrl-pr-audit-${SLURM_JOB_ID}"
mkdir -p "${SCRATCH_ROOT}"

"${SRUN_BIN}" --nodes=1 --ntasks=1 \
  --container-image="${CONTAINER}" \
  --container-mounts="/home:/home,/raid/scratch:/raid/scratch,/lustre:/lustre" \
  bash -lc "
    set -euo pipefail
    export HOME='${SCRATCH_ROOT}/home'
    export TMPDIR='${RAY_ROOT}'
    export RAY_TMPDIR='${RAY_ROOT}'
    export UV_CACHE_DIR='${SCRATCH_ROOT}/uv'
    export TORCHINDUCTOR_CACHE_DIR='${SCRATCH_ROOT}/inductor'
    export TRITON_CACHE_DIR='${SCRATCH_ROOT}/triton'
    export HF_HUB_OFFLINE=1
    export PYTHONPATH='/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/lib/python3.13/site-packages:/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/lib/python3.13/site-packages'
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
        export PYTHONPATH=\"\${case_root}:\${PYTHONPATH}:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM\"
        /opt/nemo_rl_venv/bin/python -m pytest -q \"\${tests[@]}\"
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
