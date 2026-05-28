#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMORL="${NEMORL:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
NEMORL="$(cd "${NEMORL}" && pwd)"

if [[ -f "${NEMORL}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${NEMORL}/.env"
  set +a
fi

CACHE_DIR="${IMAGE_GRPO_CACHE_DIR:-${CACHE_DIR:-${NEMORL}/.cache/mmpr_tiny}}"
CONTAINER_ROOT="${CONTAINER_ROOT:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/hanrongy/project/nemotron_omni/rl/images}"
CONTAINER="${CONTAINER:-${CONTAINER_ROOT}/super-omni-vllm20-super-vlm2-20260507-0905b74.sqsh}"
MOUNTS="${MOUNTS:-/lustre:/lustre,/home}"

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:?Set SBATCH_ACCOUNT or define it in ${NEMORL}/.env}"
SBATCH_PARTITION="${SBATCH_PARTITION:-${PARTITION:-batch_block1}}"
SBATCH_TIME="${SBATCH_TIME:-1:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
PREP_GPUS="${PREP_GPUS:-1}"
JOB_NAME="${JOB_NAME:-prepare-mmpr-tiny-cache-$(date +%Y%m%d-%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-${NEMORL}/../jobs}"
RESULTS_DIR="${RESULTS_ROOT}/${JOB_NAME}"
LOGS_DIR="${LOGS_DIR:-${RESULTS_DIR}/logs}"
mkdir -p "${LOGS_DIR}" "${CACHE_DIR}"

JOB_SCRIPT="${RESULTS_DIR}/prepare_mmpr_tiny_cache.sbatch"
cat > "${JOB_SCRIPT}" <<'JOB'
#!/usr/bin/env bash
set -euo pipefail

echo "Preparing MMPR-Tiny cache"
echo "  cache_dir=${CACHE_DIR}"
echo "  nemorl=${NEMORL}"
echo "  container=${CONTAINER}"

srun \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
  --container-image="${CONTAINER}" \
  --container-mounts="${MOUNTS}" \
  --no-container-mount-home \
  --container-workdir="${NEMORL}" \
  bash -lc '
set -euo pipefail
export PYTHONPATH="${NEMORL}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${PYTHON_BIN:-/opt/nemo_rl_venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN=python
fi
"${PYTHON_BIN}" - <<PY
import os
from nemo_rl.data.datasets.response_datasets.mmpr_tiny import _ensure_mmpr_cached

cache_dir = os.environ["CACHE_DIR"]
_ensure_mmpr_cached(cache_dir)
print(f"MMPR-Tiny cache ready: {cache_dir}", flush=True)
PY
'

test -f "${CACHE_DIR}/.mmpr_ready"
JOB
chmod +x "${JOB_SCRIPT}"

export CACHE_DIR NEMORL CONTAINER MOUNTS

echo "Submitting MMPR-Tiny cache prepare job"
echo "  job_name=${JOB_NAME}"
echo "  account=${SBATCH_ACCOUNT}"
echo "  partition=${SBATCH_PARTITION}"
echo "  gpus=${PREP_GPUS}"
echo "  cache_dir=${CACHE_DIR}"
echo "  logs=${LOGS_DIR}"

sbatch \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --gres="gpu:${PREP_GPUS}" \
  --account="${SBATCH_ACCOUNT}" \
  --job-name="${JOB_NAME}" \
  --partition="${SBATCH_PARTITION}" \
  --time="${SBATCH_TIME}" \
  --output="${LOGS_DIR}/%x_%j.log" \
  --export=ALL,CACHE_DIR,NEMORL,CONTAINER,MOUNTS \
  "${JOB_SCRIPT}"
