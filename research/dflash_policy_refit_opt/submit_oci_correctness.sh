#!/usr/bin/env bash
#SBATCH --job-name=dflash-policy-refit-opt
#SBATCH --partition=batch_long
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${SOURCE:?SOURCE must name a clean source checkout under /home}"
: "${EXPECTED_SHA:?EXPECTED_SHA must be a full commit SHA}"
: "${FINAL_ROOT:?FINAL_ROOT must name a durable directory under /lustre}"

IMAGE=${IMAGE:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260818_20260818_6296116.sqsh}
IMAGE_SHA256=${IMAGE_SHA256:-6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44}
LIVE_ROOT=/raid/scratch/${USER}/dflash-policy-refit-opt-${SLURM_JOB_ID}
CACHE_ROOT=/home/${USER}/.cache/dflash-policy-refit-opt-${EXPECTED_SHA:0:12}

mkdir -p "${LIVE_ROOT}" "${FINAL_ROOT}"
exec > >(tee "${LIVE_ROOT}/live.log") 2>&1

started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
stage=init
result=FAIL

on_exit() {
  rc=$?
  finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  result_file="${LIVE_ROOT}/result-${SLURM_JOB_ID}.txt"
  if [[ ${rc} -eq 0 ]]; then
    result=PASS
  fi
  {
    echo "stage=${stage}"
    echo "result=${result}"
    echo "exit_code=${rc}"
    echo "job_id=${SLURM_JOB_ID}"
    echo "sha=${EXPECTED_SHA}"
    echo "container=${IMAGE}"
    echo "container_sha256=${IMAGE_SHA256}"
    echo "topology=TP2 NCCL DFlash export plus hidden-capture MBS1/MBS2"
    echo "started_at=${started_at}"
    echo "finished_at=${finished_at}"
  } >"${result_file}"
  cp "${result_file}" "${FINAL_ROOT}/result-${SLURM_JOB_ID}.txt"
  cp "${LIVE_ROOT}/live.log" "${FINAL_ROOT}/live-${SLURM_JOB_ID}.log" || true
  exit "${rc}"
}

trap on_exit EXIT
trap 'result=FAIL; exit 124' TERM INT

stage=correctness
srun --nodes=1 --ntasks=1 \
  --container-image="${IMAGE}" \
  --container-mounts=/home:/home,/raid/scratch:/raid/scratch,/lustre:/lustre \
  --mpi=pmix \
  bash -lc "
    set -euo pipefail
    export UV_CACHE_DIR='${CACHE_ROOT}/uv-cache'
    export UV_PROJECT_ENVIRONMENT='${CACHE_ROOT}/venv'
    export PIP_CACHE_DIR='${CACHE_ROOT}/pip-cache'
    export XDG_CACHE_HOME='${CACHE_ROOT}/xdg-cache'
    export TORCH_EXTENSIONS_DIR='${CACHE_ROOT}/torch-extensions'
    export TMPDIR=/tmp/dpro-${SLURM_JOB_ID}
    export RAY_TMPDIR=/tmp/dpro-${SLURM_JOB_ID}
    export NVTE_CUDA_ARCHS=100
    export MAX_JOBS=8
    mkdir -p \"\${UV_CACHE_DIR}\" \"\${UV_PROJECT_ENVIRONMENT}\" \
      \"\${PIP_CACHE_DIR}\" \"\${XDG_CACHE_HOME}\" \
      \"\${TORCH_EXTENSIONS_DIR}\" \"\${TMPDIR}\"

    cd '${SOURCE}'
    test \"\$(git rev-parse HEAD)\" = '${EXPECTED_SHA}'
    test -z \"\$(git status --porcelain)\"
    git submodule status --recursive | \
      awk 'substr(\$0,1,1) ~ /[-+U]/ {bad=1} END {exit bad}'
    test \"\$(sha256sum '${IMAGE}' | awk '{print \$1}')\" = '${IMAGE_SHA256}'

    uv sync --locked --extra mcore --group test
    venv_cudnn_root=\"\${UV_PROJECT_ENVIRONMENT}/lib/python3.13/site-packages/nvidia/cudnn\"
    venv_nvidia_libs=\$(find \
      \"\${UV_PROJECT_ENVIRONMENT}/lib/python3.13/site-packages/nvidia\" \
      -mindepth 2 -maxdepth 2 -type d -name lib -print | paste -sd: -)
    test -f \"\${venv_cudnn_root}/lib/libcudnn.so.9\"
    export CUDNN_HOME=\"\${venv_cudnn_root}\"
    export CUDNN_PATH=\"\${venv_cudnn_root}/lib\"
    export LD_LIBRARY_PATH=\"\${venv_nvidia_libs}:\${LD_LIBRARY_PATH:-}\"

    uv run --no-sync ruff check \
      nemo_rl/models/megatron/draft/hidden_capture.py \
      nemo_rl/models/megatron/draft/utils.py \
      tests/unit/models/megatron/test_hidden_capture.py \
      tests/unit/models/megatron/test_dflash_export_contract.py \
      research/dflash_policy_refit_opt/oci_nccl_gate.py
    uv run --no-sync ruff format --check \
      nemo_rl/models/megatron/draft/hidden_capture.py \
      nemo_rl/models/megatron/draft/utils.py \
      tests/unit/models/megatron/test_hidden_capture.py \
      tests/unit/models/megatron/test_dflash_export_contract.py \
      research/dflash_policy_refit_opt/oci_nccl_gate.py
    uv run --no-sync pyrefly check \
      nemo_rl/models/megatron/draft/hidden_capture.py \
      nemo_rl/models/megatron/draft/utils.py
    uv run --no-sync python -m pytest -q \
      tests/unit/models/megatron/test_hidden_capture.py
    uv run --no-sync python -m pytest -q --mcore-only \
      tests/unit/models/megatron/test_dflash_export_contract.py
    CUDA_VISIBLE_DEVICES=0,1 uv run --no-sync torchrun \
      --standalone --nproc-per-node=2 \
      research/dflash_policy_refit_opt/oci_nccl_gate.py
  "

stage=complete
