#!/usr/bin/env bash
#SBATCH --job-name=adaptive-task5a-gate
#SBATCH --account=nemotron_n3_post
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/adaptive-task5a-1b9ec8b3-gate/slurm-%j.out
#SBATCH --error=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/adaptive-task5a-1b9ec8b3-gate/slurm-%j.err

set -euo pipefail

SOURCE=/home/sna/nemorl-task5a-1b9ec8b3
EXPECTED_SHA=1b9ec8b3b12f57d7b691a68ba220a7ceb22c7af1
IMAGE=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260818_20260818_6296116.sqsh
IMAGE_SHA256=6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44
RAY_ENV_HELPER=/home/sna/task5a-ray-shorttmp-harness/experiments/oci_task5a_ray_shorttmp_20260822/task5a_ray_env.sh
EXPECTED_RAY_ENV_SHA=2b31f7bde7cc0dbe2a6e97849212071e7da17e89c89b27ca6a72a0926a651d87
FINAL_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/adaptive-task5a-1b9ec8b3-gate
LIVE_ROOT=/raid/scratch/${USER}/${SLURM_JOB_ID}/adaptive-task5a
: "${HARNESS_SHA256:?submit with the verified harness SHA256}"

mkdir -p "${LIVE_ROOT}" "${FINAL_ROOT}"
started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
stage=init
result=FAIL

on_exit() {
  rc=$?
  finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
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
    echo "harness_sha256=${HARNESS_SHA256}"
    echo "ray_env_helper=${RAY_ENV_HELPER}"
    echo "ray_env_helper_sha256=${EXPECTED_RAY_ENV_SHA}"
    echo "account=${SLURM_JOB_ACCOUNT:-unknown}"
    echo "scope=ruff-format-pycompile-10-files,mcore-only-actual-wrapper-and-related,policy-split-wrappers,2-rank-consensus"
    echo "excluded=hf-gated-unrelated-tests"
    echo "started_at=${started_at}"
    echo "finished_at=${finished_at}"
  } > "${LIVE_ROOT}/result-${SLURM_JOB_ID}.txt"
  cp "${LIVE_ROOT}/result-${SLURM_JOB_ID}.txt" "${FINAL_ROOT}/result-${SLURM_JOB_ID}.txt"
  if [[ ${rc} -ne 0 ]]; then
    {
      tail -n 180 "${FINAL_ROOT}/slurm-${SLURM_JOB_ID}.out" || true
      tail -n 180 "${FINAL_ROOT}/slurm-${SLURM_JOB_ID}.err" || true
    } > "${FINAL_ROOT}/failure-tail-${SLURM_JOB_ID}.txt"
  fi
  exit "${rc}"
}
trap on_exit EXIT
test "$(sha256sum "${RAY_ENV_HELPER}" | awk '{print $1}')" = "${EXPECTED_RAY_ENV_SHA}"

verify_source() {
  test "$(git -C "${SOURCE}" rev-parse HEAD)" = "${EXPECTED_SHA}"
  test -z "$(git -C "${SOURCE}" status --porcelain --untracked-files=all)"
  git -C "${SOURCE}" submodule status --recursive \
    | awk 'substr($0, 1, 1) != " " { bad=1 } END { exit bad }'
  git -C "${SOURCE}" submodule foreach --quiet --recursive \
    'test -z "$(git status --porcelain --untracked-files=all)"'
}

container_args=(
  --container-image="${IMAGE}"
  --container-mounts=/home:/home,/raid/scratch:/raid/scratch,/lustre:/lustre
  --mpi=pmix
)

common_env='set -euo pipefail; export UV_CACHE_DIR=/raid/scratch/${USER}/${SLURM_JOB_ID}/adaptive-task5a/uv-cache; export XDG_CACHE_HOME=/raid/scratch/${USER}/${SLURM_JOB_ID}/adaptive-task5a/xdg-cache; export TORCH_EXTENSIONS_DIR=/raid/scratch/${USER}/${SLURM_JOB_ID}/adaptive-task5a/torch-extensions; export TRITON_CACHE_DIR=/raid/scratch/${USER}/${SLURM_JOB_ID}/adaptive-task5a/triton-cache; export CUDA_CACHE_PATH=/raid/scratch/${USER}/${SLURM_JOB_ID}/adaptive-task5a/cuda-cache; export RUFF_CACHE_DIR=/raid/scratch/${USER}/${SLURM_JOB_ID}/adaptive-task5a/ruff-cache; export PYTHONPYCACHEPREFIX=/raid/scratch/${USER}/${SLURM_JOB_ID}/adaptive-task5a/pycache; export UV_PROJECT_ENVIRONMENT=/raid/scratch/${USER}/${SLURM_JOB_ID}/adaptive-task5a/venv; export DURABLE_RESULT_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/adaptive-task5a-1b9ec8b3-gate; source /home/sna/task5a-ray-shorttmp-harness/experiments/oci_task5a_ray_shorttmp_20260822/task5a_ray_env.sh; mkdir -p "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$RUFF_CACHE_DIR" "$PYTHONPYCACHEPREFIX" "$UV_PROJECT_ENVIRONMENT"; cd /home/sna/nemorl-task5a-1b9ec8b3; test "$(git rev-parse HEAD)" = 1b9ec8b3b12f57d7b691a68ba220a7ceb22c7af1; test -z "$(git status --porcelain --untracked-files=all)"'
cudnn_env='cudnn_lib="${UV_PROJECT_ENVIRONMENT}/lib/python3.13/site-packages/nvidia/cudnn/lib"; test -e "${cudnn_lib}/libcudnn.so.9"; export CUDNN_HOME="${cudnn_lib}"; export CUDNN_PATH="${cudnn_lib}"; export LD_LIBRARY_PATH="${cudnn_lib}:${LD_LIBRARY_PATH:-}"'

stage=verify-source-before
verify_source

stage=bootstrap-mcore-transformer-engine
srun --nodes=1 --ntasks=1 "${container_args[@]}" bash -lc "export TASK5A_STAGE=bootstrap-mcore-transformer-engine; ${common_env}; uv sync --locked --extra mcore --group test; ${cudnn_env}; uv run --no-sync python - <<'PY'
import torch
import transformer_engine
from megatron.core import parallel_state

print('TASK5A_MCORE_TE_PRECHECK', torch.__version__, transformer_engine.__file__)
print('TASK5A_MCORE_PARALLEL_STATE', parallel_state.__file__)
PY
task5a_ray_local_smoke \"\${UV_PROJECT_ENVIRONMENT}/bin/python\""

affected_files=(
  nemo_rl/models/megatron/draft/optimizer.py
  nemo_rl/models/policy/workers/megatron_policy_worker.py
  nemo_rl/models/policy/lm_policy.py
  nemo_rl/models/policy/tq_policy.py
  tests/unit/models/megatron/test_draft_optimizer.py
  tests/unit/models/megatron/test_draft_optimizer_suspension.py
  tests/unit/models/megatron/test_megatron_worker.py
  tests/unit/models/megatron/test_megatron_split_state.py
  tests/unit/distributed/test_draft_cadence_consensus.py
  tests/unit/models/policy/test_split_api_wrappers.py
)
affected_args=$(printf ' %q' "${affected_files[@]}")

stage=static-10-files
srun --nodes=1 --ntasks=1 "${container_args[@]}" bash -lc "export TASK5A_STAGE=static-10-files; ${common_env}; ${cudnn_env}; uv run --no-sync ruff check${affected_args}; uv run --no-sync ruff format --check${affected_args}; uv run --no-sync python -m py_compile${affected_args}"

stage=focused-mcore-only
srun --nodes=1 --ntasks=1 "${container_args[@]}" bash -lc "export TASK5A_STAGE=focused-mcore-only; ${common_env}; ${cudnn_env}; uv run --no-sync python -m pytest -q -p no:cacheprovider --mcore-only tests/unit/models/megatron/test_draft_optimizer.py tests/unit/models/megatron/test_draft_optimizer_suspension.py tests/unit/models/megatron/test_megatron_worker.py tests/unit/models/megatron/test_megatron_split_state.py"

stage=related-policy-wrappers
srun --nodes=1 --ntasks=1 "${container_args[@]}" bash -lc "export TASK5A_STAGE=related-policy-wrappers; ${common_env}; ${cudnn_env}; uv run --no-sync python -m pytest -q -p no:cacheprovider tests/unit/models/policy/test_split_api_wrappers.py"

stage=two-rank-consensus
master_port=$((20000 + SLURM_JOB_ID % 20000))
srun --nodes=1 --ntasks=1 "${container_args[@]}" bash -lc "export TASK5A_STAGE=two-rank-consensus; ${common_env}; ${cudnn_env}; uv run --no-sync python -m torch.distributed.run --nnodes=1 --node-rank=0 --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=${master_port} -m pytest -q -p no:cacheprovider tests/unit/distributed/test_draft_cadence_consensus.py"

stage=verify-source-after
verify_source

stage=complete
