#!/usr/bin/env bash

set -euo pipefail

ACTION=${ACTION:-render}
RUN_GROUP=${RUN_GROUP:-$(date +%Y%m%d-%H%M%S)}
BRANCH=${BRANCH:-sna/exp-pr3294-fused-refit-spike-20260823}
EXPECTED_HEAD=${EXPECTED_HEAD:-}
ROOT=${ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna}
REPO=${REPO:-/home/sna/RL-pr3294-fused-refit-spike-20260823}
CONTAINER=${CONTAINER:-${ROOT}/containers/nemo_rl_nightly_20260818_20260818_6296116.sqsh}
RESULT_ROOT=${RESULT_ROOT:-${ROOT}/experiments/mxfp8-fused-refit-spike}
RUN_ROOT=${RUN_ROOT:-${RESULT_ROOT}/${RUN_GROUP}/microbench}
ACCOUNT=${SLURM_ACCOUNT:-nemotron_n3_post}
PARTITION=${PARTITION:-batch}
LOCAL_SCRATCH=${LOCAL_SCRATCH:-/raid/scratch/sna}

if [[ "${ACTION}" == render ]]; then
  printf 'model=Qwen3-30B-A3B\nshape=E128-I768-K2048\nbenchmark=current-vs-direct-layout\n'
  exit 0
fi
case "${ACTION}" in
  test-only) SBATCH_ACTION=(--test-only) ;;
  submit) SBATCH_ACTION=() ;;
  *) echo "ACTION must be render, test-only, or submit" >&2; exit 2 ;;
esac

git -C "${REPO}" fetch origin "${BRANCH}"
REMOTE_HEAD=$(git -C "${REPO}" rev-parse "origin/${BRANCH}")
if [[ $(git -C "${REPO}" branch --show-current) == "${BRANCH}" ]]; then
  git -C "${REPO}" merge --ff-only "${REMOTE_HEAD}"
else
  git -C "${REPO}" checkout --detach "${REMOTE_HEAD}"
fi
LOCAL_HEAD=$(git -C "${REPO}" rev-parse HEAD)
if [[ -n "${EXPECTED_HEAD}" ]]; then
  test "${LOCAL_HEAD}" = "${EXPECTED_HEAD}"
fi
git -C "${REPO}" submodule update --init --recursive \
  3rdparty/Gym-workspace/Gym
test -f "${REPO}/3rdparty/Gym-workspace/Gym/pyproject.toml"
test -e "${CONTAINER}"
mkdir -p "${RUN_ROOT}"

VENV_KEY=${VENV_KEY:-${LOCAL_HEAD}}
VENV=${LOCAL_SCRATCH}/nemo-rl-worker-cache/mxfp8-layout-microbench-${VENV_KEY}
BENCHMARK=${BENCHMARK:-benchmark_direct_expert_layout.py}
COMMAND="set -euo pipefail; cd ${REPO}; export UV_PROJECT_ENVIRONMENT=${VENV}; export UV_CACHE_DIR=${LOCAL_SCRATCH}/uv-cache; export UV_PYTHON_INSTALL_DIR=${LOCAL_SCRATCH}/uv-python; mkdir -p \"\${UV_CACHE_DIR}\" \"\${UV_PYTHON_INSTALL_DIR}\"; uv sync --locked --extra vllm --no-install-project; PYTHONPATH=${REPO} ${VENV}/bin/python experiments/mxfp8_fused_refit_spike/${BENCHMARK} | tee ${RUN_ROOT}/result.json"
export COMMAND CONTAINER
export MOUNTS=/home:/home,/lustre:/lustre,/raid/scratch:/raid/scratch

SBATCH_ARGS=(
  --nodes=1
  --ntasks=1
  --gres=gpu:4
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --time=00:30:00
  --job-name="${ACCOUNT}-mxfp8-layout-microbench"
  --output="${RUN_ROOT}/slurm-%j.out"
)
if [[ -n "${NODELIST:-}" ]]; then
  SBATCH_ARGS+=(--nodelist="${NODELIST}")
fi

exec sbatch "${SBATCH_ACTION[@]}" "${SBATCH_ARGS[@]}" \
  --wrap='srun --container-image="$CONTAINER" --container-mounts="$MOUNTS" bash -lc "$COMMAND"'
