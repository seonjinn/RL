#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=02:00:00
#SBATCH --job-name=build_pr3294_actor_venvs

set -euo pipefail

: "${REPO:?REPO is required}"
: "${CONTAINER:?CONTAINER is required}"
: "${CACHE_ROOT:?CACHE_ROOT is required}"
: "${SHARED_UV_CACHE:?SHARED_UV_CACHE is required}"
: "${RAY_BOOTSTRAP_ARCHIVE:?RAY_BOOTSTRAP_ARCHIVE is required}"

EXPECTED_REPO_SHA=${EXPECTED_REPO_SHA:-}
BUILD_TARGETS=${BUILD_TARGETS:-mcore}
BOOTSTRAP_ROOT=/tmp/nrl-actor-venv-bootstrap-${SLURM_JOB_ID}
VENV_ROOT=${CACHE_ROOT}/venvs

test -d "${REPO}/.git"
test -f "${CONTAINER}"
test -f "${RAY_BOOTSTRAP_ARCHIVE}"
if [[ -n "${EXPECTED_REPO_SHA}" ]]; then
  test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_SHA}"
fi

mkdir -p "${CACHE_ROOT}" "${SHARED_UV_CACHE}" "${VENV_ROOT}"
build_command_file=${CACHE_ROOT}/build-actor-venvs-${SLURM_JOB_ID}.sh
trap 'rm -f "${build_command_file}"' EXIT

cat >"${build_command_file}" <<EOF
set -euo pipefail
rm -rf '${BOOTSTRAP_ROOT}' '${BOOTSTRAP_ROOT}.tmp'
mkdir -p '${BOOTSTRAP_ROOT}.tmp'
tar -xzf '${RAY_BOOTSTRAP_ARCHIVE}' \
  -C '${BOOTSTRAP_ROOT}.tmp' --strip-components=1
mv '${BOOTSTRAP_ROOT}.tmp' '${BOOTSTRAP_ROOT}'
test -x '${BOOTSTRAP_ROOT}/bin/python'
test -x '${BOOTSTRAP_ROOT}/bin/uv'
export UV_PYTHON='${BOOTSTRAP_ROOT}/bin/python3.13'
export UV_CACHE_DIR='${SHARED_UV_CACHE}'
export NVTE_CUDA_ARCHS='100'
export TORCH_CUDA_ARCH_LIST='10.0'
export CMAKE_BUILD_PARALLEL_LEVEL='32'
export MAX_JOBS='32'
cd '${REPO}'
for target in ${BUILD_TARGETS}; do
  case "\${target}" in
    mcore)
      worker='nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker'
      extra='mcore'
      ;;
    vllm)
      worker='nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker'
      extra='vllm'
      ;;
    *)
      echo "Unknown BUILD_TARGETS entry: \${target}" >&2
      exit 2
      ;;
  esac
  export UV_PROJECT_ENVIRONMENT='${VENV_ROOT}/'"\${worker}"
  '${BOOTSTRAP_ROOT}/bin/uv' run --locked --extra "\${extra}" \
    --directory '${REPO}' python -c \
    "import nemo_rl; print('validated ${REPO}', nemo_rl.__file__)"
  test -x "\${UV_PROJECT_ENVIRONMENT}/bin/python"
  echo "Finished creating actor venv \${UV_PROJECT_ENVIRONMENT}"
done
rm -rf '${BOOTSTRAP_ROOT}'
EOF
chmod +x "${build_command_file}"

srun \
  --no-container-mount-home \
  --mpi=pmix \
  --container-image="${CONTAINER}" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="${REPO}" \
  bash "${build_command_file}"
