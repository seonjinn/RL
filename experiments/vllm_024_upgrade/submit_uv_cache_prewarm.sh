#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-test-only}"
REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel)}"
LYRIS_ROOT="${LYRIS_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}"
CONTAINER="${CONTAINER:-${LYRIS_ROOT}/containers/nemo_rl_nightly_20260707.sqsh}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${LYRIS_ROOT}/uv_cache/vllm024}"
RUN_TAG="${RUN_TAG:-nemorl-vllm024-uv-prewarm-20260711}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${LYRIS_ROOT}/experiments/${RUN_TAG}}"
ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
PARTITION="${PARTITION:-gb200}"
WALLTIME="${WALLTIME:-02:00:00}"

case "${MODE}" in
  dry-run|test-only|submit)
    ;;
  *)
    echo "ERROR: mode must be dry-run, test-only, or submit" >&2
    exit 2
    ;;
esac

if [[ "${MODE}" != "dry-run" && ! -f "${CONTAINER}" ]]; then
  echo "ERROR: container not found: ${CONTAINER}" >&2
  exit 2
fi
if [[ "${MODE}" == "submit" ]]; then
  if ! git -C "${REPO_DIR}" diff --quiet --ignore-submodules=dirty \
    || ! git -C "${REPO_DIR}" diff --cached --quiet --ignore-submodules=dirty; then
    echo "ERROR: submit requires a clean tracked checkout" >&2
    exit 2
  fi
  if ! git -C "${REPO_DIR}" ls-files --error-unmatch \
    experiments/vllm_024_upgrade/submit_uv_cache_prewarm.sh >/dev/null 2>&1; then
    echo "ERROR: launcher must be committed before submit" >&2
    exit 2
  fi
  if ! git -C "${REPO_DIR}" branch -r --contains HEAD | grep -q .; then
    echo "ERROR: HEAD is not present on a known remote branch" >&2
    exit 2
  fi
fi

printf -v vllm_command '%q ' \
  env \
  "UV_CACHE_DIR=${UV_CACHE_DIR}" \
  UV_LOCK_TIMEOUT=1800 \
  UV_PROJECT_ENVIRONMENT=/tmp/nemorl-vllm024-prewarm-vllm \
  uv run --locked --extra vllm --directory "${REPO_DIR}" \
  python -c 'import vllm; print(f"vllm={vllm.__version__}")'
vllm_command="${vllm_command% }"

printf -v mcore_command '%q ' \
  env \
  "UV_CACHE_DIR=${UV_CACHE_DIR}" \
  UV_LOCK_TIMEOUT=1800 \
  UV_PROJECT_ENVIRONMENT=/tmp/nemorl-vllm024-prewarm-mcore \
  uv run --locked --extra mcore --directory "${REPO_DIR}" \
  python -c 'import megatron.core; print("mcore=imported")'
mcore_command="${mcore_command% }"
inner_command="mkdir -p $(printf '%q' "${UV_CACHE_DIR}") && ${vllm_command} && ${mcore_command}"

command_parts=(
  srun
  --nodes=1
  --ntasks=1
  --cpus-per-task=32
  --container-image="${CONTAINER}"
  --container-mounts=/lustre:/lustre
  --container-workdir="${REPO_DIR}"
  --mpi=pmix
  bash -lc "${inner_command}"
)
printf -v command '%q ' "${command_parts[@]}"
command="${command% }"

sbatch_args=(
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --nodes=1
  --ntasks-per-node=1
  --cpus-per-task=32
  --exclusive
  --time="${WALLTIME}"
  --segment=1
  --dependency=
  --job-name="${ACCOUNT}-nemorl.uv-prewarm"
  --output="${EXPERIMENT_ROOT}/slurm-%j.out"
)

case "${MODE}" in
  dry-run)
    printf '[DRY-RUN] sbatch'
    printf ' %q' "${sbatch_args[@]}"
    printf ' --wrap %q\n' "${command}"
    printf '[DRY-RUN] UV_CACHE_DIR=%s UV_LOCK_TIMEOUT=1800\n' "${UV_CACHE_DIR}"
    printf '[DRY-RUN] UV_PROJECT_ENVIRONMENT=/tmp/nemorl-vllm024-prewarm-vllm uv run --locked --extra vllm\n'
    printf '[DRY-RUN] UV_PROJECT_ENVIRONMENT=/tmp/nemorl-vllm024-prewarm-mcore uv run --locked --extra mcore\n'
    ;;
  test-only)
    mkdir -p "${EXPERIMENT_ROOT}"
    sbatch --test-only "${sbatch_args[@]}" --wrap "${command}"
    ;;
  submit)
    mkdir -p "${EXPERIMENT_ROOT}"
    job_id="$(sbatch --parsable "${sbatch_args[@]}" --wrap "${command}")"
    printf '%s\n' "${job_id}"
    ;;
esac
