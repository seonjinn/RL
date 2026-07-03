#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER="${CLUSTER:-auto}"
if [[ "${CLUSTER}" == "auto" ]]; then
  case "$(hostname)" in
    *lyris*) CLUSTER=lyris ;;
    *ptyche*) CLUSTER=ptyche ;;
    *) echo "Set CLUSTER=lyris or CLUSTER=ptyche" >&2; exit 2 ;;
  esac
fi

ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
case "${CLUSTER}" in
  lyris) PARTITION="${PARTITION:-gb200}" ;;
  ptyche) PARTITION="${PARTITION:-batch}" ;;
  *) echo "Unsupported cluster: ${CLUSTER}" >&2; exit 2 ;;
esac

LUSTRE_ROOT="${LUSTRE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${LUSTRE_ROOT}/containers/vllm-openai-v0.24.0-aarch64-ubuntu2404.sqsh}"
RAY_VERSION="${RAY_VERSION:-2.55.1}"
RAY_SITE="${RAY_SITE:-${LUSTRE_ROOT}/vllm024-dynamicsd/python-sites/ray-${RAY_VERSION}-py312}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${LUSTRE_ROOT}/vllm024-dynamicsd/ray-stage/${RUN_ID}"
TEST_ONLY="${TEST_ONLY:-false}"
DRY_RUN="${DRY_RUN:-false}"

render_sbatch() {
  cat <<EOF
#!/usr/bin/env bash
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --segment=1
#SBATCH --time=00:30:00
#SBATCH --job-name=coreai_dlalgo_llm-vllm024.ray-stage
#SBATCH --output=${RUN_DIR}/slurm-%j.out

set -euo pipefail
test -s '${CONTAINER_IMAGE}'
mkdir -p '$(dirname "${RAY_SITE}")' '${RUN_DIR}'
rm -rf '${RAY_SITE}.partial'

srun --ntasks=1 \
  --container-image='${CONTAINER_IMAGE}' \
  --container-mounts='/lustre:/lustre,${SCRIPT_DIR}:/workspace/experiment' \
  --no-container-mount-home \
  --container-remap-root \
  --mpi=pmix \
  bash -lc "set -euo pipefail
python3 -m pip install --quiet --no-cache-dir \
  --target '${RAY_SITE}.partial' 'ray[default]==${RAY_VERSION}'
PYTHONPATH='${RAY_SITE}.partial' python3 -c \
  'import ray, vllm; print({\"ray\": ray.__version__, \"vllm\": vllm.__version__})'
rm -rf '${RAY_SITE}'
mv '${RAY_SITE}.partial' '${RAY_SITE}'
"

printf 'ray_version=%s\nvllm_version=0.24.0\n' '${RAY_VERSION}' > '${RAY_SITE}.metadata'
EOF
}

if [[ "${DRY_RUN}" == "true" ]]; then
  render_sbatch
  exit 0
fi

mkdir -p "${RUN_DIR}"
sbatch_file="${RUN_DIR}/submit.sbatch"
render_sbatch >"${sbatch_file}"
if [[ "${TEST_ONLY}" == "true" ]]; then
  sbatch --test-only "${sbatch_file}"
else
  job_id="$(sbatch --parsable "${sbatch_file}")"
  echo "job_id=${job_id}"
  echo "ray_site=${RAY_SITE}"
fi
