#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER="${CLUSTER:-auto}"
if [[ "${CLUSTER}" == "auto" ]]; then
  case "$(hostname)" in
    *lyris*) CLUSTER="lyris" ;;
    *ptyche*) CLUSTER="ptyche" ;;
    *)
      echo "Set CLUSTER=lyris or CLUSTER=ptyche" >&2
      exit 2
      ;;
  esac
fi

ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
case "${CLUSTER}" in
  lyris) PARTITION="${PARTITION:-gb200}" ;;
  ptyche) PARTITION="${PARTITION:-batch}" ;;
  *)
    echo "Unsupported CLUSTER=${CLUSTER}" >&2
    exit 2
    ;;
esac

LUSTRE_ROOT="${LUSTRE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${LUSTRE_ROOT}/containers/vllm-openai-v0.24.0-aarch64-ubuntu2404.sqsh}"
HF_HOME="${HF_HOME:-${LUSTRE_ROOT}/hf_home}"
DATASET_ROOT="${DATASET_ROOT:-${LUSTRE_ROOT}/vllm024-dynamicsd/speedbench}"
RUN_ID="${RUN_ID:-speedbench-487aa718-43fee0cd}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
DEPENDENCY="${DEPENDENCY:-}"
DRY_RUN="${DRY_RUN:-false}"
TEST_ONLY="${TEST_ONLY:-false}"
REQUIRE_GIT_PULL="${REQUIRE_GIT_PULL:-true}"

SPEED_DATASET_ID="nvidia/SPEED-Bench"
SPEED_DATASET_REVISION="487aa718444e816458d1a0a52bfce7a454285cf4"
MODELOPT_REPO_DISPLAY="NVIDIA/Model-Optimizer"
MODELOPT_REPO_URL="https://github.com/NVIDIA/Model-Optimizer.git"
MODELOPT_REVISION="43fee0cd70fa9e5f85782d52a4bd8ad9c8b88446"
MODELOPT_PREPARE_DATA_SCRIPT="examples/specdec_bench/prepare_data.py"

RUN_ROOT="${DATASET_ROOT}/${RUN_ID}"
if [[ "${DRY_RUN}" == "true" || "${TEST_ONLY}" == "true" ]]; then
  RUN_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/speedbench-stage.XXXXXX")/${RUN_ID}"
  trap 'rm -rf "${RUN_ROOT%/*}"' EXIT
fi

PREPARED_ROOT="${RUN_ROOT}/prepared"
SOURCE_ROOT="${RUN_ROOT}/sources"
MANIFEST="${RUN_ROOT}/prepared_manifest.json"
CHECKSUMS="${RUN_ROOT}/resolved_parquet.sha256"
JOBS_TSV="${RUN_ROOT}/jobs.tsv"

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
#SBATCH --time=${TIME_LIMIT}
#SBATCH --job-name=coreai_dlalgo_llm-speedbench-stage
#SBATCH --output=${RUN_ROOT}/slurm-%j.out

set -euo pipefail

test -s '${CONTAINER_IMAGE}'
mkdir -p '${PREPARED_ROOT}' '${SOURCE_ROOT}'
echo 'dataset_id=${SPEED_DATASET_ID}'
echo 'dataset_revision=${SPEED_DATASET_REVISION}'
echo 'model_optimizer=${MODELOPT_REPO_DISPLAY}'
echo 'model_optimizer_revision=${MODELOPT_REVISION}'
echo 'prepare_data_script=${MODELOPT_PREPARE_DATA_SCRIPT}'
echo 'dataset_license=License.pdf'
echo 'modelopt_license=LICENSE'

srun --ntasks=1 \\
  --container-image='${CONTAINER_IMAGE}' \\
  --container-mounts='/lustre:/lustre,${SCRIPT_DIR}:/workspace/experiment' \\
  --no-container-mount-home \\
  --container-remap-root \\
  --mpi=pmix \\
  bash -lc "set -euo pipefail
export HF_HOME='${HF_HOME}'
export HUGGINGFACE_HUB_CACHE='${HF_HOME}/hub'
export HF_DATASETS_CACHE='${HF_HOME}/datasets'
rm -rf /tmp/vllm024_speedbench_pydeps /tmp/vllm024-modelopt
python3 -m pip install --quiet --no-cache-dir \\
  --target /tmp/vllm024_speedbench_pydeps \\
  'datasets>=3.6,<5' \\
  'huggingface_hub>=0.32,<1' \\
  'pandas>=2,<3' \\
  'pyarrow>=18,<21' \\
  'tiktoken>=0.8,<1'
export PYTHONPATH=/tmp/vllm024_speedbench_pydeps
git clone --filter=blob:none '${MODELOPT_REPO_URL}' /tmp/vllm024-modelopt
git -C /tmp/vllm024-modelopt checkout '${MODELOPT_REVISION}'
python3 - <<'PY'
from pathlib import Path

path = Path('/tmp/vllm024-modelopt/examples/specdec_bench/specdec_bench/datasets/speed.py')
needle = 'dataset = load_dataset(\"nvidia/SPEED-Bench\", config_name_or_dataset_path, split=\"test\")'
replacement = (
    'dataset = load_dataset(\"nvidia/SPEED-Bench\", config_name_or_dataset_path, '
    'split=\"test\", revision=\"${SPEED_DATASET_REVISION}\")'
)
text = path.read_text(encoding='utf-8')
if needle not in text:
    raise SystemExit(f'Could not pin dataset revision in {path}')
path.write_text(text.replace(needle, replacement, 1), encoding='utf-8')
PY
python3 - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='${SPEED_DATASET_ID}',
    repo_type='dataset',
    revision='${SPEED_DATASET_REVISION}',
    allow_patterns=['README.md', 'License.pdf'],
    local_dir='${SOURCE_ROOT}/speedbench',
)
PY
cp /tmp/vllm024-modelopt/LICENSE '${SOURCE_ROOT}/modelopt-LICENSE'
python3 /tmp/vllm024-modelopt/${MODELOPT_PREPARE_DATA_SCRIPT} \\
  --dataset speed \\
  --config all \\
  --output_dir '${PREPARED_ROOT}'
find '${PREPARED_ROOT}' -name 'test.parquet' -print | LC_ALL=C sort | \\
  while read -r path; do sha256sum \"\${path}\"; done | tee '${CHECKSUMS}'
python3 /workspace/experiment/speedbench_dataset.py write-manifest \\
  --prepared-root '${PREPARED_ROOT}' \\
  --output '${MANIFEST}'
sha256sum '${MANIFEST}' | tee '${MANIFEST}.sha256'
" 2>&1 | tee '${RUN_ROOT}/stage.log'
EOF
}

if [[ "${DRY_RUN}" != "true" && "${REQUIRE_GIT_PULL}" == "true" ]]; then
  git -C "${SCRIPT_DIR}" pull --ff-only
fi

if [[ "${DRY_RUN}" != "true" ]]; then
  if [[ "${TEST_ONLY}" != "true" && ! -s "${CONTAINER_IMAGE}" && -z "${DEPENDENCY}" ]]; then
    echo "Missing image and no dependency supplied: ${CONTAINER_IMAGE}" >&2
    exit 3
  fi
  mkdir -p "${RUN_ROOT}"
  printf 'job_id\tmanifest\tchecksums\n' >"${JOBS_TSV}"
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "[DRY-RUN] dataset_id=${SPEED_DATASET_ID}"
  echo "[DRY-RUN] dataset_revision=${SPEED_DATASET_REVISION}"
  echo "[DRY-RUN] model_optimizer=${MODELOPT_REPO_DISPLAY}"
  echo "[DRY-RUN] model_optimizer_revision=${MODELOPT_REVISION}"
  echo "manifest=${MANIFEST}"
  render_sbatch
  exit 0
fi

SBATCH_FILE="${RUN_ROOT}/submit.sbatch"
render_sbatch >"${SBATCH_FILE}"
sbatch_args=()
if [[ -n "${DEPENDENCY}" ]]; then
  sbatch_args+=("--dependency=${DEPENDENCY}")
fi
if [[ "${TEST_ONLY}" == "true" ]]; then
  echo "[TEST-ONLY] manifest=${MANIFEST}"
  sbatch --test-only "${sbatch_args[@]}" "${SBATCH_FILE}"
  printf 'test-only\t%s\t%s\n' "${MANIFEST}" "${CHECKSUMS}" >>"${JOBS_TSV}"
  exit 0
fi

job_id="$(sbatch --parsable "${sbatch_args[@]}" "${SBATCH_FILE}")"
printf '%s\t%s\t%s\n' "${job_id}" "${MANIFEST}" "${CHECKSUMS}" | tee -a "${JOBS_TSV}"
echo "manifest=${MANIFEST}"
echo "checksums=${CHECKSUMS}"
