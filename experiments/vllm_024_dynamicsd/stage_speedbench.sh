#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

die() {
  echo "$1" >&2
  exit "${2:-2}"
}

require_safe_identifier() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[A-Za-z0-9._:-]+$ ]]; then
    die "invalid scheduler identifier ${name}=${value}"
  fi
}

require_safe_time_limit() {
  local value="$1"
  if [[ ! "${value}" =~ ^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]]; then
    die "invalid scheduler identifier TIME_LIMIT=${value}"
  fi
}

require_safe_dependency() {
  local value="$1"
  if [[ -z "${value}" ]]; then
    return
  fi
  if [[ ! "${value}" =~ ^[A-Za-z0-9._,:+-]+$ ]]; then
    die "invalid scheduler identifier DEPENDENCY=${value}"
  fi
}

render_assignment() {
  local name="$1"
  local value="$2"
  printf '%s=%q\n' "${name}" "${value}"
}

CLUSTER="${CLUSTER:-auto}"
if [[ "${CLUSTER}" == "auto" ]]; then
  case "$(hostname)" in
    *lyris*) CLUSTER="lyris" ;;
    *ptyche*) CLUSTER="ptyche" ;;
    *)
      die "Set CLUSTER=lyris or CLUSTER=ptyche"
      ;;
  esac
fi

ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
case "${CLUSTER}" in
  lyris) PARTITION="${PARTITION:-gb200}" ;;
  ptyche) PARTITION="${PARTITION:-batch}" ;;
  *)
    die "Unsupported CLUSTER=${CLUSTER}"
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

require_safe_identifier "ACCOUNT" "${ACCOUNT}"
require_safe_identifier "PARTITION" "${PARTITION}"
require_safe_identifier "RUN_ID" "${RUN_ID}"
require_safe_time_limit "${TIME_LIMIT}"
require_safe_dependency "${DEPENDENCY}"

RUN_ROOT="${DATASET_ROOT}/${RUN_ID}"
PREPARED_ROOT="${RUN_ROOT}/prepared"
SPEED_PREPARED_ROOT="${PREPARED_ROOT}/speed"
SOURCE_ROOT="${RUN_ROOT}/sources"
MANIFEST="${RUN_ROOT}/prepared_manifest.json"
CHECKSUMS="${RUN_ROOT}/resolved_parquet.sha256"
JOBS_TSV="${RUN_ROOT}/jobs.tsv"

RENDER_ROOT=""
cleanup() {
  if [[ -n "${RENDER_ROOT}" && -d "${RENDER_ROOT}" ]]; then
    rm -rf "${RENDER_ROOT}"
  fi
}
trap cleanup EXIT

render_sbatch() {
  local container_mounts="/lustre:/lustre,${SCRIPT_DIR}:/workspace/experiment"
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
EOF
  render_assignment "CONTAINER_IMAGE" "${CONTAINER_IMAGE}"
  render_assignment "CONTAINER_MOUNTS" "${container_mounts}"
  render_assignment "HF_HOME" "${HF_HOME}"
  render_assignment "PREPARED_ROOT" "${PREPARED_ROOT}"
  render_assignment "SPEED_PREPARED_ROOT" "${SPEED_PREPARED_ROOT}"
  render_assignment "SOURCE_ROOT" "${SOURCE_ROOT}"
  render_assignment "RUN_ROOT" "${RUN_ROOT}"
  render_assignment "MANIFEST" "${MANIFEST}"
  render_assignment "CHECKSUMS" "${CHECKSUMS}"
  render_assignment "SCRIPT_DIR" "${SCRIPT_DIR}"
  render_assignment "MODELOPT_REPO_URL" "${MODELOPT_REPO_URL}"
  render_assignment "MODELOPT_REVISION" "${MODELOPT_REVISION}"
  render_assignment "MODELOPT_PREPARE_DATA_SCRIPT" "${MODELOPT_PREPARE_DATA_SCRIPT}"
  render_assignment "MODELOPT_REPO_DISPLAY" "${MODELOPT_REPO_DISPLAY}"
  render_assignment "SPEED_DATASET_ID" "${SPEED_DATASET_ID}"
  render_assignment "SPEED_DATASET_REVISION" "${SPEED_DATASET_REVISION}"
  cat <<'EOF'

test -s "$CONTAINER_IMAGE"
mkdir -p "$PREPARED_ROOT" "$SOURCE_ROOT" "$RUN_ROOT"
echo "dataset_id=$SPEED_DATASET_ID"
echo "dataset_revision=$SPEED_DATASET_REVISION"
echo "model_optimizer=$MODELOPT_REPO_DISPLAY"
echo "model_optimizer_revision=$MODELOPT_REVISION"
echo "prepare_data_script=$MODELOPT_PREPARE_DATA_SCRIPT"
echo "dataset_license=License.pdf"
echo "modelopt_license=LICENSE"

JOB_TMPDIR="$(mktemp -d "${TMPDIR:-/tmp}/speedbench-stage-job.XXXXXX")"
trap 'rm -rf "$JOB_TMPDIR"' EXIT
PYDEPS_DIR="$JOB_TMPDIR/pydeps"
MODELOPT_WORK_ROOT="$JOB_TMPDIR/modelopt"

read -r -d '' PAYLOAD <<'PAYLOAD' || true
set -euo pipefail
HF_HOME="$1"
SOURCE_ROOT="$2"
MODELOPT_REPO_URL="$3"
MODELOPT_REVISION="$4"
MODELOPT_PREPARE_DATA_SCRIPT="$5"
SPEED_DATASET_ID="$6"
SPEED_DATASET_REVISION="$7"
PREPARED_ROOT="$8"
SPEED_PREPARED_ROOT="$9"
JOB_TMPDIR="${10}"
MANIFEST="${11}"
CHECKSUMS="${12}"
PYDEPS_DIR="${13}"
MODELOPT_WORK_ROOT="${14}"

export HF_HOME
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$PYDEPS_DIR"
python3 -m pip install --quiet --no-cache-dir \
  --target "$PYDEPS_DIR" \
  'datasets>=3.6,<5' \
  'huggingface_hub>=0.32,<1' \
  'pandas>=2,<3' \
  'pyarrow>=18,<21' \
  'tiktoken>=0.8,<1'
export PYTHONPATH="$PYDEPS_DIR"
git clone --filter=blob:none "$MODELOPT_REPO_URL" "$MODELOPT_WORK_ROOT"
git -C "$MODELOPT_WORK_ROOT" checkout "$MODELOPT_REVISION"
export MODELOPT_WORK_ROOT SPEED_DATASET_REVISION
python3 - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["MODELOPT_WORK_ROOT"]) / "examples/specdec_bench/specdec_bench/datasets/speed.py"
needle = 'dataset = load_dataset("nvidia/SPEED-Bench", config_name_or_dataset_path, split="test")'
replacement = (
    'dataset = load_dataset("nvidia/SPEED-Bench", config_name_or_dataset_path, '
    f'split="test", revision="{os.environ["SPEED_DATASET_REVISION"]}")'
)
text = path.read_text(encoding="utf-8")
if needle not in text:
    raise SystemExit(f"Could not pin dataset revision in {path}")
path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
PY
export SOURCE_ROOT SPEED_DATASET_ID SPEED_DATASET_REVISION
python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["SPEED_DATASET_ID"],
    repo_type="dataset",
    revision=os.environ["SPEED_DATASET_REVISION"],
    allow_patterns=["README.md", "License.pdf"],
    local_dir=os.path.join(os.environ["SOURCE_ROOT"], "speedbench"),
)
PY
cp "$MODELOPT_WORK_ROOT/LICENSE" "$SOURCE_ROOT/modelopt-LICENSE"
python3 "$MODELOPT_WORK_ROOT/$MODELOPT_PREPARE_DATA_SCRIPT" \
  --dataset speed \
  --config all \
  --output_dir "$PREPARED_ROOT"
python3 /workspace/experiment/speedbench_dataset.py write-manifest \
  --prepared-root "$SPEED_PREPARED_ROOT" \
  --output "$MANIFEST" \
  --checksums "$CHECKSUMS" \
  --dataset-license-root "$SOURCE_ROOT/speedbench" \
  --modelopt-license "$SOURCE_ROOT/modelopt-LICENSE"
sha256sum "$MANIFEST" | tee "$MANIFEST.sha256"
PAYLOAD

srun_args=(
  --ntasks=1
  "--container-image=$CONTAINER_IMAGE"
  "--container-mounts=$CONTAINER_MOUNTS"
  --no-container-mount-home
  --container-remap-root
  --mpi=pmix
  bash
  -lc
  "$PAYLOAD"
  bash
  "$HF_HOME"
  "$SOURCE_ROOT"
  "$MODELOPT_REPO_URL"
  "$MODELOPT_REVISION"
  "$MODELOPT_PREPARE_DATA_SCRIPT"
  "$SPEED_DATASET_ID"
  "$SPEED_DATASET_REVISION"
  "$PREPARED_ROOT"
  "$SPEED_PREPARED_ROOT"
  "$JOB_TMPDIR"
  "$MANIFEST"
  "$CHECKSUMS"
  "$PYDEPS_DIR"
  "$MODELOPT_WORK_ROOT"
)

srun "${srun_args[@]}" 2>&1 | tee "$RUN_ROOT/stage.log"
EOF
}

if [[ "${DRY_RUN}" != "true" && "${TEST_ONLY}" != "true" && "${REQUIRE_GIT_PULL}" == "true" ]]; then
  git -C "${SCRIPT_DIR}" pull --ff-only
fi

if [[ "${DRY_RUN}" != "true" && "${TEST_ONLY}" != "true" ]]; then
  if [[ ! -s "${CONTAINER_IMAGE}" && -z "${DEPENDENCY}" ]]; then
    die "Missing image and no dependency supplied: ${CONTAINER_IMAGE}" 3
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

if [[ "${TEST_ONLY}" == "true" ]]; then
  RENDER_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/speedbench-stage-render.XXXXXX")"
  SBATCH_FILE="${RENDER_ROOT}/submit.sbatch"
  render_sbatch >"${SBATCH_FILE}"
  sbatch_args=(--test-only)
  if [[ -n "${DEPENDENCY}" ]]; then
    sbatch_args+=("--dependency=${DEPENDENCY}")
  fi
  echo "[TEST-ONLY] manifest=${MANIFEST}"
  sbatch "${sbatch_args[@]}" "${SBATCH_FILE}"
  exit 0
fi

SBATCH_FILE="${RUN_ROOT}/submit.sbatch"
render_sbatch >"${SBATCH_FILE}"
sbatch_args=()
if [[ -n "${DEPENDENCY}" ]]; then
  sbatch_args+=("--dependency=${DEPENDENCY}")
fi
job_id="$(sbatch --parsable "${sbatch_args[@]}" "${SBATCH_FILE}")"
printf '%s\t%s\t%s\n' "${job_id}" "${MANIFEST}" "${CHECKSUMS}" | tee -a "${JOBS_TSV}"
echo "manifest=${MANIFEST}"
echo "checksums=${CHECKSUMS}"
