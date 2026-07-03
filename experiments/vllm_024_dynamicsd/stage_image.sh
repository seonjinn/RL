#!/usr/bin/env bash
set -euo pipefail

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
  ptyche) PARTITION="${PARTITION:-36x2-a01r}" ;;
  *)
    echo "Unsupported CLUSTER=${CLUSTER}" >&2
    exit 2
    ;;
esac

LUSTRE_ROOT="${LUSTRE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}"
CONTAINER_DIR="${CONTAINER_DIR:-${LUSTRE_ROOT}/containers}"
IMAGE_REF="${IMAGE_REF:-vllm/vllm-openai:v0.24.0-aarch64-ubuntu2404}"
IMAGE_URI="${IMAGE_URI:-docker://${IMAGE_REF}}"
IMAGE_NAME="${IMAGE_NAME:-vllm-openai-v0.24.0-aarch64-ubuntu2404.sqsh}"
OUTPUT_IMAGE="${OUTPUT_IMAGE:-${CONTAINER_DIR}/${IMAGE_NAME}}"
RUN_ROOT="${RUN_ROOT:-${LUSTRE_ROOT}/vllm024-dynamicsd/image-stage/$(date +%Y%m%d_%H%M%S)}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
FORCE="${FORCE:-false}"
DRY_RUN="${DRY_RUN:-false}"
TEST_ONLY="${TEST_ONLY:-false}"

SBATCH_FILE="${RUN_ROOT}/stage_image.sbatch"
TMP_IMAGE="${OUTPUT_IMAGE}.partial"

render_sbatch() {
  cat <<EOF
#!/usr/bin/env bash
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --segment=1
#SBATCH --time=${TIME_LIMIT}
#SBATCH --job-name=coreai_dlalgo_llm-vllm024.image
#SBATCH --output=${RUN_ROOT}/slurm-%j.out

set -euo pipefail

mkdir -p '${CONTAINER_DIR}' '${RUN_ROOT}'
if [[ -s '${OUTPUT_IMAGE}' && '${FORCE}' != 'true' ]]; then
  echo 'Image already exists: ${OUTPUT_IMAGE}'
  test -s '${OUTPUT_IMAGE}.metadata'
  grep -Fx 'image_ref=${IMAGE_REF}' '${OUTPUT_IMAGE}.metadata'
  test -s '${OUTPUT_IMAGE}.sha256'
  (cd '${CONTAINER_DIR}' && sha256sum --check '$(basename "${OUTPUT_IMAGE}").sha256')
  exit 0
fi

rm -f '${TMP_IMAGE}'
srun --ntasks=1 \\
  --container-image='${IMAGE_URI}' \\
  --container-save='${TMP_IMAGE}' \\
  --container-mounts=/lustre:/lustre \\
  --no-container-mount-home \\
  --container-remap-root \\
  bash -lc 'set -euo pipefail
arch="\$(uname -m)"
python3 - <<"PY"
import json
import platform
import vllm

result = {
    "architecture": platform.machine(),
    "vllm_version": vllm.__version__,
}
print(json.dumps(result, indent=2))
if platform.machine() != "aarch64":
    raise SystemExit(f"expected aarch64 image, got {platform.machine()}")
if vllm.__version__ != "0.24.0":
    raise SystemExit(f"expected vLLM 0.24.0, got {vllm.__version__}")
PY
command -v nsys || true
' | tee '${RUN_ROOT}/image_validation.log'

test -s '${TMP_IMAGE}'
mv -f '${TMP_IMAGE}' '${OUTPUT_IMAGE}'
sha256sum '${OUTPUT_IMAGE}' | tee '${OUTPUT_IMAGE}.sha256'
cat > '${OUTPUT_IMAGE}.metadata' <<META
image_ref=${IMAGE_REF}
image_uri=${IMAGE_URI}
staged_at=\$(date --iso-8601=seconds)
cluster=${CLUSTER}
slurm_job_id=\${SLURM_JOB_ID}
META
ls -lh '${OUTPUT_IMAGE}' '${OUTPUT_IMAGE}.sha256' '${OUTPUT_IMAGE}.metadata'
EOF
}

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "[DRY-RUN] cluster=${CLUSTER} output_image=${OUTPUT_IMAGE}"
  render_sbatch
  exit 0
fi

mkdir -p "${RUN_ROOT}"
render_sbatch >"${SBATCH_FILE}"
if [[ "${TEST_ONLY}" == "true" ]]; then
  sbatch --test-only "${SBATCH_FILE}"
  echo "test_only=true"
  echo "sbatch_file=${SBATCH_FILE}"
  exit 0
fi

job_id="$(sbatch --parsable "${SBATCH_FILE}")"
echo "job_id=${job_id}"
echo "run_root=${RUN_ROOT}"
echo "sbatch_file=${SBATCH_FILE}"
echo "output_image=${OUTPUT_IMAGE}"
