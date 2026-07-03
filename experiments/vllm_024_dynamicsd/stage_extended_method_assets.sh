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
    echo "Unsupported cluster: ${CLUSTER}" >&2
    exit 2
    ;;
esac

LUSTRE_ROOT="${LUSTRE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${LUSTRE_ROOT}/containers/vllm-openai-v0.24.0-aarch64-ubuntu2404.sqsh}"
HF_HOME="${HF_HOME:-${LUSTRE_ROOT}/hf_home}"
ASSET_ROOT="${ASSET_ROOT:-${LUSTRE_ROOT}/vllm024-dynamicsd/assets}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_extended_assets}"
RUN_DIR="${ASSET_ROOT}/staging/${RUN_ID}"
SBATCH_FILE="${RUN_DIR}/submit.sbatch"
VLLM_COMMIT="ee0da84ab9e04ac7610e28580af62c365e898389"
ANGELSLIM_COMMIT="6a97dab2f17c0a3c031065329f092c4f61108a6f"
PARD_COMMIT="6f279bf3f1680e0b5d71c562ca5b91bdeef4c038"
QWEN8_REVISION="b968826d9c46dd6066d109eabc6255188de91218"
PARD_REVISION="f9f650fbab180c26498817718f0db5cae8f25136"
PARD2_REVISION="67a1516c8f6fc145cda99916799a0cbb3a4af135"
DFLASH_REVISION="9b41424b7109f9c5413454f481b09a82b85333f4"
DFLARE_REVISION="55e2c8d86d76ce1e79fa3b8642c7f80091285a82"
DRY_RUN="${DRY_RUN:-false}"
TEST_ONLY="${TEST_ONLY:-false}"
REQUIRE_GIT_PULL="${REQUIRE_GIT_PULL:-true}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"

render_sbatch() {
  cat <<EOF
#!/usr/bin/env bash
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --segment=1
#SBATCH --time=${TIME_LIMIT}
#SBATCH --job-name=coreai_dlalgo_llm-vllm024.extended-assets
#SBATCH --output=${RUN_DIR}/slurm-%j.out

set -euo pipefail

test -s '${CONTAINER_IMAGE}'
mkdir -p '${ASSET_ROOT}/src' '${ASSET_ROOT}/python' '${HF_HOME}'

export HF_HOME='${HF_HOME}'
export ASSET_ROOT='${ASSET_ROOT}'
export VLLM_COMMIT='${VLLM_COMMIT}'
export ANGELSLIM_COMMIT='${ANGELSLIM_COMMIT}'
export PARD_COMMIT='${PARD_COMMIT}'
export QWEN8_REVISION='${QWEN8_REVISION}'
export PARD_REVISION='${PARD_REVISION}'
export PARD2_REVISION='${PARD2_REVISION}'
export DFLASH_REVISION='${DFLASH_REVISION}'
export DFLARE_REVISION='${DFLARE_REVISION}'

srun --ntasks=1 \\
  --container-image='${CONTAINER_IMAGE}' \\
  --container-mounts='/lustre:/lustre,${SCRIPT_DIR}:/workspace/experiment' \\
  --no-container-mount-home \\
  --container-remap-root \\
  --mpi=pmix \\
  bash /workspace/experiment/stage_extended_method_assets_in_container.sh
EOF
}

if [[ "${DRY_RUN}" != "true" && "${REQUIRE_GIT_PULL}" == "true" ]]; then
  git -C "${SCRIPT_DIR}" pull --ff-only
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  render_sbatch
  exit 0
fi

mkdir -p "${RUN_DIR}"
render_sbatch >"${SBATCH_FILE}"
if [[ "${TEST_ONLY}" == "true" ]]; then
  sbatch --test-only "${SBATCH_FILE}"
  exit 0
fi

job_id="$(sbatch --parsable "${SBATCH_FILE}")"
printf 'job_id=%s\nrun_dir=%s\n' "${job_id}" "${RUN_DIR}"
