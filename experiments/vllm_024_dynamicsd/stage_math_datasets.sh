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
DATASET_ROOT="${DATASET_ROOT:-${LUSTRE_ROOT}/vllm024-dynamicsd/datasets}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SOURCES="${SOURCES:-dapo_math_17k openmathinstruct2}"
LIMIT="${LIMIT:-1024}"
OFFSET="${OFFSET:-0}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
DEPENDENCY="${DEPENDENCY:-}"
DRY_RUN="${DRY_RUN:-false}"
TEST_ONLY="${TEST_ONLY:-false}"
REQUIRE_GIT_PULL="${REQUIRE_GIT_PULL:-true}"
RUN_ROOT="${DATASET_ROOT}/stage-${RUN_ID}"
MANIFEST="${RUN_ROOT}/jobs.tsv"

source_spec() {
  case "$1" in
    dapo_math_17k)
      echo "BytedTsinghua-SIA/DAPO-Math-17k|65877096c24ffa7abc4e4fa5edb95cf3413a5674"
      ;;
    openmathinstruct2)
      echo "nvidia/OpenMathInstruct-2|469216e3f46f4dacf476b382e192485ea51a143e"
      ;;
    *)
      echo "Unsupported source: $1" >&2
      return 2
      ;;
  esac
}

render_sbatch() {
  local source="$1"
  local dataset="$2"
  local revision="$3"
  local output="$4"
  local run_dir="$5"
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
#SBATCH --job-name=coreai_dlalgo_llm-vllm024.data-${source}
#SBATCH --output=${run_dir}/slurm-%j.out

set -euo pipefail

test -s '${CONTAINER_IMAGE}'
mkdir -p '${DATASET_ROOT}' '${run_dir}'
echo 'source=${source}'
echo 'dataset=${dataset}'
echo 'revision=${revision}'
echo 'streaming=true'

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
rm -rf /tmp/vllm024_dataset_pydeps
python3 -m pip install --quiet --no-cache-dir \
  --target /tmp/vllm024_dataset_pydeps 'datasets>=3.6,<5'
export PYTHONPATH=/tmp/vllm024_dataset_pydeps
python3 /workspace/experiment/materialize_math_prompts.py \\
  --source '${source}' \\
  --output '${output}' \\
  --limit '${LIMIT}' \\
  --offset '${OFFSET}' \\
  --streaming
sha256sum '${output}' | tee '${output}.sha256'
" 2>&1 | tee '${run_dir}/materialize.log'
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
  printf 'job_id\tsource\toutput\trun_dir\n' >"${MANIFEST}"
fi

for source in ${SOURCES}; do
  spec="$(source_spec "${source}")"
  dataset="${spec%%|*}"
  revision="${spec##*|}"
  output="${DATASET_ROOT}/${source}_${revision:0:12}_prompts_${LIMIT}_offset${OFFSET}.jsonl"
  run_dir="${RUN_ROOT}/${source}"
  sbatch_file="${run_dir}/submit.sbatch"
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY-RUN] dataset_source=${source}"
    render_sbatch "${source}" "${dataset}" "${revision}" "${output}" "${run_dir}"
    continue
  fi

  mkdir -p "${run_dir}"
  render_sbatch "${source}" "${dataset}" "${revision}" "${output}" "${run_dir}" >"${sbatch_file}"
  sbatch_args=()
  if [[ -n "${DEPENDENCY}" ]]; then
    sbatch_args+=("--dependency=${DEPENDENCY}")
  fi
  if [[ "${TEST_ONLY}" == "true" ]]; then
    sbatch --test-only "${sbatch_args[@]}" "${sbatch_file}"
    printf 'test-only\t%s\t%s\t%s\n' "${source}" "${output}" "${run_dir}" >>"${MANIFEST}"
    continue
  fi
  job_id="$(sbatch --parsable "${sbatch_args[@]}" "${sbatch_file}")"
  printf '%s\t%s\t%s\t%s\n' "${job_id}" "${source}" "${output}" "${run_dir}" | tee -a "${MANIFEST}"
done

if [[ "${DRY_RUN}" != "true" ]]; then
  echo "manifest=${MANIFEST}"
  echo "dataset_root=${DATASET_ROOT}"
fi
