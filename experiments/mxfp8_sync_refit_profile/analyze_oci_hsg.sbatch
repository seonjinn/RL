#!/bin/bash
#SBATCH --job-name=mxfp8-refit-stats
#SBATCH --partition=batch
#SBATCH --account=nemotron_n3_post
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=00:30:00

set -euo pipefail

ROOT=${ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna}
PROFILE_RUN=${PROFILE_RUN:?Set PROFILE_RUN to the completed profile run directory}
CONTAINER=${CONTAINER:-${ROOT}/containers/nemo_rl_nightly_20260818_20260818_6296116.sqsh}
OUTPUT_DIR=${OUTPUT_DIR:-${PROFILE_RUN}/nvtx_stats}
LOCAL_SCRATCH=${LOCAL_SCRATCH:-/raid/scratch/${USER}/mxfp8-refit-stats-${SLURM_JOB_ID}}

mkdir -p "${OUTPUT_DIR}" "${LOCAL_SCRATCH}"

srun \
  --container-image="${CONTAINER}" \
  --container-mounts="/lustre:/lustre,/raid/scratch:/raid/scratch" \
  bash -s -- "${PROFILE_RUN}" "${OUTPUT_DIR}" "${LOCAL_SCRATCH}" <<'EOF'
set -euo pipefail

profile_run=$1
output_dir=$2
local_scratch=$3

cleanup() {
  rm -rf "${local_scratch}"
}
trap cleanup EXIT

mapfile -t reports < <(
  find "${profile_run}" -maxdepth 9 -type f -name '*.nsys-rep' -print | sort
)
if (( ${#reports[@]} == 0 )); then
  echo "No Nsight Systems reports found under ${profile_run}" >&2
  exit 1
fi

for report in "${reports[@]}"; do
  report_name=$(basename "${report}" .nsys-rep)
  node_name=$(basename "$(dirname "$(dirname "$(dirname "$(dirname "${report}")")")")")
  if [[ ${node_name} == ray ]]; then
    node_name=head
  fi
  stem="${node_name}__${report_name}"
  local_report="${local_scratch}/${stem}.nsys-rep"

  cp "${report}" "${local_report}"
  nsys stats \
    --force-export=true \
    --report nvtx_sum \
    --format csv \
    --output "${output_dir}/${stem}" \
    "${local_report}"
  rm -f "${local_report}" "${local_scratch}/${stem}.sqlite"
done

printf '%s\n' "${reports[@]}" > "${output_dir}/reports.txt"
EOF

echo "NVTX statistics: ${OUTPUT_DIR}"
