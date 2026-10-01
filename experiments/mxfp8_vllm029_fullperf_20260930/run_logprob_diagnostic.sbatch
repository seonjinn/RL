#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail

: "${INPUT_FILES:?Set INPUT_FILES to a colon-separated list of JSONL files}"
: "${OUTPUT_FILE:?Set OUTPUT_FILE to the durable JSON output path}"
: "${SOURCE_ROOT:?Set SOURCE_ROOT to the NeMo-RL checkout under /home}"

scratch_root="/raid/scratch/${USER}/logprob-diagnostic-${SLURM_JOB_ID}"
mkdir -p "${scratch_root}"
trap 'rm -rf "${scratch_root}"' EXIT

IFS=: read -r -a input_files <<< "${INPUT_FILES}"
local_files=()
for input_file in "${input_files[@]}"; do
  run_name="$(basename "$(dirname "$(dirname "$(dirname "${input_file}")")")")"
  local_file="${scratch_root}/${run_name}-$(basename "${input_file}")"
  cp "${input_file}" "${local_file}"
  local_files+=("${local_file}")
done

python3 \
  "${SOURCE_ROOT}/experiments/mxfp8_vllm029_fullperf_20260930/inspect_logprob_nonfinite.py" \
  "${local_files[@]}" \
  --output "${scratch_root}/summary.json"

mkdir -p "$(dirname "${OUTPUT_FILE}")"
cp "${scratch_root}/summary.json" "${OUTPUT_FILE}"
cat "${OUTPUT_FILE}"
