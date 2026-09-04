#!/bin/bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
SCRIPT_ROOT=$(git -C "${script_dir}" rev-parse --show-toplevel)
readonly SCRIPT_ROOT
readonly SCRIPTS_DIRECTORY=${SCRIPT_ROOT}/experiments/pr3652_validation_container/scripts
readonly DOWNLOAD_BATCH=${SCRIPTS_DIRECTORY}/oci_hsg_download_validated_nightly.sbatch
readonly SMOKE_BATCH=${SCRIPTS_DIRECTORY}/oci_hsg_smoke_validated_nightly.sbatch
readonly SMOKE_BODY=${SCRIPTS_DIRECTORY}/oci_hsg_smoke_validated_nightly.sh

for batch_script in "${DOWNLOAD_BATCH}" "${SMOKE_BATCH}"; do
  if grep -Fq 'BASH_SOURCE' "${batch_script}"; then
    echo "Spool-dependent helper path in ${batch_script}" >&2
    exit 1
  fi
  grep -Fq 'SCRIPT_ROOT:?Set SCRIPT_ROOT' "${batch_script}"
  grep -Fq 'EXPECTED_TOOLING_SHA:?Set EXPECTED_TOOLING_SHA' "${batch_script}"
  grep -Fq 'validate_tooling_root' "${batch_script}"
done

grep -Fq '#SBATCH --output=/lustre/' "${SMOKE_BATCH}"
grep -Fq '#SBATCH --error=/lustre/' "${SMOKE_BATCH}"
grep -Fq 'readonly CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260904_c6edc455e0fac52d.sqsh' "${SMOKE_BATCH}"
if grep -Fq "CONTAINER=\${CONTAINER:-" "${SMOKE_BATCH}"; then
  echo 'Container override is not permitted' >&2
  exit 1
fi
grep -Fq 'readonly MAIN_PYTHON=/opt/nemo_rl_venv/bin/python' "${SMOKE_BODY}"
if grep -Fq "MAIN_PYTHON=\${MAIN_PYTHON:-" "${SMOKE_BODY}"; then
  echo 'Main Python override is not permitted' >&2
  exit 1
fi
grep -Fq "PYTHONPYCACHEPREFIX=\${SCRATCH_DIRECTORY}/pycache" "${SMOKE_BATCH}"
grep -Fq "XDG_CACHE_HOME=\${SCRATCH_DIRECTORY}/xdg-cache" "${SMOKE_BATCH}"
grep -Fq "UV_CACHE_DIR=\${SCRATCH_DIRECTORY}/uv-cache" "${SMOKE_BATCH}"
grep -Fq "TORCHINDUCTOR_CACHE_DIR=\${SCRATCH_DIRECTORY}/torchinductor-cache" "${SMOKE_BATCH}"
grep -Fq "TRITON_CACHE_DIR=\${SCRATCH_DIRECTORY}/triton-cache" "${SMOKE_BATCH}"
grep -Fq 'rev-parse HEAD' "${SMOKE_BODY}"
grep -Fq 'status --porcelain' "${SMOKE_BODY}"

download_root_validation_line=$(grep -n '^validate_tooling_root$' "${DOWNLOAD_BATCH}" | cut -d: -f1)
download_rclone_line=$(grep -n 'rclone copyto' "${DOWNLOAD_BATCH}" | cut -d: -f1)
test "${download_root_validation_line}" -lt "${download_rclone_line}"
smoke_validator_line=$(grep -n "^\"\${VALIDATOR}\" \"\${CONTAINER}\"" "${SMOKE_BATCH}" | cut -d: -f1)
smoke_srun_line=$(grep -n '/cm/local/apps/slurm/current/bin/srun' "${SMOKE_BATCH}" | cut -d: -f1)
test "${smoke_validator_line}" -lt "${smoke_srun_line}"

for wrapper in "${SCRIPTS_DIRECTORY}"/submit_oci_hsg_*_validated_nightly.sh; do
  grep -Fq -- "--chdir=\"\${SCRIPT_ROOT}\"" "${wrapper}"
  grep -Fq -- "SCRIPT_ROOT=\${SCRIPT_ROOT},EXPECTED_TOOLING_SHA=\${EXPECTED_TOOLING_SHA}" "${wrapper}"
  grep -Fq -- '--test-only' "${wrapper}"
  if grep -Fq 'eval ' "${wrapper}"; then
    echo "Unsafe eval in ${wrapper}" >&2
    exit 1
  fi
done

printf 'validation tooling static checks passed\n'
