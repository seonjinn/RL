#!/usr/bin/env bash
set -euo pipefail

readonly EXPERIMENT=qwen3_30ba3b_dflash_dspark_20step_20260822
readonly SOURCE_ROOT=/home/sna/nemorl-pr11-q30-dflash-body-green
readonly CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260818_20260818_6296116.sqsh
readonly DURABLE_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/qwen3_30ba3b_dflash_dspark_20step_20260822
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() { echo "usage: $0 --render|--test-only|--submit" >&2; exit 2; }

render() {
  local root="$1" run artifact script
  run="q30-container-python-diag-$(python3 -c 'import uuid; print(uuid.uuid4().hex)')"
  artifact="${root}/diagnostics/${run}"
  script="${artifact}/job.sbatch"
  mkdir -p "${artifact}"
  cp "${SCRIPT_DIR}/diagnose_container_python.py" "${artifact}/diagnose_container_python.py"
  cp "${SCRIPT_DIR}/verify_df9_configs.py" "${artifact}/verify_df9_configs.py"
  cp "${SCRIPT_DIR}/check_checkpoint_state_dict.py" "${artifact}/check_checkpoint_state_dict.py"
  cp "${SCRIPT_DIR}/configs/dflash.yaml" "${artifact}/dflash.yaml"
  cp "${SCRIPT_DIR}/configs/dspark.yaml" "${artifact}/dspark.yaml"
  cat >"${artifact}/driver.sh" <<DRIVER
#!/usr/bin/env bash
set -euo pipefail
python3 -c 'import ray; print("RAY_IMPORT_PASS", ray.__version__)'
ray status
NRL_FORCE_REBUILD_VENVS=true uv run --directory "${SOURCE_ROOT}" python "${artifact}/verify_df9_configs.py" --source-root "${SOURCE_ROOT}" --config "${artifact}/dflash.yaml"
python3 "${artifact}/check_checkpoint_state_dict.py" --variant dflash --checkpoint /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/sd1/sd1-direct-q30-base-opb-dflash-b8-16n/exported-checkpoint-25391
python3 "${artifact}/check_checkpoint_state_dict.py" --variant dspark --checkpoint /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/sd1/sd1-direct-q30-base-opb-dspark-b8-16n/exported-checkpoint-25391
echo CAPTURE_CONTRACT_PASS
DRIVER
  chmod 700 "${artifact}/driver.sh"
  cat >"${script}" <<SBATCH
#!/usr/bin/env bash
#SBATCH --job-name=sna-q30-pydiag
#SBATCH --account=nemotron_n4_post
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --output=${artifact}/slurm-%j.out
#SBATCH --error=${artifact}/slurm-%j.err
set -euo pipefail
export CONTAINER="${CONTAINER}"
export MOUNTS="/lustre:/lustre,/home:/home"
export GPUS_PER_NODE=4
export BASE_LOG_DIR="${artifact}"
export NRL_FORCE_REBUILD_VENVS=true
export SETUP_COMMAND='python3 "${artifact}/diagnose_container_python.py" > "${artifact}/container-python.json"'
export COMMAND='bash "${artifact}/driver.sh"'
exec bash "${SOURCE_ROOT}/ray.sub"
SBATCH
  chmod 700 "${script}"
  printf '%s\n' "${script}"
}

case "${1:-}" in
  --render) render "${Q30_20STEP_DIAGNOSTIC_RENDER_ROOT:?Q30_20STEP_DIAGNOSTIC_RENDER_ROOT is required}" ;;
  --test-only) sbatch --test-only "$(render "${DURABLE_ROOT}")" 2>&1 ;;
  --submit) sbatch "$(render "${DURABLE_ROOT}")" ;;
  *) usage ;;
esac
