#!/usr/bin/env bash
set -euo pipefail

REMOTE_REPO="${REMOTE_REPO:?set REMOTE_REPO}"
CONTAINER="${CONTAINER:?set CONTAINER}"
ACCOUNT="${ACCOUNT:?set ACCOUNT}"
HF_HOME="${HF_HOME:?set HF_HOME}"
MODEL_IDS="${MODEL_IDS:?set MODEL_IDS as comma-separated repository IDs}"
PARTITION="${PARTITION:-batch}"
USE_GRES="${USE_GRES:-auto}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
SEGMENT="${SEGMENT:-1}"
WALLTIME="${WALLTIME:-01:00:00}"
JOB_NAME="${JOB_NAME:-${ACCOUNT}-nemorl-specdec.stage-hf}"
LOG_DIR="${LOG_DIR:-${REMOTE_REPO}/experiments/cluster_smoke/logs}"
SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:---comment=metrics}"
DRY_RUN="${DRY_RUN:-false}"

if [[ ! -f "${CONTAINER}" ]]; then
  echo "ERROR: container not found: ${CONTAINER}" >&2
  exit 2
fi

resource_args=(
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --nodes=1
  --ntasks-per-node=1
  --cpus-per-task=16
  --mem=0
  --time="${WALLTIME}"
  --segment="${SEGMENT}"
  --job-name="${JOB_NAME}"
  --output="${LOG_DIR}/slurm-%j.out"
)

case "${USE_GRES}" in
  true|True|TRUE|1|yes|Yes|YES)
    resource_args+=(--gres="gpu:${GPUS_PER_NODE}")
    ;;
  false|False|FALSE|0|no|No|NO)
    ;;
  auto)
    case "${PARTITION}" in
      gb200*|gb300*) ;;
      *) resource_args+=(--gres="gpu:${GPUS_PER_NODE}") ;;
    esac
    ;;
  *)
    echo "ERROR: USE_GRES must be true, false, or auto" >&2
    exit 2
    ;;
esac

read -r -a extra_args <<< "${SBATCH_EXTRA_ARGS}"
stage_command="env HF_HOME=${HF_HOME} MODEL_IDS=${MODEL_IDS} HF_HUB_ENABLE_HF_TRANSFER=1 /opt/nemo_rl_venv/bin/python ${REMOTE_REPO}/scripts/stage_hf_snapshots.py"
run_command="srun --nodes=1 --ntasks=1 --no-container-mount-home --container-image=${CONTAINER} --container-mounts=/lustre:/lustre --container-workdir=${REMOTE_REPO} bash -lc \"${stage_command}\""

if [[ "${DRY_RUN}" == "true" || "${DRY_RUN}" == "True" ]]; then
  printf '[DRY-RUN] sbatch'
  printf ' %q' "${resource_args[@]}" "${extra_args[@]}"
  printf ' --wrap %q\n' "${run_command}"
  exit 0
fi

mkdir -p "${LOG_DIR}"
sbatch --parsable "${resource_args[@]}" "${extra_args[@]}" --wrap "${run_command}"
