#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-test-only}"
REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel)}"
LYRIS_ROOT="${LYRIS_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}"
HF_HOME="${HF_HOME:-${LYRIS_ROOT}/hf_home}"
CONTAINER="${CONTAINER:-${LYRIS_ROOT}/containers/nemo_rl_nightly_20260707.sqsh}"
REPO_ID="${REPO_ID:-inference-optimization/Qwen3-30B-A3B-speculator.dflash}"
CACHE_DIR="${CACHE_DIR:-${HF_HOME}/hub}"
RUN_TAG="${RUN_TAG:-nemorl-v024-q30-dflash-prewarm-20260709}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${LYRIS_ROOT}/experiments/${RUN_TAG}}"
SUMMARY_JSON="${SUMMARY_JSON:-${EXPERIMENT_ROOT}/snapshot_summary.json}"
ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
PARTITION="${PARTITION:-gb200}"
WALLTIME="${WALLTIME:-02:00:00}"

case "${MODE}" in
  dry-run|test-only|submit)
    ;;
  *)
    echo "ERROR: mode must be dry-run, test-only, or submit" >&2
    exit 2
    ;;
esac

if [[ "${MODE}" != "dry-run" && ! -f "${CONTAINER}" ]]; then
  echo "ERROR: container not found: ${CONTAINER}" >&2
  exit 2
fi
if [[ "${MODE}" == "submit" ]]; then
  if ! git -C "${REPO_DIR}" diff --quiet --ignore-submodules=dirty \
    || ! git -C "${REPO_DIR}" diff --cached --quiet --ignore-submodules=dirty; then
    echo "ERROR: submit requires a clean tracked checkout" >&2
    exit 2
  fi
  if ! git -C "${REPO_DIR}" branch -r --contains HEAD | grep -q .; then
    echo "ERROR: HEAD is not present on a known remote branch" >&2
    exit 2
  fi
fi

downloader="${REPO_DIR}/experiments/vllm_024_upgrade/download_hf_snapshot.py"
if [[ "${MODE}" != "dry-run" && ! -f "${downloader}" ]]; then
  echo "ERROR: downloader not found: ${downloader}" >&2
  exit 2
fi

printf -v inner_command '%q ' \
  env "HF_HOME=${HF_HOME}" \
  /opt/nemo_rl_venv/bin/python "${downloader}" \
  --repo-id "${REPO_ID}" \
  --cache-dir "${CACHE_DIR}" \
  --summary-json "${SUMMARY_JSON}"
inner_command="${inner_command% }"

command_parts=(
  srun
  --nodes=1
  --ntasks=1
  --cpus-per-task=16
  --container-image="${CONTAINER}"
  --container-mounts=/lustre:/lustre
  --container-workdir="${REPO_DIR}"
  --mpi=pmix
  bash -lc "${inner_command}"
)
printf -v command '%q ' "${command_parts[@]}"
command="${command% }"

sbatch_args=(
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --nodes=1
  --ntasks-per-node=1
  --cpus-per-task=16
  --exclusive
  --time="${WALLTIME}"
  --segment=1
  --dependency=
  --job-name="${ACCOUNT}-nemorl.dflash-prewarm"
  --output="${EXPERIMENT_ROOT}/slurm-%j.out"
)

case "${MODE}" in
  dry-run)
    printf '[DRY-RUN] sbatch'
    printf ' %q' "${sbatch_args[@]}"
    printf ' --wrap %q\n' "${command}"
    printf '[DRY-RUN] repo_id=%s cache_dir=%s summary=%s\n' \
      "${REPO_ID}" "${CACHE_DIR}" "${SUMMARY_JSON}"
    ;;
  test-only)
    mkdir -p "${EXPERIMENT_ROOT}"
    sbatch --test-only "${sbatch_args[@]}" --wrap "${command}"
    ;;
  submit)
    mkdir -p "${EXPERIMENT_ROOT}"
    job_id="$(sbatch --parsable "${sbatch_args[@]}" --wrap "${command}")"
    printf '%s\n' "${job_id}"
    ;;
esac
