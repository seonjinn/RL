#!/usr/bin/env bash
set -euo pipefail

readonly SOURCE_SHA=443e7243ae2a235b6dcd8f4918fea86e693630a9
readonly CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260818_20260818_6296116.sqsh
readonly EXPERIMENT=qwen3_30ba3b_dflash_dspark_20step_20260822
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SOURCE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly DURABLE_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/qwen3_30ba3b_dflash_dspark_20step_20260822

usage() { echo "usage: $0 --emit-manifest|--write-sbatch|--submit dflash|dspark" >&2; exit 2; }
[[ $# -eq 2 ]] || usage
mode=$1
variant=$2
case "${variant}" in dflash|dspark) ;; *) usage ;; esac
if [[ "${variant}" == dflash ]]; then
  checkpoint=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/sd1/sd1-direct-q30-base-opb-dflash-b8-16n/exported-checkpoint-25391
else
  checkpoint=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/sd1/sd1-direct-q30-base-opb-dspark-b8-16n/exported-checkpoint-25391
fi

wandb_run_id="q30-20step-${variant}-$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}"
artifact_dir="${DURABLE_ROOT}/artifacts/${wandb_run_id}"
sbatch_path="${artifact_dir}/job.sbatch"

emit_manifest() {
  printf '{"variant":"%s","source_sha":"%s","container":"%s","slurm":{"partition":"batch","qos":"normal","time":"04:00:00","nodes":4,"gpus_per_node":4},"gates":["setup","state_dict","cudagraph","step1","step2"],"max_steps":20,"wandb_run_id":"%s"}\n' "${variant}" "${SOURCE_SHA}" "${CONTAINER}" "${wandb_run_id}"
}

write_sbatch() {
  mkdir -p "${artifact_dir}"
  cp "${SCRIPT_DIR}/configs/${variant}.yaml" "${artifact_dir}/resolved-input-${variant}.yaml"
  cat >"${sbatch_path}" <<SBATCH
#!/usr/bin/env bash
#SBATCH --job-name=sna-q30-20-${variant}
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --output=${artifact_dir}/slurm-%j.out
#SBATCH --error=${artifact_dir}/slurm-%j.err
set -euo pipefail
cd "${SOURCE_ROOT}"
test "\$(git rev-parse HEAD)" = "${SOURCE_SHA}"
git diff --quiet && git diff --cached --quiet
git submodule foreach --recursive 'git diff --quiet && git diff --cached --quiet' >/dev/null
test -r "${CONTAINER}"
echo SETUP_GATE_PASS | tee "${artifact_dir}/gates.log"
python3 "${SCRIPT_DIR}/check_checkpoint_state_dict.py" --variant "${variant}" --checkpoint "${checkpoint}" | tee -a "${artifact_dir}/gates.log"
echo CUDAGRAPH_GATE_REQUIRED | tee -a "${artifact_dir}/gates.log"
export WANDB_RUN_ID="${wandb_run_id}"
srun --container-image="${CONTAINER}" --container-mounts="/lustre:/lustre,/home:/home" bash -lc '
  cd "${SOURCE_ROOT}"
  uv run examples/run_grpo.py --config "${SCRIPT_DIR}/configs/${variant}.yaml" \\
    logger.log_dir="${artifact_dir}/logs" \\
    logger.wandb_enabled=True logger.wandb.project=nemo-rl \\
    logger.wandb.name="${wandb_run_id}" 2>&1 | tee "${artifact_dir}/train.log"
'
grep -qE 'CUDA graph|cudagraph' "${artifact_dir}/train.log"
echo CUDAGRAPH_GATE_PASS | tee -a "${artifact_dir}/gates.log"
grep -qE 'step.?1|global_step.?1' "${artifact_dir}/train.log"
echo STEP1_GATE_PASS | tee -a "${artifact_dir}/gates.log"
grep -qE 'step.?2|global_step.?2' "${artifact_dir}/train.log"
echo STEP2_GATE_PASS | tee -a "${artifact_dir}/gates.log"
SBATCH
  chmod 700 "${sbatch_path}"
  printf '%s\n' "${sbatch_path}"
}

case "${mode}" in
  --emit-manifest) emit_manifest ;;
  --write-sbatch) write_sbatch ;;
  --submit) sbatch "$(write_sbatch)" ;;
  *) usage ;;
esac
