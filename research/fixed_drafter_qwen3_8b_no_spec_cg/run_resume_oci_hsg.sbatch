#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --partition=batch_long
#SBATCH --job-name=qwen8-nospec-scale-resume

set -euo pipefail

: "${REMOTE_REPO:?Set REMOTE_REPO to an exact /home checkout}"
: "${EXPECTED_HEAD:?Set EXPECTED_HEAD}"
: "${FINAL_DIR:?Set FINAL_DIR under /lustre}"
: "${CONTAINER:?Set CONTAINER}"
: "${TARGET_SNAPSHOT:?Set TARGET_SNAPSHOT}"
: "${WANDB_API_KEY:?WANDB_API_KEY is required}"

readonly training_horizon_steps=1000
readonly checkpoint_dir="${FINAL_DIR}/checkpoints"
readonly gate_manifest="${FINAL_DIR}/gate-manifest.json"
readonly scratch_root="/raid/scratch/nrl-qwen3-8b-fixed-4k/${SLURM_JOB_ID}"
readonly train_log="${scratch_root}/logs/train.log"
readonly config="${REMOTE_REPO}/research/fixed_drafter_qwen3_8b_no_spec_cg/config.yaml"
readonly contract="${REMOTE_REPO}/research/fixed_drafter_qwen3_8b_no_spec_cg/contract.py"

[[ "${REMOTE_REPO}" == /home/* ]]
[[ "${FINAL_DIR}" == /lustre/* ]]
test "$(git -C "${REMOTE_REPO}" rev-parse HEAD)" = "${EXPECTED_HEAD}"
git -C "${REMOTE_REPO}" diff-index --quiet --ignore-submodules=all HEAD --
test -z "$(git -C "${REMOTE_REPO}" ls-files --others --exclude-standard)"
git -C "${REMOTE_REPO}" submodule foreach --quiet --recursive \
  'git diff-index --quiet --ignore-submodules=all HEAD --'

previous_step="$(python3 "${contract}" --checkpoint-dir "${checkpoint_dir}" --print-latest-step)"
python3 "${contract}" --checkpoint-dir "${checkpoint_dir}" --expected-step "${previous_step}" \
  --gate-manifest "${gate_manifest}" --git-sha "${EXPECTED_HEAD}" --validate-gate-manifest
wandb_run_id="$(python3 "${contract}" --checkpoint-dir "${checkpoint_dir}" \
  --gate-manifest "${gate_manifest}" --git-sha "${EXPECTED_HEAD}" --print-manifest-wandb-run-id)"
if [[ "${previous_step}" == "${training_horizon_steps}" ]]; then exit 0; fi

mkdir -p "${scratch_root}/logs" "${scratch_root}/tmp" "${scratch_root}/ray" "${FINAL_DIR}/logs"
archive_log() {
  status=$?
  trap - EXIT
  set +e
  test -f "${train_log}" && cp "${train_log}" "${FINAL_DIR}/logs/job-${SLURM_JOB_ID}-from-${previous_step}.log"
  exit "${status}"
}
trap archive_log EXIT

srun --ntasks=1 --container-image="${CONTAINER}" \
  --container-mounts="/home:/home,/raid:/raid,/lustre:/lustre" --no-container-mount-home \
  bash -lc "set -euo pipefail
export PYTHONPATH='${REMOTE_REPO}'
export HF_HOME='/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home'
export TMPDIR='${scratch_root}/tmp'
export RAY_TMPDIR='${scratch_root}/ray'
export TOKENIZERS_PARALLELISM=false
cd '${REMOTE_REPO}'
/opt/nemo_rl_venv/bin/python '${contract}' --checkpoint-dir '${checkpoint_dir}' \
  --expected-step '${previous_step}' --verify-nemo-resume-paths
UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv uv run --frozen --no-sync \
  examples/run_grpo.py --config '${config}' \
  grpo.max_num_steps='${training_horizon_steps}' grpo.val_at_end=true \
  checkpointing.save_period=100 checkpointing.checkpoint_must_save_by=00:03:30:00 \
  checkpointing.checkpoint_dir='${checkpoint_dir}' logger.log_dir='${scratch_root}/logs' \
  logger.wandb.name='qwen3-8b-no-specdec-pps64-gps8-gbs512-step1000-seed42' \
  'logger.wandb.tags=[baseline,no-specdec,qwen3-8b,cudagraph,target-only-grpo,seed42,pps64-gps8-gbs512,step1000]' \
  logger.wandb.config.stage_steps='${training_horizon_steps}' \
  +logger.wandb.config.training_horizon_steps='${training_horizon_steps}' \
  +logger.wandb.id='${wandb_run_id}' +logger.wandb.resume=must \
  2>&1 | tee '${train_log}'
"
grep -Fq "Capturing CUDA graphs (PIECEWISE)" "${train_log}"
grep -Fq "Graph capturing finished" "${train_log}"
current_step="$(python3 "${contract}" --checkpoint-dir "${checkpoint_dir}" --print-latest-step)"
python3 "${contract}" --previous-step "${previous_step}" --current-step "${current_step}" \
  --checkpoint-dir "${checkpoint_dir}" --expected-step "${current_step}"
