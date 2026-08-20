#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --partition=batch_long
#SBATCH --job-name=qwen8-nospec-scale-gate

set -euo pipefail

: "${REMOTE_REPO:?Set REMOTE_REPO to an exact /home checkout}"
: "${EXPECTED_HEAD:?Set EXPECTED_HEAD}"
: "${FINAL_DIR:?Set FINAL_DIR under /lustre}"
: "${CONTAINER:?Set CONTAINER}"
: "${TARGET_SNAPSHOT:?Set TARGET_SNAPSHOT}"
: "${WANDB_API_KEY:?WANDB_API_KEY is required}"

readonly expected_container_sha256=6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44
readonly training_horizon_steps=1000
readonly scratch_root="/raid/scratch/nrl-qwen3-8b-fixed-4k/${SLURM_JOB_ID}"
readonly checkpoint_dir="${FINAL_DIR}/checkpoints"
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
grep -Fqx "sha256=${expected_container_sha256}" "${CONTAINER}.metadata.txt"
test "$(basename "${TARGET_SNAPSHOT}")" = b968826d9c46dd6066d109eabc6255188de91218

mkdir -p "${scratch_root}/logs" "${scratch_root}/tmp" "${scratch_root}/ray" "${checkpoint_dir}" "${FINAL_DIR}/logs"
archive_log() {
  status=$?
  trap - EXIT
  set +e
  test -f "${train_log}" && cp "${train_log}" "${FINAL_DIR}/logs/job-${SLURM_JOB_ID}-gate.log"
  test -f "${scratch_root}/contract.json" && cp "${scratch_root}/contract.json" "${FINAL_DIR}/contract.json"
  exit "${status}"
}
trap archive_log EXIT

python3 "${contract}" --config "${config}" --validate-config | tee "${scratch_root}/contract.json"
srun --ntasks=1 --container-image="${CONTAINER}" \
  --container-mounts="/home:/home,/raid:/raid,/lustre:/lustre" --no-container-mount-home \
  bash -lc "set -euo pipefail
export PYTHONPATH='${REMOTE_REPO}'
export HF_HOME='/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home'
export TMPDIR='${scratch_root}/tmp'
export RAY_TMPDIR='${scratch_root}/ray'
export TOKENIZERS_PARALLELISM=false
cd '${REMOTE_REPO}'
UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv uv run --frozen --no-sync \
  examples/run_grpo.py --config '${config}' \
  grpo.max_num_steps='${training_horizon_steps}' \
  checkpointing.save_period=100 checkpointing.checkpoint_must_save_by=00:00:00:01 \
  checkpointing.checkpoint_dir='${checkpoint_dir}' logger.log_dir='${scratch_root}/logs' \
  logger.wandb.name='qwen3-8b-no-specdec-pps64-gps8-gbs512-step1000-seed42' \
  'logger.wandb.tags=[baseline,no-specdec,qwen3-8b,cudagraph,target-only-grpo,seed42,pps64-gps8-gbs512,step1000]' \
  logger.wandb.config.stage_steps='${training_horizon_steps}' \
  +logger.wandb.config.training_horizon_steps='${training_horizon_steps}' \
  2>&1 | tee '${train_log}'
"
grep -Fq "Capturing CUDA graphs (PIECEWISE)" "${train_log}"
grep -Fq "Graph capturing finished" "${train_log}"
python3 "${contract}" --checkpoint-dir "${checkpoint_dir}" --expected-step 1 \
  --wandb-log "${train_log}" --gate-manifest "${FINAL_DIR}/gate-manifest.json" \
  --git-sha "${EXPECTED_HEAD}" --create-gate-manifest
