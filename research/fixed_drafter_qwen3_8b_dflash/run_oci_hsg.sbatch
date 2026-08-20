#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --partition=batch_long
#SBATCH --job-name=qwen8-dflash-scale-gate

set -euo pipefail

: "${REMOTE_REPO:?Set REMOTE_REPO to an exact /home checkout}"
: "${EXPECTED_HEAD:?Set EXPECTED_HEAD}"
: "${FINAL_DIR:?Set FINAL_DIR under /lustre}"
: "${CONTAINER:?Set CONTAINER}"
: "${TARGET_SNAPSHOT:?Set TARGET_SNAPSHOT}"
: "${DRAFTER_SNAPSHOT:?Set DRAFTER_SNAPSHOT}"
: "${DFLASH_K:?Set DFLASH_K to 5 or 7}"
: "${WANDB_API_KEY:?WANDB_API_KEY is required}"

readonly expected_container_sha256=6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44
readonly expected_drafter_config_sha256=9834d608c9ca53d5548b415471ae9e8ebc9aab6cedfc2a7af95b6bd097373102
readonly training_horizon_steps=1000
readonly scratch_root="/raid/scratch/nrl-qwen3-8b-fixed-4k/${SLURM_JOB_ID}"
readonly config="${REMOTE_REPO}/research/fixed_drafter_qwen3_8b_dflash/config_k${DFLASH_K}.yaml"
readonly checkpoint_dir="${FINAL_DIR}/checkpoints"
readonly train_log="${scratch_root}/logs/train.log"

[[ "${REMOTE_REPO}" == /home/* ]]
[[ "${FINAL_DIR}" == /lustre/* ]]
[[ "${DFLASH_K}" == 5 || "${DFLASH_K}" == 7 ]]
test "$(git -C "${REMOTE_REPO}" rev-parse HEAD)" = "${EXPECTED_HEAD}"
git -C "${REMOTE_REPO}" diff-index --quiet --ignore-submodules=all HEAD --
test -z "$(git -C "${REMOTE_REPO}" ls-files --others --exclude-standard)"
git -C "${REMOTE_REPO}" submodule foreach --quiet --recursive \
  'git diff-index --quiet --ignore-submodules=all HEAD --'
grep -Fqx "sha256=${expected_container_sha256}" "${CONTAINER}.metadata.txt"
test "$(basename "${TARGET_SNAPSHOT}")" = b968826d9c46dd6066d109eabc6255188de91218
test "$(basename "${DRAFTER_SNAPSHOT}")" = 9b41424b7109f9c5413454f481b09a82b85333f4
test "$(sha256sum "${DRAFTER_SNAPSHOT}/config.json" | awk '{print $1}')" = "${expected_drafter_config_sha256}"

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

srun --ntasks=1 \
  --container-image="${CONTAINER}" \
  --container-mounts="/home:/home,/raid:/raid,/lustre:/lustre" \
  --no-container-mount-home \
  bash -lc "set -euo pipefail
export PYTHONPATH='${REMOTE_REPO}'
export HF_HOME='/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home'
export TMPDIR='${scratch_root}/tmp'
export RAY_TMPDIR='${scratch_root}/ray'
export TOKENIZERS_PARALLELISM=false
cd '${REMOTE_REPO}'
/opt/nemo_rl_venv/bin/python '${REMOTE_REPO}/research/fixed_drafter_qwen3_8b_dflash/contract.py' \
  --config '${config}' --steps 1 --k '${DFLASH_K}' | tee '${scratch_root}/contract.json'
UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv uv run --frozen --no-sync \
  examples/run_grpo.py --config '${config}' \
  grpo.max_num_steps='${training_horizon_steps}' \
  checkpointing.save_period=100 \
  checkpointing.checkpoint_must_save_by=00:00:00:01 \
  checkpointing.checkpoint_dir='${checkpoint_dir}' \
  logger.log_dir='${scratch_root}/logs' \
  logger.wandb.name='qwen3-8b-dflash-fixed-k${DFLASH_K}-pps64-gps8-gbs512-step1000-seed42' \
  'logger.wandb.tags=[fixed-drafter,dflash,qwen3-8b,k${DFLASH_K},cudagraph,target-only-grpo,seed42,pps64-gps8-gbs512,step1000]' \
  logger.wandb.config.stage_steps='${training_horizon_steps}' \
  +logger.wandb.config.training_horizon_steps='${training_horizon_steps}' \
  2>&1 | tee '${train_log}'
"

grep -Fq "Capturing CUDA graphs (PIECEWISE)" "${train_log}"
grep -Fq "Graph capturing finished" "${train_log}"
python3 "${REMOTE_REPO}/research/fixed_drafter_qwen3_8b_dflash/resume_contract.py" \
  --checkpoint-dir "${checkpoint_dir}" \
  --expected-step 1 \
  --wandb-log "${train_log}" \
  --gate-manifest "${FINAL_DIR}/gate-manifest.json" \
  --dflash-k "${DFLASH_K}" \
  --git-sha "${EXPECTED_HEAD}" \
  --target-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --drafter-revision 9b41424b7109f9c5413454f481b09a82b85333f4 \
  --container-sha256 "${expected_container_sha256}" \
  --training-horizon-steps "${training_horizon_steps}" \
  --create-gate-manifest
