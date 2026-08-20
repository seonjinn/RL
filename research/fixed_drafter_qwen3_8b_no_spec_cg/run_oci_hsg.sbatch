#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --account=nemotron_n3_post
#SBATCH --partition=batch_long
#SBATCH --job-name=qwen8-nospec-cg-gate

set -euo pipefail

: "${REMOTE_REPO:?Set REMOTE_REPO}"
: "${EXPECTED_HEAD:?Set EXPECTED_HEAD}"
: "${RUN_DIR:?Set RUN_DIR}"
: "${CONTAINER:?Set CONTAINER}"
: "${TARGET_SNAPSHOT:?Set TARGET_SNAPSHOT}"
: "${WANDB_API_KEY:?WANDB_API_KEY is required}"
: "${TRAINING_HORIZON_STEPS:?Set TRAINING_HORIZON_STEPS to 1000}"

expected_container_sha256=6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44
test "${TRAINING_HORIZON_STEPS}" = 1000
test "$(git -C "${REMOTE_REPO}" rev-parse HEAD)" = "${EXPECTED_HEAD}"
git -C "${REMOTE_REPO}" diff-index --quiet --ignore-submodules=all HEAD --
test -z "$(git -C "${REMOTE_REPO}" ls-files --others --exclude-standard)"
! git -C "${REMOTE_REPO}" submodule status --recursive | grep -Eq '^[+-U]'
git -C "${REMOTE_REPO}" submodule foreach --quiet --recursive \
  'git diff-index --quiet --ignore-submodules=all HEAD --'
test -f "${CONTAINER}"
grep -Fqx "sha256=${expected_container_sha256}" "${CONTAINER}.metadata.txt"
test "$(basename "${TARGET_SNAPSHOT}")" = b968826d9c46dd6066d109eabc6255188de91218
test -f "${TARGET_SNAPSHOT}/config.json"

config="${REMOTE_REPO}/research/fixed_drafter_qwen3_8b_no_spec_cg/config.yaml"
contract="${REMOTE_REPO}/research/fixed_drafter_qwen3_8b_no_spec_cg/contract.py"
mkdir -p "${RUN_DIR}/checkpoints" "${RUN_DIR}/logs"
python3 "${contract}" --config "${config}" --validate-config | tee "${RUN_DIR}/contract.json"

srun --ntasks=1 \
  --container-image="${CONTAINER}" \
  --container-mounts="/lustre:/lustre" \
  --no-container-mount-home \
  bash -lc "set -euo pipefail
export PYTHONPATH='${REMOTE_REPO}'
export HF_HOME='/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home'
export TMPDIR='/tmp/nrl-${SLURM_JOB_ID}'
export RAY_TMPDIR='/tmp/nrl-${SLURM_JOB_ID}'
mkdir -p '/tmp/nrl-${SLURM_JOB_ID}'
export TOKENIZERS_PARALLELISM=false
cd '${REMOTE_REPO}'
UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv uv run --frozen --no-sync \
  examples/run_grpo.py --config '${config}' \
  grpo.max_num_steps='${TRAINING_HORIZON_STEPS}' \
  checkpointing.save_period=100 \
  checkpointing.checkpoint_must_save_by=00:00:00:01 \
  checkpointing.checkpoint_dir='${RUN_DIR}/checkpoints' \
  logger.log_dir='${RUN_DIR}/logs' \
  logger.wandb.name='qwen3-8b-no-specdec-cudagraph-step1000-seed42' \
  'logger.wandb.tags=[baseline,no-specdec,qwen3-8b,cudagraph,target-only-grpo,seed42,step1000]' \
  logger.wandb.config.stage_steps='${TRAINING_HORIZON_STEPS}' \
  +logger.wandb.config.training_horizon_steps='${TRAINING_HORIZON_STEPS}' \
  2>&1 | tee '${RUN_DIR}/train.log'
"

grep -Fq "Capturing CUDA graphs (PIECEWISE)" "${RUN_DIR}/train.log"
grep -Fq "Graph capturing finished" "${RUN_DIR}/train.log"
python3 "${contract}" \
  --checkpoint-dir "${RUN_DIR}/checkpoints" \
  --expected-step 1 \
  --wandb-log "${RUN_DIR}/train.log" \
  --gate-manifest "${RUN_DIR}/gate-manifest.json" \
  --git-sha "${EXPECTED_HEAD}" \
  --create-gate-manifest
