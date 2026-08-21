#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --partition=batch_long
#SBATCH --job-name=q8-dflash-nonnsys-ab
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -Eeuo pipefail

: "${REMOTE_REPO:?Set exact /home checkout}"
: "${EXPECTED_HEAD:?Set signed harness SHA}"
: "${FINAL_DIR:?Set arm-specific /lustre result directory}"
: "${CONTAINER:?Set immutable container}"
: "${TARGET_SNAPSHOT:?Set exact target snapshot}"
: "${DRAFTER_SNAPSHOT:?Set exact drafter snapshot}"
: "${ARM:?Set fixed or online}"
: "${WANDB_RUN_ID:?Set a fresh W&B run ID}"
: "${WANDB_PROJECT:?Set the W&B project}"
: "${WANDB_API_KEY:?W&B cluster secret is required}"

readonly optimized_source_sha=79e80af96a13522e6049658663a8c40ab21e8314
readonly container_sha=6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44
readonly target_revision=b968826d9c46dd6066d109eabc6255188de91218
readonly drafter_revision=9b41424b7109f9c5413454f481b09a82b85333f4
readonly experiment="${REMOTE_REPO}/research/qwen3_8b_dflash_nonnsys_ab"
readonly contract="${experiment}/runtime_contract.py"
readonly scratch_root="/raid/scratch/dflash-ab/${SLURM_JOB_ID}"
readonly ray_root="/raid/scratch/dflash-ab-ray/${SLURM_JOB_ID}"
readonly train_log="${scratch_root}/train.log"

[[ "${REMOTE_REPO}" == /home/* ]]
[[ "${FINAL_DIR}" == /lustre/* ]]
[[ "${WANDB_PROJECT}" == sna-nemo-rl-online-drafter ]]
mkdir -p "${scratch_root}" "${ray_root}" "${FINAL_DIR}"
archive() {
  local status=$?
  trap - EXIT
  set +e
  test -f "${train_log}" && tail -n 10000 "${train_log}" > "${FINAL_DIR}/train.log"
  {
    echo "job_id=${SLURM_JOB_ID}"
    echo "exit_code=${status}"
    echo "arm=${ARM}"
    echo "git_sha=${EXPECTED_HEAD}"
    echo "optimized_source_sha=${optimized_source_sha}"
    echo "wandb_run_id=${WANDB_RUN_ID}"
    echo "container_sha256=${container_sha}"
    echo "finished_at=$(date -Is)"
  } > "${FINAL_DIR}/result.txt"
  exit "${status}"
}
trap archive EXIT

test "$(git -C "${REMOTE_REPO}" rev-parse HEAD)" = "${EXPECTED_HEAD}"
test "$(git -C "${REMOTE_REPO}" merge-base "${optimized_source_sha}" "${EXPECTED_HEAD}")" = \
  "${optimized_source_sha}"
unexpected_changes="$(
  git -C "${REMOTE_REPO}" diff --name-only \
    "${optimized_source_sha}" "${EXPECTED_HEAD}" -- | \
    grep -v '^research/qwen3_8b_dflash_nonnsys_ab/' || true
)"
test -z "${unexpected_changes}"
git -C "${REMOTE_REPO}" diff-index --quiet --ignore-submodules=all HEAD --
test -z "$(git -C "${REMOTE_REPO}" ls-files --others --exclude-standard)"
git -C "${REMOTE_REPO}" submodule foreach --quiet --recursive \
  'git diff-index --quiet --ignore-submodules=all HEAD --'
grep -Fqx "sha256=${container_sha}" "${CONTAINER}.metadata.txt"
test "$(basename "${TARGET_SNAPSHOT}")" = "${target_revision}"
test "$(basename "${DRAFTER_SNAPSHOT}")" = "${drafter_revision}"

read -r config_path draft_training_enabled update_probe_enabled < <(
  python3 "${contract}" --arm "${ARM}"
)
readonly config="${REMOTE_REPO}/${config_path}"
test -f "${config}"

srun --ntasks=1 --container-image="${CONTAINER}" \
  --container-mounts="/home:/home,/raid:/raid,/lustre:/lustre" \
  --no-container-mount-home bash -lc "set -Eeuo pipefail
export PYTHONPATH='${REMOTE_REPO}'
export HF_HOME='/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home'
export TMPDIR='${scratch_root}'
export RAY_TMPDIR='${ray_root}'
export TOKENIZERS_PARALLELISM=false
export WANDB__DISABLE_STATS=true
unset NRL_NSYS_WORKER_PATTERNS
unset NRL_NSYS_PROFILE_STEP_RANGE
unset NRL_NSYS_EXTRA_OPTIONS
cd '${REMOTE_REPO}'
UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv uv run --frozen --no-sync \
  examples/run_grpo.py --config '${config}' \
  grpo.max_num_steps=50 grpo.val_period=1000000 \
  grpo.val_at_start=false grpo.val_at_end=false \
  checkpointing.enabled=false \
  policy.model_name='${TARGET_SNAPSHOT}' policy.tokenizer.name='${TARGET_SNAPSHOT}' \
  policy.draft.model_name='${DRAFTER_SNAPSHOT}' \
  policy.draft.update_probe_enabled='${update_probe_enabled}' \
  policy.generation.vllm_kwargs.speculative_config.model='${DRAFTER_SNAPSHOT}' \
  logger.log_dir='${scratch_root}/logs' \
  logger.wandb_enabled=true logger.tensorboard_enabled=false \
  logger.wandb.entity=nvidia logger.wandb.project='${WANDB_PROJECT}' \
  +logger.wandb.id='${WANDB_RUN_ID}' +logger.wandb.resume=never \
  +logger.wandb.config.ab_arm='${ARM}' \
  logger.wandb.config.draft_training_enabled='${draft_training_enabled}' \
  +logger.wandb.config.optimized_source_sha='${optimized_source_sha}' \
  +logger.wandb.config.harness_sha='${EXPECTED_HEAD}' \
  +logger.wandb.config.performance_window='steps_5_through_49' \
  2>&1 | tee '${train_log}'
"

grep -Fq "Step 50" "${train_log}"
grep -Fq "Capturing CUDA graphs (PIECEWISE)" "${train_log}"
grep -Fq "Graph capturing finished" "${train_log}"
echo "WANDB_URL=https://wandb.ai/nvidia/${WANDB_PROJECT}/runs/${WANDB_RUN_ID}"
