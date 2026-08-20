#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --partition=batch_long
#SBATCH --job-name=q8-online-dflash-k7-resume

set -euo pipefail

: "${REMOTE_REPO:?Set exact /home checkout}"
: "${EXPECTED_HEAD:?Set signed composition SHA}"
: "${FINAL_DIR:?Set /lustre result directory}"
: "${CONTAINER:?Set immutable container}"
: "${TARGET_SNAPSHOT:?Set exact target snapshot}"
: "${DRAFTER_SNAPSHOT:?Set exact drafter snapshot}"
: "${SOURCE_GATE_SHA:?Set exact smoke checkpoint composition SHA}"
: "${STAGE_MIN_STEP:?Set cumulative milestone}"
: "${STAGE_DEADLINE:?Set checkpoint deadline}"
: "${WANDB_RUN_ID:?Set the persistent science W&B run ID}"
: "${WANDB_PROJECT:?Set the science W&B project}"
: "${WANDB_RESUME:?Set allow for the first science segment and must thereafter}"
: "${WANDB_API_KEY:?W&B cluster secret is required}"

readonly container_sha=6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44
readonly target_revision=b968826d9c46dd6066d109eabc6255188de91218
readonly drafter_revision=9b41424b7109f9c5413454f481b09a82b85333f4
readonly experiment="${REMOTE_REPO}/research/qwen3_8b_dflash_online_cp1"
readonly contract="${experiment}/resume_contract.py"
readonly checkpoint_dir="${FINAL_DIR}/checkpoints"
readonly manifest="${FINAL_DIR}/gate-manifest.json"
readonly scratch_root="/raid/scratch/o/${SLURM_JOB_ID}"
readonly ray_root="/raid/scratch/r/${SLURM_JOB_ID}"
readonly train_log="${scratch_root}/train.log"

[[ "${REMOTE_REPO}" == /home/* ]]
[[ "${FINAL_DIR}" == /lustre/* ]]
[[ "${WANDB_PROJECT}" == nemo-rl-specdec-eval || "${WANDB_PROJECT}" == sna-nemo-rl-online-drafter ]]
[[ "${WANDB_RESUME}" == allow || "${WANDB_RESUME}" == must ]]
test "$(git -C "${REMOTE_REPO}" rev-parse HEAD)" = "${EXPECTED_HEAD}"
git -C "${REMOTE_REPO}" diff-index --quiet --ignore-submodules=all HEAD --
test -z "$(git -C "${REMOTE_REPO}" ls-files --others --exclude-standard)"
git -C "${REMOTE_REPO}" submodule foreach --quiet --recursive \
  'git diff-index --quiet --ignore-submodules=all HEAD --'
grep -Fqx "sha256=${container_sha}" "${CONTAINER}.metadata.txt"
test "$(basename "${TARGET_SNAPSHOT}")" = "${target_revision}"
test "$(basename "${DRAFTER_SNAPSHOT}")" = "${drafter_revision}"
previous_step="$(python3 "${contract}" --checkpoint-dir "${checkpoint_dir}" --print-latest-step)"
python3 "${contract}" --checkpoint-dir "${checkpoint_dir}" \
  --expected-step "${previous_step}" --manifest "${manifest}" \
  --git-sha "${SOURCE_GATE_SHA}" --target-revision "${target_revision}" \
  --drafter-revision "${drafter_revision}" --container-sha256 "${container_sha}" \
  --validate-manifest
if (( previous_step >= STAGE_MIN_STEP )); then exit 0; fi

mkdir -p "${scratch_root}" "${ray_root}" "${FINAL_DIR}/logs"
archive() {
  status=$?
  trap - EXIT
  set +e
  test -f "${train_log}" && tail -n 4000 "${train_log}" > \
    "${FINAL_DIR}/logs/job-${SLURM_JOB_ID}-to-${STAGE_MIN_STEP}.log"
  exit "${status}"
}
trap archive EXIT

srun --ntasks=1 --container-image="${CONTAINER}" \
  --container-mounts="/home:/home,/raid:/raid,/lustre:/lustre" \
  --no-container-mount-home bash -lc "set -euo pipefail
export PYTHONPATH='${REMOTE_REPO}'
export HF_HOME='/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home'
export TMPDIR='${scratch_root}'
export RAY_TMPDIR='${ray_root}'
export TOKENIZERS_PARALLELISM=false
cd '${REMOTE_REPO}'
UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv uv run --frozen --no-sync \
  examples/run_grpo.py --config '${experiment}/config.yaml' \
  grpo.max_num_steps=1000 checkpointing.save_period=100 \
  checkpointing.checkpoint_must_save_by='${STAGE_DEADLINE}' \
  checkpointing.checkpoint_dir='${checkpoint_dir}' \
  policy.model_name='${TARGET_SNAPSHOT}' policy.tokenizer.name='${TARGET_SNAPSHOT}' \
  policy.draft.model_name='${DRAFTER_SNAPSHOT}' \
  policy.generation.vllm_kwargs.speculative_config.model='${DRAFTER_SNAPSHOT}' \
  logger.log_dir='${scratch_root}/logs' \
  logger.wandb.project='${WANDB_PROJECT}' \
  +logger.wandb.id='${WANDB_RUN_ID}' +logger.wandb.resume='${WANDB_RESUME}' \
  +logger.wandb.config.git_sha='${EXPECTED_HEAD}' \
  +logger.wandb.config.stage_steps='${STAGE_MIN_STEP}' \
  2>&1 | tee '${train_log}'
"

grep -Fq "Capturing CUDA graphs (PIECEWISE)" "${train_log}"
grep -Fq "Graph capturing finished" "${train_log}"
current_step="$(python3 "${contract}" --checkpoint-dir "${checkpoint_dir}" --print-latest-step)"
python3 "${contract}" --checkpoint-dir "${checkpoint_dir}" \
  --previous-step "${previous_step}" --expected-step "${current_step}" \
  --required-min-step "${STAGE_MIN_STEP}"
