#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --partition=batch_long
#SBATCH --job-name=q8-online-k7-nonnsys-sanity-b

set -euo pipefail

: "${REMOTE_REPO:?Set exact /home checkout}"
: "${EXPECTED_HEAD:?Set signed execution SHA}"
: "${FINAL_DIR:?Set fresh /lustre result directory}"
: "${CONTAINER:?Set immutable container}"
: "${TARGET_SNAPSHOT:?Set exact target snapshot}"
: "${DRAFTER_SNAPSHOT:?Set exact drafter snapshot}"
: "${WANDB_RUN_ID:?Set fresh W&B run ID}"
: "${WANDB_PROJECT:?Set the W&B project}"
: "${WANDB_RESUME:?Set the initial W&B resume policy}"
: "${WANDB_API_KEY:?W&B cluster secret is required}"

readonly container_sha=6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44
readonly target_revision=b968826d9c46dd6066d109eabc6255188de91218
readonly drafter_revision=9b41424b7109f9c5413454f481b09a82b85333f4
readonly experiment="${REMOTE_REPO}/research/qwen3_8b_dflash_nonnsys_sanity_b"
readonly resume_contract="${REMOTE_REPO}/research/qwen3_8b_dflash_online_cp1/resume_contract.py"
readonly checkpoint_dir="${FINAL_DIR}/checkpoints"
readonly scratch_root="/raid/scratch/o/${SLURM_JOB_ID}"
readonly ray_root="/raid/scratch/r/${SLURM_JOB_ID}"
readonly train_log="${scratch_root}/train.log"
readonly evidence_log="${scratch_root}/sanity-evidence.log"

[[ "${REMOTE_REPO}" == /home/* ]]
[[ "${FINAL_DIR}" == /lustre/* ]]
[[ "${WANDB_PROJECT}" == sna-nemo-rl-online-drafter ]]
[[ "${WANDB_RESUME}" == allow ]]
test "$(git -C "${REMOTE_REPO}" rev-parse HEAD)" = "${EXPECTED_HEAD}"
git -C "${REMOTE_REPO}" diff-index --quiet --ignore-submodules=all HEAD --
test -z "$(git -C "${REMOTE_REPO}" ls-files --others --exclude-standard)"
git -C "${REMOTE_REPO}" submodule foreach --quiet --recursive \
  'git diff-index --quiet --ignore-submodules=all HEAD --'
grep -Fqx "sha256=${container_sha}" "${CONTAINER}.metadata.txt"
test "$(basename "${TARGET_SNAPSHOT}")" = "${target_revision}"
test "$(basename "${DRAFTER_SNAPSHOT}")" = "${drafter_revision}"

mkdir -p "${scratch_root}" "${ray_root}" "${FINAL_DIR}/logs" "${checkpoint_dir}"
archive() {
  status=$?
  trap - EXIT
  set +e
  if test -f "${evidence_log}"; then
    tail -n 4000 "${evidence_log}" > "${FINAL_DIR}/logs/job-${SLURM_JOB_ID}-sanity.log"
  elif test -f "${train_log}"; then
    tail -n 4000 "${train_log}" > "${FINAL_DIR}/logs/job-${SLURM_JOB_ID}-sanity.log"
  fi
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
  examples/run_grpo.py --config '${experiment}/online_config.yaml' \
  grpo.max_num_steps=50 grpo.val_period=1 checkpointing.save_period=50 \
  checkpointing.checkpoint_must_save_by=00:00:50:00 \
  checkpointing.checkpoint_dir='${checkpoint_dir}' \
  policy.model_name='${TARGET_SNAPSHOT}' policy.tokenizer.name='${TARGET_SNAPSHOT}' \
  policy.draft.model_name='${DRAFTER_SNAPSHOT}' \
  policy.draft.update_probe_enabled=false \
  policy.generation.vllm_kwargs.speculative_config.model='${DRAFTER_SNAPSHOT}' \
  logger.log_dir='${scratch_root}/logs' \
  logger.wandb.project='${WANDB_PROJECT}' \
  logger.wandb.name='qwen3-8b-dflash-online-k7-nonnsys-step50-sanity-b-${WANDB_RUN_ID}' \
  +logger.wandb.id='${WANDB_RUN_ID}' +logger.wandb.resume='${WANDB_RESUME}' \
  +logger.wandb.config.git_sha='${EXPECTED_HEAD}' \
  +logger.wandb.config.stage_steps=50 \
  2>&1 | tee '${train_log}'
cp '${train_log}' '${evidence_log}'
grep -RhaE 'draft_refit_(manifest|load|finalize)=' '${ray_root}' \
  >> '${evidence_log}' || true
/opt/nemo_rl_venv/bin/python '${experiment}/validate_online_sanity.py' \
  --wandb-run-path 'nvidia/${WANDB_PROJECT}/${WANDB_RUN_ID}' \
  --log '${evidence_log}'
"

grep -Fq "Capturing CUDA graphs (PIECEWISE)" "${train_log}"
grep -Fq "Graph capturing finished" "${train_log}"
current_step="$(python3 "${resume_contract}" \
  --checkpoint-dir "${checkpoint_dir}" --print-latest-step)"
python3 "${resume_contract}" --checkpoint-dir "${checkpoint_dir}" \
  --previous-step 0 --expected-step "${current_step}" --required-min-step 50 \
  --manifest "${FINAL_DIR}/gate-manifest.json" \
  --git-sha "${EXPECTED_HEAD}" --wandb-run-id "${WANDB_RUN_ID}" \
  --target-revision "${target_revision}" --drafter-revision "${drafter_revision}" \
  --container-sha256 "${container_sha}" --write-manifest
python3 "${experiment}/write_runtime_manifest.py" --arm online \
  --output "${FINAL_DIR}/sanity-manifest.json" --git-sha "${EXPECTED_HEAD}" \
  --wandb-run-id "${WANDB_RUN_ID}" --job-id "${SLURM_JOB_ID}" \
  --checkpoint-step "${current_step}" --validator online-nonprobe-sanity
echo "WANDB_URL=https://wandb.ai/nvidia/${WANDB_PROJECT}/runs/${WANDB_RUN_ID}"

