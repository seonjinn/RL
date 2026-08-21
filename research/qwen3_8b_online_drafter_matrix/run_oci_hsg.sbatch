#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --partition=batch_long
#SBATCH --job-name=q8-online-matrix

set -euo pipefail

: "${REMOTE_REPO:?Set exact /home checkout}"
: "${EXPECTED_HEAD:?Set signed matrix SHA}"
: "${FINAL_DIR:?Set /lustre result directory}"
: "${CONTAINER:?Set immutable container}"
: "${TARGET_SNAPSHOT:?Set exact target snapshot}"
: "${DRAFTER_SNAPSHOT:?Set exact drafter snapshot}"
: "${MATRIX_ARM:?Set baseline, dflash-k5, or dflash-k7}"
: "${WANDB_RUN_ID:?Set persistent W&B run ID}"
: "${WANDB_RESUME:?Set allow or must}"
: "${CHECKPOINT_SOURCE_SHA:=${EXPECTED_HEAD}}"
: "${CHECKPOINT_RUNTIME_SHA:=${EXPECTED_HEAD}}"
: "${IS_GATE:?Set 1 for gate}"
: "${STAGE_MIN_STEP:?Set cumulative milestone}"
: "${STAGE_DEADLINE:?Set checkpoint deadline}"
: "${WANDB_API_KEY:?W&B cluster secret is required}"

readonly container_sha=6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44
readonly target_revision=b968826d9c46dd6066d109eabc6255188de91218
readonly drafter_revision=9b41424b7109f9c5413454f481b09a82b85333f4
readonly experiment="${REMOTE_REPO}/research/qwen3_8b_online_drafter_matrix"
readonly contract="${experiment}/runtime_contract.py"
config_name="$(python3 "${contract}" --arm "${MATRIX_ARM}" --print-config)"
readonly config_name
readonly config="${experiment}/${config_name}"
readonly checkpoint_dir="${FINAL_DIR}/checkpoints"
if [[ "${IS_GATE}" == 1 ]]; then
  readonly manifest="${FINAL_DIR}/gate-manifest.json"
else
  readonly manifest="${FINAL_DIR}/resume-manifest.json"
fi
readonly scratch_root="/raid/scratch/m/${SLURM_JOB_ID}"
readonly ray_root="/raid/scratch/r/${SLURM_JOB_ID}"
readonly train_log="${scratch_root}/train.log"

[[ "${REMOTE_REPO}" == /home/* ]]
[[ "${FINAL_DIR}" == /lustre/* ]]
[[ "${WANDB_RESUME}" == allow || "${WANDB_RESUME}" == must ]]
[[ "${IS_GATE}" == 0 || "${IS_GATE}" == 1 ]]
test "$(git -C "${REMOTE_REPO}" rev-parse HEAD)" = "${EXPECTED_HEAD}"
git -C "${REMOTE_REPO}" diff-index --quiet --ignore-submodules=all HEAD --
test -z "$(git -C "${REMOTE_REPO}" ls-files --others --exclude-standard)"
git -C "${REMOTE_REPO}" submodule foreach --quiet --recursive \
  'git diff-index --quiet --ignore-submodules=all HEAD --'
grep -Fqx "sha256=${container_sha}" "${CONTAINER}.metadata.txt"
test "$(basename "${TARGET_SNAPSHOT}")" = "${target_revision}"
test "$(basename "${DRAFTER_SNAPSHOT}")" = "${drafter_revision}"
test -f "${config}"

if [[ "${IS_GATE}" == 1 ]]; then
  previous_step=0
  val_period=1
  [[ "${WANDB_RESUME}" == allow ]]
  test ! -e "${manifest}"
else
  previous_step="$(python3 "${contract}" --checkpoint-dir "${checkpoint_dir}" --print-latest-step)"
  val_period=1000000
  [[ "${WANDB_RESUME}" == must ]]
  python3 "${contract}" --arm "${MATRIX_ARM}" --checkpoint-dir "${checkpoint_dir}" \
    --manifest "${manifest}" --runtime-git-sha "${CHECKPOINT_RUNTIME_SHA}" \
    --checkpoint-source-sha "${CHECKPOINT_SOURCE_SHA}" \
    --wandb-run-id "${WANDB_RUN_ID}" \
    --target-revision "${target_revision}" --drafter-revision "${drafter_revision}" \
    --container-sha256 "${container_sha}" --validate-resume-manifest
  if (( previous_step >= STAGE_MIN_STEP )); then exit 0; fi
fi

mkdir -p "${scratch_root}" "${ray_root}" "${FINAL_DIR}/logs" "${checkpoint_dir}"
archive() {
  status=$?
  trap - EXIT
  set +e
  test -f "${train_log}" && tail -n 6000 "${train_log}" > \
    "${FINAL_DIR}/logs/job-${SLURM_JOB_ID}-to-${STAGE_MIN_STEP}.log"
  exit "${status}"
}
trap archive EXIT

srun --ntasks=1 --kill-on-bad-exit=1 --wait=0 \
  --container-image="${CONTAINER}" \
  --container-mounts="/home:/home,/raid:/raid,/lustre:/lustre" \
  --no-container-mount-home bash -lc "set -euo pipefail
export PYTHONPATH='${REMOTE_REPO}'
export HF_HOME='/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home'
export TMPDIR='${scratch_root}'
export RAY_TMPDIR='${ray_root}'
export TOKENIZERS_PARALLELISM=false
cd '${REMOTE_REPO}'
UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv uv run --frozen --no-sync \
  examples/run_grpo.py --config '${config}' \
  grpo.max_num_steps=1000 grpo.val_period='${val_period}' \
  grpo.num_prompts_per_step=8 grpo.num_generations_per_prompt=4 \
  checkpointing.save_period=100 \
  checkpointing.checkpoint_must_save_by='${STAGE_DEADLINE}' \
  checkpointing.checkpoint_dir='${checkpoint_dir}' \
  policy.model_name='${TARGET_SNAPSHOT}' policy.tokenizer.name='${TARGET_SNAPSHOT}' \
  policy.train_global_batch_size=32 \
  policy.sequence_packing.enabled=false \
  policy.megatron_cfg.sequence_parallel=false \
  policy.draft.model_name='${DRAFTER_SNAPSHOT}' \
  logger.log_dir='${scratch_root}/logs' \
  logger.wandb.project=sna-nemo-rl-online-drafter \
  +logger.wandb.id='${WANDB_RUN_ID}' +logger.wandb.resume='${WANDB_RESUME}' \
  +logger.wandb.config.git_sha='${CHECKPOINT_SOURCE_SHA}' \
  +logger.wandb.config.checkpoint_source_sha='${CHECKPOINT_SOURCE_SHA}' \
  +logger.wandb.config.runtime_git_sha='${CHECKPOINT_RUNTIME_SHA}' \
  +logger.wandb.config.runner_git_sha='${EXPECTED_HEAD}' \
  +logger.wandb.config.matrix_arm='${MATRIX_ARM}' \
  2>&1 | tee '${train_log}'
grep -RhaE 'draft_(update_probe=complete|refit_(manifest|load|finalize)=)' \
  '${ray_root}' >> '${train_log}' || true
if [[ '${IS_GATE}' == 1 ]]; then
  /opt/nemo_rl_venv/bin/python '${experiment}/validate_gate.py' \
    --arm '${MATRIX_ARM}' \
    --wandb-run-path 'nvidia/sna-nemo-rl-online-drafter/${WANDB_RUN_ID}' \
    --log '${train_log}'
fi
"

if [[ "${IS_GATE}" == 1 ]]; then
  grep -Fq "cudagraph_mode: PIECEWISE" "${train_log}"
  grep -Fq "Graph capturing finished" "${train_log}"
fi
current_step="$(python3 "${contract}" --checkpoint-dir "${checkpoint_dir}" --print-latest-step)"
python3 "${contract}" --previous-step "${previous_step}" --current-step "${current_step}" \
  --required-min-step "${STAGE_MIN_STEP}"
if [[ "${IS_GATE}" == 1 ]]; then
  python3 "${contract}" --arm "${MATRIX_ARM}" --checkpoint-dir "${checkpoint_dir}" \
    --manifest "${manifest}" --git-sha "${EXPECTED_HEAD}" \
    --wandb-run-id "${WANDB_RUN_ID}" --target-revision "${target_revision}" \
    --drafter-revision "${drafter_revision}" --container-sha256 "${container_sha}" \
    --write-manifest
fi
echo "WANDB_URL=https://wandb.ai/nvidia/sna-nemo-rl-online-drafter/runs/${WANDB_RUN_ID}"
