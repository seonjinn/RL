#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --partition=batch_long
#SBATCH --job-name=q8-fixed-k7-nonnsys-sanity-b

set -euo pipefail

: "${REMOTE_REPO:?Set exact /home checkout}"
: "${EXPECTED_HEAD:?Set signed execution SHA}"
: "${FINAL_DIR:?Set fresh /lustre result directory}"
: "${CONTAINER:?Set immutable container}"
: "${TARGET_SNAPSHOT:?Set exact target snapshot}"
: "${DRAFTER_SNAPSHOT:?Set exact drafter snapshot}"
: "${WANDB_RUN_ID:?Set fresh W&B run ID}"
: "${WANDB_PROJECT:?Set W&B project}"
: "${WANDB_RESUME:?Set W&B resume policy}"
: "${IS_GATE:?Set 1 for smoke, 0 for continuation}"
: "${STAGE_MIN_STEP:?Set cumulative milestone}"
: "${STAGE_DEADLINE:?Set checkpoint deadline}"
: "${SOURCE_GATE_SHA:?Set exact smoke composition SHA}"
: "${WANDB_API_KEY:?W&B cluster secret is required}"

readonly container_sha=6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44
readonly target_revision=b968826d9c46dd6066d109eabc6255188de91218
readonly drafter_revision=9b41424b7109f9c5413454f481b09a82b85333f4
readonly experiment="${REMOTE_REPO}/research/qwen3_8b_dflash_nonnsys_sanity_b"
readonly fixed_source="${REMOTE_REPO}/research/qwen3_8b_dflash_fixed_dense_control"
readonly contract="${fixed_source}/resume_contract.py"
readonly checkpoint_dir="${FINAL_DIR}/checkpoints"
readonly manifest="${FINAL_DIR}/gate-manifest.json"
readonly scratch_root="/raid/scratch/fixed-sanity/${SLURM_JOB_ID}"
readonly ray_root="/raid/scratch/r/${SLURM_JOB_ID}"
readonly train_log="${scratch_root}/train.log"

[[ "${REMOTE_REPO}" == /home/* ]]
[[ "${FINAL_DIR}" == /lustre/* ]]
[[ "${WANDB_PROJECT}" == sna-nemo-rl-online-drafter ]]
[[ "${WANDB_RESUME}" == allow ]]
[[ "${IS_GATE}" == 1 ]]
[[ "${STAGE_MIN_STEP}" == 50 ]]
[[ "${STAGE_DEADLINE}" == 00:00:50:00 ]]
test "$(git -C "${REMOTE_REPO}" rev-parse HEAD)" = "${EXPECTED_HEAD}"
git -C "${REMOTE_REPO}" diff-index --quiet --ignore-submodules=all HEAD --
test -z "$(git -C "${REMOTE_REPO}" ls-files --others --exclude-standard)"
git -C "${REMOTE_REPO}" submodule foreach --quiet --recursive \
  'git diff-index --quiet --ignore-submodules=all HEAD --'
grep -Fqx "sha256=${container_sha}" "${CONTAINER}.metadata.txt"
test "$(basename "${TARGET_SNAPSHOT}")" = "${target_revision}"
test "$(basename "${DRAFTER_SNAPSHOT}")" = "${drafter_revision}"
python3 "${fixed_source}/runtime_contract.py" --ray-temp-root "${ray_root}"
test ! -e "${manifest}"

mkdir -p "${scratch_root}" "${ray_root}" "${FINAL_DIR}/logs" "${checkpoint_dir}"
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
  examples/run_grpo.py --config '${experiment}/fixed_config.yaml' \
  grpo.max_num_steps=50 grpo.val_period=1 checkpointing.save_period=50 \
  checkpointing.checkpoint_must_save_by='${STAGE_DEADLINE}' \
  checkpointing.checkpoint_dir='${checkpoint_dir}' \
  policy.model_name='${TARGET_SNAPSHOT}' policy.tokenizer.name='${TARGET_SNAPSHOT}' \
  policy.draft.model_name='${DRAFTER_SNAPSHOT}' \
  policy.draft.update_probe_enabled=false \
  policy.generation.vllm_kwargs.speculative_config.model='${DRAFTER_SNAPSHOT}' \
  logger.log_dir='${scratch_root}/logs' \
  logger.wandb.project='${WANDB_PROJECT}' \
  logger.wandb.name='qwen3-8b-dflash-fixed-k7-nonnsys-step50-sanity-b-${WANDB_RUN_ID}' \
  +logger.wandb.id='${WANDB_RUN_ID}' +logger.wandb.resume='${WANDB_RESUME}' \
  +logger.wandb.config.git_sha='${EXPECTED_HEAD}' \
  +logger.wandb.config.stage_steps='${STAGE_MIN_STEP}' \
  2>&1 | tee '${train_log}'
/opt/nemo_rl_venv/bin/python '${fixed_source}/validate_gate.py' \
  --wandb-run-path 'nvidia/${WANDB_PROJECT}/${WANDB_RUN_ID}' \
  --log '${train_log}'
"

grep -Fq "Capturing CUDA graphs (PIECEWISE)" "${train_log}"
grep -Fq "Graph capturing finished" "${train_log}"
current_step="$(python3 "${contract}" \
  --checkpoint-dir "${checkpoint_dir}" --print-latest-step)"
python3 "${contract}" --checkpoint-dir "${checkpoint_dir}" \
  --previous-step 0 --expected-step "${current_step}" \
  --required-min-step "${STAGE_MIN_STEP}"
python3 "${contract}" --checkpoint-dir "${checkpoint_dir}" \
  --expected-step "${current_step}" --manifest "${manifest}" \
  --git-sha "${EXPECTED_HEAD}" --wandb-run-id "${WANDB_RUN_ID}" \
  --target-revision "${target_revision}" --drafter-revision "${drafter_revision}" \
  --container-sha256 "${container_sha}" --write-manifest
python3 "${experiment}/write_runtime_manifest.py" --arm fixed \
  --output "${FINAL_DIR}/sanity-manifest.json" --git-sha "${EXPECTED_HEAD}" \
  --wandb-run-id "${WANDB_RUN_ID}" --job-id "${SLURM_JOB_ID}" \
  --checkpoint-step "${current_step}" --validator fixed-dense-control-gate
echo "WANDB_URL=https://wandb.ai/nvidia/${WANDB_PROJECT}/runs/${WANDB_RUN_ID}"

