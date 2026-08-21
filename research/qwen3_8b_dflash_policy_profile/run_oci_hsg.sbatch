#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --partition=batch_long
#SBATCH --job-name=q8-dflash-policy-profile
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -Eeuo pipefail

: "${REMOTE_REPO:?Set exact /home checkout}"
: "${EXPECTED_HEAD:?Set signed source SHA}"
: "${FINAL_DIR:?Set the arm-specific /lustre result directory}"
: "${CONTAINER:?Set immutable container}"
: "${TARGET_SNAPSHOT:?Set exact target snapshot}"
: "${DRAFTER_SNAPSHOT:?Set exact drafter snapshot}"
: "${PROFILE_ARM:?Set fixed-control, online-current, or online-probe-off}"

readonly container_sha=6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44
readonly target_revision=b968826d9c46dd6066d109eabc6255188de91218
readonly drafter_revision=9b41424b7109f9c5413454f481b09a82b85333f4
readonly profile_dir="${REMOTE_REPO}/research/qwen3_8b_dflash_policy_profile"
readonly contract="${profile_dir}/runtime_contract.py"
readonly scratch_root="/raid/scratch/p/${SLURM_JOB_ID}"
readonly ray_root="/raid/scratch/r/${SLURM_JOB_ID}"
readonly train_log="${scratch_root}/train.log"

[[ "${REMOTE_REPO}" == /home/* ]]
[[ "${FINAL_DIR}" == /lustre/* ]]
test "$(git -C "${REMOTE_REPO}" rev-parse HEAD)" = "${EXPECTED_HEAD}"
git -C "${REMOTE_REPO}" diff-index --quiet --ignore-submodules=all HEAD --
test -z "$(git -C "${REMOTE_REPO}" ls-files --others --exclude-standard)"
git -C "${REMOTE_REPO}" submodule foreach --quiet --recursive \
  'git diff-index --quiet --ignore-submodules=all HEAD --'
grep -Fqx "sha256=${container_sha}" "${CONTAINER}.metadata.txt"
test "$(basename "${TARGET_SNAPSHOT}")" = "${target_revision}"
test "$(basename "${DRAFTER_SNAPSHOT}")" = "${drafter_revision}"

read -r config_name update_probe_enabled < <(
  python3 "${contract}" --arm "${PROFILE_ARM}"
)
readonly config="${REMOTE_REPO}/research/${config_name}/config.yaml"
test -f "${config}"

mkdir -p "${scratch_root}" "${ray_root}" "${FINAL_DIR}"
profile_report_count=0
copy_profiles() {
  local index=0
  while IFS= read -r -d '' report; do
    cp "${report}" "${FINAL_DIR}/policy-rank-${index}.nsys-rep"
    index=$((index + 1))
  done < <(
    find "${scratch_root}" "${ray_root}" -type f -name '*.nsys-rep' -print0 2>/dev/null
  )
  profile_report_count=${index}
}
archive() {
  local status=$?
  trap - EXIT
  set +e
  test -f "${train_log}" && tail -n 4000 "${train_log}" > "${FINAL_DIR}/train.log"
  copy_profiles
  {
    echo "job_id=${SLURM_JOB_ID}"
    echo "exit_code=${status}"
    echo "git_sha=${EXPECTED_HEAD}"
    echo "profile_arm=${PROFILE_ARM}"
    echo "container_sha256=${container_sha}"
    echo "profile_reports=${profile_report_count}"
    echo "finished_at=$(date -Is)"
  } > "${FINAL_DIR}/result.txt"
  exit "${status}"
}
trap archive EXIT

srun --ntasks=1 --container-image="${CONTAINER}" \
  --container-mounts="/home:/home,/raid:/raid,/lustre:/lustre" \
  --no-container-mount-home bash -lc "set -Eeuo pipefail
export PYTHONPATH='${REMOTE_REPO}'
export HF_HOME='/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home'
export TMPDIR='${scratch_root}'
export RAY_TMPDIR='${ray_root}'
export TOKENIZERS_PARALLELISM=false
export NRL_NSYS_WORKER_PATTERNS=megatron_policy_worker
export NRL_NSYS_PROFILE_STEP_RANGE=3:6
export NRL_NSYS_EXTRA_OPTIONS='{"cuda-graph-trace": "node", "cpuctxsw": "none", "o": "${scratch_root}/policy_%p"}'
cd '${REMOTE_REPO}'
UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv uv run --frozen --no-sync \
  examples/run_grpo.py --config '${config}' \
  grpo.max_num_steps=6 grpo.val_period=1000000 \
  grpo.val_at_start=false grpo.val_at_end=false \
  checkpointing.enabled=false \
  policy.model_name='${TARGET_SNAPSHOT}' policy.tokenizer.name='${TARGET_SNAPSHOT}' \
  policy.draft.model_name='${DRAFTER_SNAPSHOT}' \
  policy.draft.update_probe_enabled='${update_probe_enabled}' \
  policy.generation.vllm_kwargs.speculative_config.model='${DRAFTER_SNAPSHOT}' \
  logger.log_dir='${scratch_root}/logs' \
  logger.wandb_enabled=false logger.tensorboard_enabled=false \
  2>&1 | tee '${train_log}'
"

grep -Fq "Step 6" "${train_log}"
grep -Fq "Capturing CUDA graphs (PIECEWISE)" "${train_log}"
grep -Fq "Graph capturing finished" "${train_log}"
copy_profiles
python3 "${contract}" --validate-result-dir "${FINAL_DIR}"
echo "profile_gate=green"
