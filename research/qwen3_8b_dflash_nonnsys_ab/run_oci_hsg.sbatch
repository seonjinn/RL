#!/bin/bash
# shellcheck disable=SC1083
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
: "${PRODUCT_HEAD:?Set exact product source SHA}"
: "${SOURCE_ARM:?Set base or optimized}"
: "${PARITY_PROOF:?Set source parity proof}"

readonly product_head="${PRODUCT_HEAD}"
readonly container_sha=6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44
readonly target_revision=b968826d9c46dd6066d109eabc6255188de91218
readonly drafter_revision=9b41424b7109f9c5413454f481b09a82b85333f4
readonly experiment="${REMOTE_REPO}/research/qwen3_8b_dflash_nonnsys_ab"
readonly contract="${experiment}/runtime_contract.py"
readonly source_parity="${experiment}/source_parity.py"
readonly arm=online
test "${ARM}" = "${arm}"
readonly scratch_root="/raid/scratch/dflash-ab/${SLURM_JOB_ID}"
readonly ray_root="/raid/scratch/dflash-ab-ray/${SLURM_JOB_ID}"
readonly train_log="${scratch_root}/train.log"
readonly outer_log="${FINAL_DIR}/outer.log"

[[ "${REMOTE_REPO}" == /home/* ]]
[[ "${FINAL_DIR}" == /lustre/* ]]
[[ "${WANDB_PROJECT}" == sna-nemo-rl-online-drafter ]]
mkdir -p "${scratch_root}" "${ray_root}" "${FINAL_DIR}"
phase=bootstrap
archive() {
  local status=$?
  trap - EXIT
  set +e
  echo "phase=${phase} rc=${status}"
  test -f "${train_log}" && tail -n 10000 "${train_log}" > "${FINAL_DIR}/train.log"
  {
    echo "job_id=${SLURM_JOB_ID}"
    echo "exit_code=${status}"
    echo "arm=${ARM}"
    echo "git_sha=${EXPECTED_HEAD}"
    echo "product_source_sha=${product_head}"
    echo "source_arm=${SOURCE_ARM}"
    echo "wandb_run_id=${WANDB_RUN_ID}"
    echo "container_sha256=${container_sha}"
    echo "terminal_phase=${phase}"
    echo "finished_at=$(date -Is)"
  } > "${FINAL_DIR}/result.txt"
  exit "${status}"
}
trap archive EXIT
exec > >(tee -a "${outer_log}") 2>&1

begin_phase() {
  phase=$1
  echo "phase=${phase} status=begin"
}

finish_phase() {
  echo "phase=${phase} rc=0"
}

begin_phase source_integrity
test "$(git -C "${REMOTE_REPO}" rev-parse HEAD)" = "${EXPECTED_HEAD}"
test "$(git -C "${REMOTE_REPO}" merge-base "${product_head}" "${EXPECTED_HEAD}")" = \
  "${product_head}"
unexpected_changes="$(
  git -C "${REMOTE_REPO}" diff --name-only \
    "${product_head}" "${EXPECTED_HEAD}" -- | \
    grep -v '^research/qwen3_8b_dflash_nonnsys_ab/' || true
)"
test -z "${unexpected_changes}"
git -C "${REMOTE_REPO}" diff-index --quiet --ignore-submodules=all HEAD --
test -z "$(git -C "${REMOTE_REPO}" ls-files --others --exclude-standard)"
git -C "${REMOTE_REPO}" submodule foreach --quiet --recursive \
  'git diff-index --quiet --ignore-submodules=all HEAD --'
finish_phase

begin_phase source_parity
python3 "${source_parity}" validate-proof \
  --proof "${PARITY_PROOF}" \
  --source-arm "${SOURCE_ARM}" \
  --product-head "${product_head}" \
  --harness-head "${EXPECTED_HEAD}"
finish_phase

begin_phase artifact_integrity
grep -Fqx "sha256=${container_sha}" "${CONTAINER}.metadata.txt"
test "$(basename "${TARGET_SNAPSHOT}")" = "${target_revision}"
test "$(basename "${DRAFTER_SNAPSHOT}")" = "${drafter_revision}"
finish_phase

begin_phase arm_contract
read -r config_path _ < <(
  python3 "${contract}" --arm "${arm}"
)
readonly config="${REMOTE_REPO}/${config_path}"
test -f "${config}"
finish_phase

begin_phase srun_launch
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
mapfile -t runtime_overrides < <(
  /opt/nemo_rl_venv/bin/python '${experiment}/resolved_parity.py' emit-overrides \
    --arm '${arm}' \
    --target-snapshot '${TARGET_SNAPSHOT}' \
    --drafter-snapshot '${DRAFTER_SNAPSHOT}' \
    --scratch-root '${scratch_root}' \
    --wandb-run-id '${WANDB_RUN_ID}' \
    --wandb-project '${WANDB_PROJECT}' \
    --expected-head '${EXPECTED_HEAD}' \
    --product-source-sha '${product_head}' \
    --source-arm '${SOURCE_ARM}'
)
test "\${#runtime_overrides[@]}" -gt 0
UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv uv run --frozen --no-sync \
  examples/run_grpo.py --config '${config}' \
  "\${runtime_overrides[@]}" \
  2>&1 | tee '${train_log}' >/dev/null
"
finish_phase

begin_phase science_validation
grep -Fq "Step 50" "${train_log}"
grep -Fq "Capturing CUDA graphs (PIECEWISE)" "${train_log}"
grep -Fq "Graph capturing finished" "${train_log}"
finish_phase
echo "WANDB_URL=https://wandb.ai/nvidia/${WANDB_PROJECT}/runs/${WANDB_RUN_ID}"
