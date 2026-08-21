#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --partition=batch_long
#SBATCH --job-name=q8-dflash-ab-parity
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -Eeuo pipefail

: "${REMOTE_REPO:?Set exact /home parity checkout}"
: "${EXPECTED_HEAD:?Set signed harness SHA}"
: "${FINAL_DIR:?Set /lustre parity result directory}"
: "${CONTAINER:?Set immutable container}"
: "${TARGET_SNAPSHOT:?Set exact target snapshot}"
: "${DRAFTER_SNAPSHOT:?Set exact drafter snapshot}"

readonly optimized_source_sha=79e80af96a13522e6049658663a8c40ab21e8314
readonly container_sha=6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44
readonly target_revision=b968826d9c46dd6066d109eabc6255188de91218
readonly drafter_revision=9b41424b7109f9c5413454f481b09a82b85333f4
readonly experiment="${REMOTE_REPO}/research/qwen3_8b_dflash_nonnsys_ab"
readonly scratch_root="/raid/scratch/dflash-ab-parity/${SLURM_JOB_ID}"
readonly proof="${FINAL_DIR}/resolved-parity.json"
readonly outer_log="${FINAL_DIR}/outer.log"

[[ "${REMOTE_REPO}" == /home/* ]]
[[ "${FINAL_DIR}" == /lustre/* ]]
mkdir -p "${scratch_root}" "${FINAL_DIR}"
phase=bootstrap
archive() {
  local status=$?
  trap - EXIT
  set +e
  echo "phase=${phase} rc=${status}"
  {
    echo "job_id=${SLURM_JOB_ID}"
    echo "exit_code=${status}"
    echo "git_sha=${EXPECTED_HEAD}"
    echo "optimized_source_sha=${optimized_source_sha}"
    echo "container_sha256=${container_sha}"
    echo "terminal_phase=${phase}"
    echo "finished_at=$(date -Is)"
  } > "${FINAL_DIR}/result.txt"
  exit "${status}"
}
trap archive EXIT
exec > >(tee -a "${outer_log}") 2>&1

phase=source_integrity
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
# shellcheck disable=SC2016
git -C "${REMOTE_REPO}" submodule foreach --quiet --recursive \
  'git diff-index --quiet --ignore-submodules=all HEAD -- && test -z "$(git ls-files --others --exclude-standard)"'

phase=artifact_integrity
grep -Fqx "sha256=${container_sha}" "${CONTAINER}.metadata.txt"
test "$(basename "${TARGET_SNAPSHOT}")" = "${target_revision}"
test "$(basename "${DRAFTER_SNAPSHOT}")" = "${drafter_revision}"

phase=container_resolved_parity
srun --ntasks=1 --container-image="${CONTAINER}" \
  --container-mounts="/home:/home,/raid:/raid,/lustre:/lustre" \
  --no-container-mount-home bash -lc "set -Eeuo pipefail
export PYTHONPATH='${REMOTE_REPO}'
export PYTHONDONTWRITEBYTECODE=1
cd '${REMOTE_REPO}'
/opt/nemo_rl_venv/bin/python '${experiment}/resolved_parity.py' check \
  --online-config '${REMOTE_REPO}/research/qwen3_8b_dflash_online_cp1/config.yaml' \
  --fixed-config '${REMOTE_REPO}/research/qwen3_8b_dflash_fixed_dense_control/config.yaml' \
  --proof '${proof}' \
  --target-snapshot '${TARGET_SNAPSHOT}' \
  --drafter-snapshot '${DRAFTER_SNAPSHOT}' \
  --scratch-root '/raid/scratch/dflash-ab/PARITY' \
  --wandb-run-id PARITY \
  --wandb-project sna-nemo-rl-online-drafter \
  --expected-head '${EXPECTED_HEAD}' \
  --container-sha256 '${container_sha}'
"
grep -Fq '"status": "passed"' "${proof}"
phase=complete
