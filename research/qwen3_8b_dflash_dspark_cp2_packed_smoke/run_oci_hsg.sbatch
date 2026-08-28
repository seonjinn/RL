#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --exclusive
#SBATCH --job-name=q8-cp2-pack-smoke
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -Eeuo pipefail

: "${ARM:?Set dflash or dspark}"
: "${REMOTE_REPO:?Set exact /home checkout}"
: "${EXPECTED_HEAD:?Set exact harness commit}"
: "${FINAL_DIR:?Set provider-specific /lustre result directory}"
: "${CONTAINER:?Set immutable container path}"
: "${CONTAINER_SHA256:?Set immutable container SHA256}"
: "${TARGET_SNAPSHOT:?Set exact Qwen3-8B target snapshot}"
: "${DRAFTER_SNAPSHOT:?Set exact provider snapshot}"
: "${WANDB_API_KEY:?W&B cluster secret is required}"
: "${WANDB_PROJECT:?Set W&B project}"
: "${WANDB_RUN_ID:?Set unique W&B run ID}"

readonly product_head=26603aedd28fae9db852e4b84d85d6cfcd7729e8
readonly target_revision=b968826d9c46dd6066d109eabc6255188de91218
readonly dflash_revision=9b41424b7109f9c5413454f481b09a82b85333f4
readonly dspark_revision=03326e5043815da1f81b109078b2889737c26017
readonly experiment_path=research/qwen3_8b_dflash_dspark_cp2_packed_smoke
readonly experiment="${REMOTE_REPO}/${experiment_path}"
readonly scratch_root="/raid/scratch/${SLURM_JOB_ID}"
readonly ray_root="${scratch_root}/ray"
readonly train_log="${scratch_root}/train.log"
readonly final_log="${FINAL_DIR}/train.log"
readonly source_proof="${FINAL_DIR}/source-proof.txt"

[[ "${ARM}" == dflash || "${ARM}" == dspark ]]
[[ "${REMOTE_REPO}" == /home/* ]]
[[ "${FINAL_DIR}" == /lustre/* ]]
test "$(basename "${TARGET_SNAPSHOT}")" = "${target_revision}"
if [[ "${ARM}" == dflash ]]; then
  test "$(basename "${DRAFTER_SNAPSHOT}")" = "${dflash_revision}"
else
  test "$(basename "${DRAFTER_SNAPSHOT}")" = "${dspark_revision}"
fi

mkdir -p "${scratch_root}" "${ray_root}" "${FINAL_DIR}"
exec > >(tee -a "${FINAL_DIR}/outer.log") 2>&1

phase=bootstrap
archive() {
  local status=$?
  trap - EXIT
  set +e
  test -f "${train_log}" && tail -n 20000 "${train_log}" > "${final_log}"
  printf 'job_id=%s\nexit_code=%s\narm=%s\nproduct_head=%s\nharness_head=%s\nterminal_phase=%s\nwandb_run_id=%s\n' \
    "${SLURM_JOB_ID}" "${status}" "${ARM}" "${product_head}" "${EXPECTED_HEAD}" \
    "${phase}" "${WANDB_RUN_ID}" > "${FINAL_DIR}/result.txt"
  exit "${status}"
}
trap archive EXIT

phase=source_integrity
test "$(git -C "${REMOTE_REPO}" rev-parse HEAD)" = "${EXPECTED_HEAD}"
test "$(git -C "${REMOTE_REPO}" merge-base "${product_head}" "${EXPECTED_HEAD}")" = "${product_head}"
test -z "$(git -C "${REMOTE_REPO}" status --porcelain --untracked-files=all)"
mapfile -t changed_paths < <(git -C "${REMOTE_REPO}" diff --name-only "${product_head}" "${EXPECTED_HEAD}")
test "${#changed_paths[@]}" -gt 0
for path in "${changed_paths[@]}"; do
  [[ "${path}" == "${experiment_path}/"* ]]
done
git -C "${REMOTE_REPO}" submodule status --recursive > "${source_proof}"
test -s "${source_proof}"
test "$(grep -Ec '^[-+U]' "${source_proof}" || true)" -eq 0

phase=artifact_integrity
test -f "${CONTAINER}"
test "$(sha256sum "${CONTAINER}" | cut -d' ' -f1)" = "${CONTAINER_SHA256}"
test -d "${TARGET_SNAPSHOT}"
test -d "${DRAFTER_SNAPSHOT}"

phase=run
config_path=$(python3 "${experiment}/contract.py" \
  --arm "${ARM}" --target-snapshot "${TARGET_SNAPSHOT}" \
  --drafter-snapshot "${DRAFTER_SNAPSHOT}" --scratch-root "${scratch_root}" \
  --wandb-run-id "${WANDB_RUN_ID}" --expected-head "${EXPECTED_HEAD}" \
  --wandb-project "${WANDB_PROJECT}" --print-config)
test -f "${config_path}"
srun --ntasks=1 --kill-on-bad-exit=1 --wait=0 \
  --container-image="${CONTAINER}" \
  --container-mounts="/home:/home,/raid:/raid,/lustre:/lustre" \
  --no-container-mount-home bash -lc "set -Eeuo pipefail
export PYTHONPATH='${REMOTE_REPO}'
export HF_HOME='/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home'
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TMPDIR='${scratch_root}'
export RAY_TMPDIR='${ray_root}'
export TOKENIZERS_PARALLELISM=false
export WANDB__DISABLE_STATS=true
cd '${REMOTE_REPO}'
mapfile -t runtime_overrides < <(
  /opt/nemo_rl_venv/bin/python '${experiment}/contract.py' \
    --arm '${ARM}' --target-snapshot '${TARGET_SNAPSHOT}' \
    --drafter-snapshot '${DRAFTER_SNAPSHOT}' --scratch-root '${scratch_root}' \
    --wandb-run-id '${WANDB_RUN_ID}' --expected-head '${EXPECTED_HEAD}' \
    --wandb-project '${WANDB_PROJECT}'
)
test \"\${#runtime_overrides[@]}\" -gt 0
UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv uv run --frozen --no-sync \
  examples/run_grpo.py --config '${config_path}' \
  \"\${runtime_overrides[@]}\" 2>&1 | tee '${train_log}' >/dev/null
"

phase=evidence
grep -Fq "Step 2" "${train_log}"
grep -Fq "Draft Loss:" "${train_log}"
grep -Fq "draft_update_probe=complete" "${train_log}"
grep -Fq "draft_refit_manifest=draft_count=" "${train_log}"
grep -Fq "draft_post_update_refit=complete step=1" "${train_log}"
grep -Fq "Capturing CUDA graphs (PIECEWISE)" "${train_log}"
grep -Fq "Graph capturing finished" "${train_log}"
printf '{"arm":"%s","wandb_run_id":"%s","cp":2,"sequence_packing":true,"status":"passed"}\n' \
  "${ARM}" "${WANDB_RUN_ID}" > "${FINAL_DIR}/evidence.json"

phase=complete
