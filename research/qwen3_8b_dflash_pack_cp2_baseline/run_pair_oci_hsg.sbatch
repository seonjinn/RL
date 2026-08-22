#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=04:30:00
#SBATCH --partition=batch_long
#SBATCH --qos=normal
#SBATCH --exclusive
#SBATCH --job-name=q8-pack-cp2-pair
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -Eeuo pipefail

: "${REMOTE_REPO:?Set exact /home checkout}"
: "${EXPECTED_HEAD:?Set signed harness SHA}"
: "${FINAL_DIR:?Set replicate-specific /lustre directory}"
: "${CONTAINER:?Set immutable container}"
: "${TARGET_SNAPSHOT:?Set exact target snapshot}"
: "${DRAFTER_SNAPSHOT:?Set exact DFlash snapshot}"
: "${REPLICATE:?Set replicate 1, 2, or 3}"
: "${FIRST_ARM:?Set fixed or online}"
: "${FIXED_WANDB_RUN_ID:?Set unique fixed W&B ID}"
: "${ONLINE_WANDB_RUN_ID:?Set unique online W&B ID}"
: "${WANDB_PROJECT:?Set unique W&B project}"
: "${WANDB_API_KEY:?W&B cluster secret is required}"

readonly product_head=443e7243ae2a235b6dcd8f4918fea86e693630a9
readonly container_sha=6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44
readonly dataset_revision=65877096c24ffa7abc4e4fa5edb95cf3413a5674
readonly source_parquet_sha=534375d6bb8630d22ab46a56e11f2ffec1d288d8f7d04099bc82d68948705941
readonly target_revision=b968826d9c46dd6066d109eabc6255188de91218
readonly drafter_revision=9b41424b7109f9c5413454f481b09a82b85333f4
readonly experiment_path=research/qwen3_8b_dflash_pack_cp2_baseline
readonly experiment="${REMOTE_REPO}/${experiment_path}"
readonly dataset_cache="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/datasets/BytedTsinghua-SIA___dapo-math-17k/default/0.0.0/${dataset_revision}"
readonly source_parquet="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/datasets--BytedTsinghua-SIA--DAPO-Math-17k/blobs/${source_parquet_sha}"
readonly scratch_root="/raid/scratch/${SLURM_JOB_ID}"
readonly ray_root="${scratch_root}/r"
readonly outer_log="${FINAL_DIR}/outer.log"
readonly reference_manifest="${FINAL_DIR}/dataset-manifest.json"

[[ "${REMOTE_REPO}" == /home/* ]]
[[ "${FINAL_DIR}" == /lustre/* ]]
[[ "${REPLICATE}" =~ ^[123]$ ]]
[[ "${FIRST_ARM}" == fixed || "${FIRST_ARM}" == online ]]
test "${FIXED_WANDB_RUN_ID}" != "${ONLINE_WANDB_RUN_ID}"
test "$(basename "${TARGET_SNAPSHOT}")" = "${target_revision}"
test "$(basename "${DRAFTER_SNAPSHOT}")" = "${drafter_revision}"
test -d "${dataset_cache}"
test -f "${source_parquet}"
mkdir -p "${scratch_root}" "${ray_root}" "${FINAL_DIR}"
exec > >(tee -a "${outer_log}") 2>&1

phase=bootstrap
archive() {
  local status=$?
  trap - EXIT
  set +e
  for arm in fixed online; do
    local train_log="${scratch_root}/${arm}/train.log"
    test -f "${train_log}" && mkdir -p "${FINAL_DIR}/${arm}" && \
      tail -n 20000 "${train_log}" > "${FINAL_DIR}/${arm}/train.log"
  done
  printf 'job_id=%s\nexit_code=%s\nproduct_source_sha=%s\nharness_sha=%s\nreplicate=%s\nfirst_arm=%s\nterminal_phase=%s\n' \
    "${SLURM_JOB_ID}" "${status}" "${product_head}" "${EXPECTED_HEAD}" \
    "${REPLICATE}" "${FIRST_ARM}" "${phase}" > "${FINAL_DIR}/result.txt"
  exit "${status}"
}
trap archive EXIT

phase=source_integrity
python3 "${experiment}/source_guard.py" --checkout "${REMOTE_REPO}" \
  --harness-head "${EXPECTED_HEAD}" --proof "${FINAL_DIR}/source-proof.json"
grep -Fq '"status": "passed"' "${FINAL_DIR}/source-proof.json"

phase=artifact_integrity
grep -Fqx "sha256=${container_sha}" "${CONTAINER}.metadata.txt"
test "$(sha256sum "${CONTAINER}" | cut -d' ' -f1)" = "${container_sha}"
test "$(sha256sum "${source_parquet}" | cut -d' ' -f1)" = "${source_parquet_sha}"

phase=dataset_manifest
python3 "${experiment}/dataset_manifest.py" build \
  --cache-root "${dataset_cache}" --source-parquet "${source_parquet}" \
  --source-parquet-sha256 "${source_parquet_sha}" --output "${reference_manifest}"

phase=resolved_config_parity
srun --ntasks=1 --kill-on-bad-exit=1 --wait=0 \
  --container-image="${CONTAINER}" \
  --container-mounts="/home:/home,/raid:/raid,/lustre:/lustre" \
  --no-container-mount-home bash -lc "set -Eeuo pipefail
export PYTHONPATH='${REMOTE_REPO}'
cd '${REMOTE_REPO}'
/opt/nemo_rl_venv/bin/python '${experiment}/parity.py' \
  --config '${experiment}/config.yaml' \
  --target-snapshot '${TARGET_SNAPSHOT}' --drafter-snapshot '${DRAFTER_SNAPSHOT}' \
  --expected-head '${EXPECTED_HEAD}' --proof '${FINAL_DIR}/resolved-parity.json'
"
grep -Fq '"status": "passed"' "${FINAL_DIR}/resolved-parity.json"

run_arm() {
  local arm=$1
  local wandb_id=$2
  local arm_scratch="${scratch_root}/${arm}"
  local arm_ray="${ray_root}/${arm}"
  local arm_final="${FINAL_DIR}/${arm}"
  local arm_manifest="${arm_final}/dataset-manifest.json"
  local train_log="${arm_scratch}/train.log"
  mkdir -p "${arm_scratch}" "${arm_ray}" "${arm_final}"
  phase="manifest_${arm}"
  python3 "${experiment}/dataset_manifest.py" build \
    --cache-root "${dataset_cache}" --source-parquet "${source_parquet}" \
    --source-parquet-sha256 "${source_parquet_sha}" --output "${arm_manifest}"
  python3 "${experiment}/dataset_manifest.py" verify-pair \
    --reference "${reference_manifest}" --candidate "${arm_manifest}"
  phase="run_${arm}"
  srun --ntasks=1 --kill-on-bad-exit=1 --wait=0 \
    --container-image="${CONTAINER}" \
    --container-mounts="/home:/home,/raid:/raid,/lustre:/lustre" \
    --no-container-mount-home bash -lc "set -Eeuo pipefail
export PYTHONPATH='${REMOTE_REPO}'
export HF_HOME='/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home'
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TMPDIR='${arm_scratch}'
export RAY_TMPDIR='${arm_ray}'
export TOKENIZERS_PARALLELISM=false
export WANDB__DISABLE_STATS=true
cd '${REMOTE_REPO}'
mapfile -t runtime_overrides < <(
  /opt/nemo_rl_venv/bin/python '${experiment}/runtime_contract.py' \
    --arm '${arm}' --target-snapshot '${TARGET_SNAPSHOT}' \
    --drafter-snapshot '${DRAFTER_SNAPSHOT}' --scratch-root '${arm_scratch}' \
    --wandb-run-id '${wandb_id}' --expected-head '${EXPECTED_HEAD}' \
    --wandb-project '${WANDB_PROJECT}'
)
test \"\${#runtime_overrides[@]}\" -gt 0
UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv uv run --frozen --no-sync \
  examples/run_grpo.py --config '${experiment}/config.yaml' \
  \"\${runtime_overrides[@]}\" 2>&1 | tee '${train_log}' >/dev/null
"
  grep -Fq "Step 30" "${train_log}"
  grep -Fq "Capturing CUDA graphs (PIECEWISE)" "${train_log}"
  grep -Fq "Graph capturing finished" "${train_log}"
  if [[ "${arm}" == online ]]; then
    grep -Fq "Draft Loss:" "${train_log}"
  else
    test "$(grep -c 'draft_update_probe=complete' "${train_log}" || true)" -eq 0
    test "$(grep -c 'draft_refit_finalize=complete' "${train_log}" || true)" -eq 0
  fi
  printf '{"arm":"%s","replicate":%s,"wandb_run_id":"%s","cache_hash_bound":true}\n' \
    "${arm}" "${REPLICATE}" "${wandb_id}" > "${arm_final}/evidence.json"
  echo "WANDB_URL=https://wandb.ai/nvidia/${WANDB_PROJECT}/runs/${wandb_id}"
}

if [[ "${FIRST_ARM}" == fixed ]]; then second_arm=online; else second_arm=fixed; fi
for arm in "${FIRST_ARM}" "${second_arm}"; do
  if [[ "${arm}" == fixed ]]; then
    run_arm fixed "${FIXED_WANDB_RUN_ID}"
  else
    run_arm online "${ONLINE_WANDB_RUN_ID}"
  fi
done

phase=complete
