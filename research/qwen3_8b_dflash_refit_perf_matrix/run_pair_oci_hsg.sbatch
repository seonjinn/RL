#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=04:30:00
#SBATCH --partition=batch_long
#SBATCH --job-name=q8-dflash-refit-pair
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -Eeuo pipefail

: "${REMOTE_REPO:?Set exact /home checkout}"
: "${EXPECTED_HEAD:?Set signed harness SHA}"
: "${FINAL_DIR:?Set pair-specific /lustre result directory}"
: "${CONTAINER:?Set immutable container}"
: "${TARGET_SNAPSHOT:?Set exact target snapshot}"
: "${DRAFTER_SNAPSHOT:?Set exact drafter snapshot}"
: "${PAIR_SHAPE:?Set one matrix shape}"
: "${FIRST_ARM:?Set fixed or online}"
: "${FIXED_WANDB_RUN_ID:?Set fresh fixed W&B run ID}"
: "${ONLINE_WANDB_RUN_ID:?Set fresh online W&B run ID}"
: "${WANDB_PROJECT:?Set W&B project}"
: "${WANDB_API_KEY:?W&B cluster secret is required}"

readonly product_head=4d8a54538d694f81f65bf2b431c5b5ed6a3017ca
readonly container_sha=6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44
readonly target_revision=b968826d9c46dd6066d109eabc6255188de91218
readonly drafter_revision=9b41424b7109f9c5413454f481b09a82b85333f4
readonly experiment_path=research/qwen3_8b_dflash_refit_perf_matrix
readonly experiment="${REMOTE_REPO}/${experiment_path}"
readonly scratch_root="/raid/scratch/dflash-refit-matrix/${SLURM_JOB_ID}"
readonly ray_root="/raid/scratch/dflash-refit-matrix-ray/${SLURM_JOB_ID}"
readonly outer_log="${FINAL_DIR}/outer.log"
readonly parity_proof="${FINAL_DIR}/resolved-parity.json"
readonly source_proof="${FINAL_DIR}/source-parity.json"

[[ "${REMOTE_REPO}" == /home/* ]]
[[ "${FINAL_DIR}" == /lustre/* ]]
[[ "${WANDB_PROJECT}" == sna-nemo-rl-online-drafter ]]
[[ "${PAIR_SHAPE}" =~ ^gbs(32_mbs1|64_mbs[12])$ ]]
[[ "${FIRST_ARM}" == fixed || "${FIRST_ARM}" == online ]]
test "${FIXED_WANDB_RUN_ID}" != "${ONLINE_WANDB_RUN_ID}"
mkdir -p "${scratch_root}" "${ray_root}" "${FINAL_DIR}"

phase=bootstrap
archive() {
  local status=$?
  trap - EXIT
  set +e
  for arm in fixed online; do
    local train_log="${scratch_root}/${arm}/train.log"
    test -f "${train_log}" && tail -n 20000 "${train_log}" > "${FINAL_DIR}/${arm}/train.log"
  done
  {
    echo "job_id=${SLURM_JOB_ID}"
    echo "exit_code=${status}"
    echo "git_sha=${EXPECTED_HEAD}"
    echo "product_source_sha=${product_head}"
    echo "pair_shape=${PAIR_SHAPE}"
    echo "first_arm=${FIRST_ARM}"
    echo "container_sha256=${container_sha}"
    echo "node=${SLURMD_NODENAME:-unknown}"
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
python3 "${experiment}/source_guard.py" \
  --checkout "${REMOTE_REPO}" \
  --product-head "${product_head}" \
  --harness-head "${EXPECTED_HEAD}" \
  --proof "${source_proof}"
grep -Fq '"status": "passed"' "${source_proof}"
finish_phase

begin_phase artifact_integrity
grep -Fqx "sha256=${container_sha}" "${CONTAINER}.metadata.txt"
test "$(basename "${TARGET_SNAPSHOT}")" = "${target_revision}"
test "$(basename "${DRAFTER_SNAPSHOT}")" = "${drafter_revision}"
finish_phase

begin_phase resolved_parity
srun --ntasks=1 --container-image="${CONTAINER}" \
  --container-mounts="/home:/home,/raid:/raid,/lustre:/lustre" \
  --no-container-mount-home bash -lc "set -Eeuo pipefail
export PYTHONPATH='${REMOTE_REPO}'
export PYTHONDONTWRITEBYTECODE=1
cd '${REMOTE_REPO}'
/opt/nemo_rl_venv/bin/python '${experiment}/parity.py' check \
  --shape '${PAIR_SHAPE}' \
  --online-config '${REMOTE_REPO}/research/qwen3_8b_dflash_online_cp1/config.yaml' \
  --fixed-config '${REMOTE_REPO}/research/qwen3_8b_dflash_fixed_dense_control/config.yaml' \
  --target-snapshot '${TARGET_SNAPSHOT}' \
  --drafter-snapshot '${DRAFTER_SNAPSHOT}' \
  --expected-head '${EXPECTED_HEAD}' \
  --container-sha256 '${container_sha}' \
  --proof '${parity_proof}'
"
grep -Fq '"status": "passed"' "${parity_proof}"
finish_phase

run_arm() {
  local arm=$1
  local wandb_id=$2
  local cell="${PAIR_SHAPE}_${arm}"
  local arm_scratch="${scratch_root}/${arm}"
  local arm_ray="${ray_root}/${arm}"
  local arm_final="${FINAL_DIR}/${arm}"
  local train_log="${arm_scratch}/train.log"
  mkdir -p "${arm_scratch}" "${arm_ray}" "${arm_final}"
  phase="run_${cell}"
  srun --ntasks=1 --container-image="${CONTAINER}" \
    --container-mounts="/home:/home,/raid:/raid,/lustre:/lustre" \
    --no-container-mount-home bash -lc "set -Eeuo pipefail
export PYTHONPATH='${REMOTE_REPO}'
export HF_HOME='/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home'
export TMPDIR='${arm_scratch}'
export RAY_TMPDIR='${arm_ray}'
export TOKENIZERS_PARALLELISM=false
export WANDB__DISABLE_STATS=true
unset NRL_NSYS_WORKER_PATTERNS
unset NRL_NSYS_PROFILE_STEP_RANGE
unset NRL_NSYS_EXTRA_OPTIONS
cd '${REMOTE_REPO}'
config_path=\$(/opt/nemo_rl_venv/bin/python -c \"import sys; sys.path.insert(0, '${experiment}'); import runtime_contract; print(runtime_contract.resolve_cell('${cell}').config_path)\")
mapfile -t runtime_overrides < <(
  /opt/nemo_rl_venv/bin/python '${experiment}/runtime_contract.py' \
    --cell '${cell}' \
    --target-snapshot '${TARGET_SNAPSHOT}' \
    --drafter-snapshot '${DRAFTER_SNAPSHOT}' \
    --scratch-root '${arm_scratch}' \
    --wandb-run-id '${wandb_id}' \
    --wandb-project '${WANDB_PROJECT}' \
    --expected-head '${EXPECTED_HEAD}'
)
test \"\${#runtime_overrides[@]}\" -gt 0
UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv uv run --frozen --no-sync \
  examples/run_grpo.py --config '${REMOTE_REPO}/\${config_path}' \
  \"\${runtime_overrides[@]}\" \
  2>&1 | tee '${train_log}' >/dev/null
"
  grep -Fq "Step 50" "${train_log}"
  grep -Fq "Capturing CUDA graphs (PIECEWISE)" "${train_log}"
  grep -Fq "Graph capturing finished" "${train_log}"
  local update_count refit_count
  update_count=$(grep -c "draft_update_probe=complete" "${train_log}" || true)
  refit_count=$(grep -c "draft_refit_finalize=complete" "${train_log}" || true)
  if [[ "${arm}" == online ]]; then
    grep -Fq "Draft Loss:" "${train_log}"
  else
    test "${update_count}" -eq 0
    test "${refit_count}" -eq 0
  fi
  printf '{\n  "cell": "%s",\n  "draft_update_count": %d,\n  "draft_refit_count": %d,\n  "wandb_run_id": "%s"\n}\n' \
    "${cell}" "${update_count}" "${refit_count}" "${wandb_id}" \
    > "${arm_final}/evidence.json"
  echo "WANDB_URL=https://wandb.ai/nvidia/${WANDB_PROJECT}/runs/${wandb_id}"
}

if [[ "${FIRST_ARM}" == fixed ]]; then
  readonly second_arm=online
else
  readonly second_arm=fixed
fi

for arm in "${FIRST_ARM}" "${second_arm}"; do
  if [[ "${arm}" == fixed ]]; then
    run_arm fixed "${FIXED_WANDB_RUN_ID}"
  else
    run_arm online "${ONLINE_WANDB_RUN_ID}"
  fi
done

begin_phase write_manifest
cat > "${FINAL_DIR}/wandb-runs.json" <<EOF
{
  "runs": [
    {
      "cell": "${PAIR_SHAPE}_fixed",
      "run_path": "nvidia/${WANDB_PROJECT}/${FIXED_WANDB_RUN_ID}",
      "evidence": "${FINAL_DIR}/fixed/evidence.json"
    },
    {
      "cell": "${PAIR_SHAPE}_online",
      "run_path": "nvidia/${WANDB_PROJECT}/${ONLINE_WANDB_RUN_ID}",
      "evidence": "${FINAL_DIR}/online/evidence.json"
    }
  ]
}
EOF
finish_phase
phase=complete
