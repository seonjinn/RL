#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --account=nemotron_sw_post
#SBATCH --partition=batch_long
#SBATCH --job-name=qwen8-dflash-resume

set -euo pipefail

: "${REMOTE_REPO:?Set REMOTE_REPO}"
: "${EXPECTED_HEAD:?Set EXPECTED_HEAD}"
: "${RUN_DIR:?Set RUN_DIR}"
: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR}"
: "${GATE_MANIFEST:?Set GATE_MANIFEST}"
: "${EXPECTED_PREVIOUS_STEP:?Set EXPECTED_PREVIOUS_STEP}"
: "${TARGET_TOTAL_STEPS:?Set TARGET_TOTAL_STEPS}"
: "${CONTAINER:?Set CONTAINER}"
: "${TARGET_SNAPSHOT:?Set TARGET_SNAPSHOT}"
: "${DRAFTER_SNAPSHOT:?Set DRAFTER_SNAPSHOT}"
: "${WANDB_API_KEY:?WANDB_API_KEY is required}"

expected_container_sha256=6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44
dflash_k="${DFLASH_K:?Set DFLASH_K to 3, 5, 7, or 9}"
case "${dflash_k}" in
  3|5|7|9) ;;
  *) echo "DFLASH_K must be 3, 5, 7, or 9; got ${dflash_k}." >&2; exit 1 ;;
esac

test "$(git -C "${REMOTE_REPO}" rev-parse HEAD)" = "${EXPECTED_HEAD}"
test -z "$(git -C "${REMOTE_REPO}" status --porcelain --untracked-files=no)"
if git -C "${REMOTE_REPO}" submodule status --recursive | grep -Eq '^[+-U]'; then
  echo "Submodules are missing or do not match the gitlinks." >&2
  exit 1
fi
test -f "${CONTAINER}"
grep -Fqx "sha256=${expected_container_sha256}" "${CONTAINER}.metadata.txt"
test "$(basename "${TARGET_SNAPSHOT}")" = b968826d9c46dd6066d109eabc6255188de91218
test "$(basename "${DRAFTER_SNAPSHOT}")" = 9b41424b7109f9c5413454f481b09a82b85333f4

contract="${REMOTE_REPO}/research/fixed_drafter_qwen3_8b_dflash/resume_contract.py"
python3 "${contract}" \
  --previous-step "${EXPECTED_PREVIOUS_STEP}" \
  --target-step "${TARGET_TOTAL_STEPS}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --expected-step "${EXPECTED_PREVIOUS_STEP}" \
  --gate-manifest "${GATE_MANIFEST}" \
  --dflash-k "${dflash_k}" \
  --git-sha "${EXPECTED_HEAD}" \
  --target-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --drafter-revision 9b41424b7109f9c5413454f481b09a82b85333f4 \
  --container-sha256 "${expected_container_sha256}" \
  --validate-gate-manifest
wandb_run_id="$(python3 "${contract}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --expected-step "${EXPECTED_PREVIOUS_STEP}" \
  --gate-manifest "${GATE_MANIFEST}" \
  --dflash-k "${dflash_k}" \
  --git-sha "${EXPECTED_HEAD}" \
  --target-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --drafter-revision 9b41424b7109f9c5413454f481b09a82b85333f4 \
  --container-sha256 "${expected_container_sha256}" \
  --print-manifest-wandb-run-id)"

mkdir -p "${RUN_DIR}/logs"
if [[ "${TARGET_TOTAL_STEPS}" == 1000 ]]; then
  val_at_end=true
else
  val_at_end=false
fi
config="${REMOTE_REPO}/research/fixed_drafter_qwen3_8b_dflash/config_k${dflash_k}.yaml"

srun --ntasks=1 \
  --container-image="${CONTAINER}" \
  --container-mounts="/lustre:/lustre" \
  --no-container-mount-home \
  bash -lc "set -euo pipefail
export PYTHONPATH='${REMOTE_REPO}'
export HF_HOME='/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home'
export TMPDIR='/tmp/nrl-${SLURM_JOB_ID}'
export RAY_TMPDIR='/tmp/nrl-${SLURM_JOB_ID}'
mkdir -p '/tmp/nrl-${SLURM_JOB_ID}'
export TOKENIZERS_PARALLELISM=false
cd '${REMOTE_REPO}'
/opt/nemo_rl_venv/bin/python '${contract}' \
  --checkpoint-dir '${CHECKPOINT_DIR}' \
  --expected-step '${EXPECTED_PREVIOUS_STEP}' \
  --verify-nemo-resume-paths
UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv uv run --frozen --no-sync \
  examples/run_grpo.py --config '${config}' \
  grpo.max_num_steps='${TARGET_TOTAL_STEPS}' \
  grpo.val_at_end='${val_at_end}' \
  checkpointing.save_period=100 \
  checkpointing.checkpoint_dir='${CHECKPOINT_DIR}' \
  logger.log_dir='${RUN_DIR}/logs' \
  +logger.wandb.id='${wandb_run_id}' \
  +logger.wandb.resume=must \
  2>&1 | tee '${RUN_DIR}/train.log'
"

grep -Fq "Capturing CUDA graphs (PIECEWISE)" "${RUN_DIR}/train.log"
grep -Fq "Graph capturing finished" "${RUN_DIR}/train.log"

python3 "${contract}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --expected-step "${TARGET_TOTAL_STEPS}"
