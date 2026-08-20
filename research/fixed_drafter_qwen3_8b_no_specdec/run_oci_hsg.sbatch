#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --account=nemotron_n3_post
#SBATCH --partition=batch_long
#SBATCH --job-name=qwen8-no-specdec-k0

set -euo pipefail

: "${REMOTE_REPO:?Set REMOTE_REPO}"
: "${EXPECTED_HEAD:?Set EXPECTED_HEAD}"
: "${RUN_DIR:?Set RUN_DIR}"
: "${CONTAINER:?Set CONTAINER}"
: "${TARGET_SNAPSHOT:?Set TARGET_SNAPSHOT}"
: "${DFLASH_RESOLVED_CONFIG:?Set DFLASH_RESOLVED_CONFIG}"
: "${WANDB_API_KEY:?WANDB_API_KEY must be present}"

expected_container_sha256=6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44
config="${REMOTE_REPO}/research/fixed_drafter_qwen3_8b_no_specdec/config.yaml"
parity="${REMOTE_REPO}/research/fixed_drafter_qwen3_8b_no_specdec/parity.py"

test "$(git -C "${REMOTE_REPO}" rev-parse HEAD)" = "${EXPECTED_HEAD}"
test -z "$(git -C "${REMOTE_REPO}" status --porcelain)"
if git -C "${REMOTE_REPO}" submodule status --recursive | grep -Eq '^[+-U]'; then
  echo "Submodules are missing or do not match the gitlinks." >&2
  exit 1
fi
test -f "${CONTAINER}"
grep -Fqx "sha256=${expected_container_sha256}" "${CONTAINER}.metadata.txt"
test "$(basename "${TARGET_SNAPSHOT}")" = b968826d9c46dd6066d109eabc6255188de91218
test -f "${TARGET_SNAPSHOT}/config.json"
test -f "${TARGET_SNAPSHOT}/tokenizer.json"
test -f "${DFLASH_RESOLVED_CONFIG}"
mkdir -p "${RUN_DIR}/checkpoints" "${RUN_DIR}/logs"

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
/opt/nemo_rl_venv/bin/python '${parity}' \
  --baseline '${config}' \
  --dflash '${DFLASH_RESOLVED_CONFIG}' | tee '${RUN_DIR}/parity.json'
UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv uv run --frozen --no-sync \
  examples/run_grpo.py --config '${config}' \
  checkpointing.checkpoint_dir='${RUN_DIR}/checkpoints' \
  logger.log_dir='${RUN_DIR}/logs' \
  2>&1 | tee '${RUN_DIR}/train.log'
"
