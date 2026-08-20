#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --account=nemotron_n3_post
#SBATCH --partition=batch_long
#SBATCH --job-name=qwen8-dflash-drift

set -euo pipefail

: "${REMOTE_REPO:?Set REMOTE_REPO}"
: "${EXPECTED_HEAD:?Set EXPECTED_HEAD}"
: "${STAGE_STEPS:?Set STAGE_STEPS to 1, 10, or 100}"
: "${RUN_DIR:?Set RUN_DIR}"
: "${CONTAINER:?Set CONTAINER}"
: "${TARGET_SNAPSHOT:?Set TARGET_SNAPSHOT}"
: "${DRAFTER_SNAPSHOT:?Set DRAFTER_SNAPSHOT}"

expected_container_sha256=6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44
expected_drafter_config_sha256=9834d608c9ca53d5548b415471ae9e8ebc9aab6cedfc2a7af95b6bd097373102
dflash_k="${DFLASH_K:-15}"
case "${dflash_k}" in
  3|5)
    config="${REMOTE_REPO}/research/fixed_drafter_qwen3_8b_dflash/config_k${dflash_k}.yaml"
    if [[ -z "${WANDB_API_KEY:-}" ]]; then
      echo "WANDB_API_KEY is required for the K-sweep arms." >&2
      exit 1
    fi
    ;;
  15)
    config="${REMOTE_REPO}/research/fixed_drafter_qwen3_8b_dflash/config.yaml"
    ;;
  *)
    echo "DFLASH_K must be 3, 5, or baseline 15; got ${dflash_k}." >&2
    exit 1
    ;;
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
test -f "${TARGET_SNAPSHOT}/config.json"
test -f "${TARGET_SNAPSHOT}/tokenizer.json"
test -f "${DRAFTER_SNAPSHOT}/model.safetensors"
test "$(sha256sum "${DRAFTER_SNAPSHOT}/config.json" | awk '{print $1}')" = "${expected_drafter_config_sha256}"
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
/opt/nemo_rl_venv/bin/python '${REMOTE_REPO}/research/fixed_drafter_qwen3_8b_dflash/contract.py' \
  --config '${config}' --steps '${STAGE_STEPS}' --k '${dflash_k}' | tee '${RUN_DIR}/contract.json'
UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv uv run --frozen --no-sync \
  examples/run_grpo.py --config '${config}' \
  grpo.max_num_steps='${STAGE_STEPS}' \
  checkpointing.save_period='${STAGE_STEPS}' \
  checkpointing.checkpoint_dir='${RUN_DIR}/checkpoints' \
  logger.log_dir='${RUN_DIR}/logs' \
  2>&1 | tee '${RUN_DIR}/train.log'
"
