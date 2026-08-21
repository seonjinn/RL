#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --partition=batch_long
#SBATCH --job-name=q8-online-dspark-k7

set -euo pipefail

: "${REMOTE_REPO:?Set exact /home checkout}"
: "${EXPECTED_HEAD:?Set signed composition SHA}"
: "${FINAL_DIR:?Set /lustre result directory}"
: "${CONTAINER:?Set immutable container}"
: "${TARGET_SNAPSHOT:?Set exact target snapshot}"
: "${DRAFTER_SNAPSHOT:?Set exact drafter snapshot}"
: "${WANDB_RUN_ID:?Set fresh persistent W&B run ID}"
: "${WANDB_RESUME:?Set allow for smoke or must for resume}"
: "${STAGE_MIN_STEP:?Set cumulative milestone}"
: "${STAGE_DEADLINE:?Set checkpoint deadline}"
: "${STAGE_MODE:?Set smoke or resume}"
: "${WANDB_API_KEY:?W&B cluster secret is required}"

readonly container_sha=6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44
readonly target_revision=b968826d9c46dd6066d109eabc6255188de91218
readonly drafter_revision=03326e5043815da1f81b109078b2889737c26017
readonly openai_wheel='https://files.pythonhosted.org/packages/5b/e6/ca496976bacdd7f86d4fa69359c2d9b5a547d7acbf0ae5cac7bff107ff50/openai-2.25.0-py3-none-any.whl#sha256=7c7c01a0a4b69771a29913e28d06bc1e9cee8781b4d206bb56cb946d8d2fcb23'
readonly preflight_code='import importlib.metadata as m; assert m.version("openai") == "2.25.0"; assert m.version("vllm") == "0.25.1"; from openai.types.responses import NamespaceTool; from vllm import LLM; print("RUNTIME_PREFLIGHT=openai=2.25.0,vllm=0.25.1")'
readonly experiment="${REMOTE_REPO}/research/qwen3_8b_dspark_online_cp1"
readonly checkpoint_dir="${FINAL_DIR}/checkpoints"
readonly manifest="${FINAL_DIR}/gate-manifest.json"
readonly scratch_root="/raid/scratch/o/${SLURM_JOB_ID}"
readonly ray_root="/raid/scratch/r/${SLURM_JOB_ID}"
readonly train_log="${scratch_root}/train.log"
readonly evidence_log="${scratch_root}/evidence.log"

[[ "${REMOTE_REPO}" == /home/* ]]
[[ "${FINAL_DIR}" == /lustre/* ]]
[[ "${STAGE_MODE}" == smoke || "${STAGE_MODE}" == resume ]]
[[ "${WANDB_RESUME}" == allow || "${WANDB_RESUME}" == must ]]
test "$(git -C "${REMOTE_REPO}" rev-parse HEAD)" = "${EXPECTED_HEAD}"
git -C "${REMOTE_REPO}" diff-index --quiet --ignore-submodules=all HEAD --
test -z "$(git -C "${REMOTE_REPO}" ls-files --others --exclude-standard)"
git -C "${REMOTE_REPO}" submodule foreach --quiet --recursive \
  'git diff-index --quiet --ignore-submodules=all HEAD --'
grep -Fqx "sha256=${container_sha}" "${CONTAINER}.metadata.txt"
test "$(basename "${TARGET_SNAPSHOT}")" = "${target_revision}"
test "$(basename "${DRAFTER_SNAPSHOT}")" = "${drafter_revision}"

previous_step=0
if [[ "${STAGE_MODE}" == resume ]]; then
  previous_step="$(python3 "${experiment}/resume_contract.py" \
    --checkpoint-dir "${checkpoint_dir}" --print-latest-step)"
  python3 "${experiment}/resume_contract.py" --checkpoint-dir "${checkpoint_dir}" \
    --expected-step "${previous_step}" --manifest "${manifest}" \
    --git-sha "${EXPECTED_HEAD}" --target-revision "${target_revision}" \
    --drafter-revision "${drafter_revision}" --container-sha256 "${container_sha}" \
    --validate-manifest
else
  test ! -e "${manifest}"
  test ! -d "${checkpoint_dir}"
fi
if (( previous_step >= STAGE_MIN_STEP )); then exit 0; fi

mkdir -p "${scratch_root}" "${ray_root}" "${FINAL_DIR}/logs" "${checkpoint_dir}"
archive() {
  status=$?
  trap - EXIT
  set +e
  if test -f "${evidence_log}"; then
    tail -n 4000 "${evidence_log}" > \
      "${FINAL_DIR}/logs/job-${SLURM_JOB_ID}-to-${STAGE_MIN_STEP}.log"
  elif test -f "${train_log}"; then
    tail -n 4000 "${train_log}" > \
      "${FINAL_DIR}/logs/job-${SLURM_JOB_ID}-to-${STAGE_MIN_STEP}.log"
  fi
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
uv pip install --python /opt/nemo_rl_venv/bin/python --no-deps '${openai_wheel}'
/opt/nemo_rl_venv/bin/python -c $(printf '%q' "${preflight_code}")
UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv uv run --frozen --no-sync \
  examples/run_grpo.py --config '${experiment}/config.yaml' \
  grpo.max_num_steps=1000 grpo.val_period=1 checkpointing.save_period=100 \
  checkpointing.checkpoint_must_save_by='${STAGE_DEADLINE}' \
  checkpointing.checkpoint_dir='${checkpoint_dir}' \
  policy.model_name='${TARGET_SNAPSHOT}' policy.tokenizer.name='${TARGET_SNAPSHOT}' \
  policy.draft.model_name='${DRAFTER_SNAPSHOT}' \
  policy.generation.vllm_kwargs.speculative_config.model='${DRAFTER_SNAPSHOT}' \
  logger.log_dir='${scratch_root}/logs' \
  +logger.wandb.id='${WANDB_RUN_ID}' +logger.wandb.resume='${WANDB_RESUME}' \
  +logger.wandb.config.git_sha='${EXPECTED_HEAD}' \
  +logger.wandb.config.stage_min_step='${STAGE_MIN_STEP}' \
  2>&1 | tee '${train_log}'
cp '${train_log}' '${evidence_log}'
grep -RhaE 'draft_(update_probe=complete|refit_(manifest|load|finalize)=)' \
  '${ray_root}' >> '${evidence_log}' || true
"

grep -Fq "Capturing CUDA graphs (PIECEWISE)" "${train_log}"
grep -Fq "Graph capturing finished" "${train_log}"
current_step="$(python3 "${experiment}/resume_contract.py" \
  --checkpoint-dir "${checkpoint_dir}" --print-latest-step)"
python3 "${experiment}/resume_contract.py" --checkpoint-dir "${checkpoint_dir}" \
  --expected-step "${current_step}" --previous-step "${previous_step}" \
  --required-min-step "${STAGE_MIN_STEP}"

if [[ "${STAGE_MODE}" == smoke ]]; then
  /opt/nemo_rl_venv/bin/python "${experiment}/validate_gate.py" \
    --wandb-run-path "nvidia/sna-nemo-rl-online-drafter/${WANDB_RUN_ID}" \
    --log "${evidence_log}"
  python3 "${experiment}/resume_contract.py" --checkpoint-dir "${checkpoint_dir}" \
    --expected-step "${current_step}" --manifest "${manifest}" \
    --git-sha "${EXPECTED_HEAD}" --wandb-run-id "${WANDB_RUN_ID}" \
    --target-revision "${target_revision}" --drafter-revision "${drafter_revision}" \
    --container-sha256 "${container_sha}" --write-manifest
fi
echo "WANDB_URL=https://wandb.ai/nvidia/sna-nemo-rl-online-drafter/runs/${WANDB_RUN_ID}"
