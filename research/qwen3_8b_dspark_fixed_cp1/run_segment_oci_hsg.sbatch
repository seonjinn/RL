#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --partition=batch_long
#SBATCH --job-name=q8-fixed-dspark-k7

set -euo pipefail

: "${REMOTE_REPO:?Set exact /home checkout}"
: "${EXPECTED_HEAD:?Set signed composition SHA}"
: "${FINAL_DIR:?Set /lustre result directory}"
: "${CONTAINER:?Set immutable container}"
: "${TARGET_SNAPSHOT:?Set exact target snapshot}"
: "${DRAFTER_SNAPSHOT:?Set exact drafter snapshot}"
: "${WANDB_RUN_ID:?Set fresh persistent W&B run ID}"
: "${WANDB_RESUME:?Set allow for smoke/start or must for resume}"
: "${STAGE_MIN_STEP:?Set cumulative milestone}"
: "${STAGE_DEADLINE:?Set checkpoint deadline}"
: "${STAGE_MODE:?Set smoke, start, or resume}"
: "${ARM_NAME:?Set dspark-k5 or dspark-k7}"
: "${NUM_SPECULATIVE_TOKENS:?Set K=5 or K=7}"
: "${WANDB_API_KEY:?W&B cluster secret is required}"

readonly container_sha=6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44
readonly target_revision=b968826d9c46dd6066d109eabc6255188de91218
readonly drafter_revision=03326e5043815da1f81b109078b2889737c26017
readonly openai_wheel='https://files.pythonhosted.org/packages/5b/e6/ca496976bacdd7f86d4fa69359c2d9b5a547d7acbf0ae5cac7bff107ff50/openai-2.25.0-py3-none-any.whl#sha256=7c7c01a0a4b69771a29913e28d06bc1e9cee8781b4d206bb56cb946d8d2fcb23'
readonly preflight_code='import importlib.metadata as m; assert m.version("openai") == "2.25.0"; assert m.version("vllm") == "0.25.1"; from openai.types.responses import NamespaceTool; from vllm import LLM; print("RUNTIME_PREFLIGHT=openai=2.25.0,vllm=0.25.1")'
readonly experiment="${REMOTE_REPO}/research/qwen3_8b_dspark_fixed_cp1"
readonly checkpoint_dir="${FINAL_DIR}/checkpoints"
readonly manifest="${FINAL_DIR}/gate-manifest.json"
readonly smoke_proof="${FINAL_DIR}/smoke-proof.json"
readonly scheduler_log="/raid/scratch/nrl-dspark-fixed-${SLURM_JOB_ID}.out"
readonly scratch_root="/raid/scratch/f/${SLURM_JOB_ID}"
readonly mcore_source="${REMOTE_REPO}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
readonly mcore_overlay="${scratch_root}/mcore-overlay"
readonly actor_venv_root="${scratch_root}/actor-venvs"
readonly mcore_actor=nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
readonly vllm_actor=nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker
readonly rollout_actor=nemo_rl.experience.sync_rollout_actor.SyncRolloutActor
readonly ray_root="/raid/scratch/r/${SLURM_JOB_ID}"
readonly train_log="${scratch_root}/train.log"
readonly evidence_log="${scratch_root}/evidence.log"

[[ "${REMOTE_REPO}" == /home/* ]]
[[ "${FINAL_DIR}" == /lustre/* ]]
[[ "${STAGE_MODE}" == smoke || "${STAGE_MODE}" == start || "${STAGE_MODE}" == resume ]]
[[ "${ARM_NAME}" == dspark-k5 || "${ARM_NAME}" == dspark-k7 ]]
[[ "${ARM_NAME}" == "dspark-k${NUM_SPECULATIVE_TOKENS}" ]]
[[ "${NUM_SPECULATIVE_TOKENS}" == 5 || "${NUM_SPECULATIVE_TOKENS}" == 7 ]]
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
run_max_steps=1000
checkpointing_enabled=true
if [[ "${STAGE_MODE}" == smoke ]]; then
  run_max_steps=2
  checkpointing_enabled=false
fi
if [[ "${STAGE_MODE}" == resume ]]; then
  previous_step="$(python3 "${experiment}/resume_contract.py" \
    --checkpoint-dir "${checkpoint_dir}" --print-latest-step)"
  python3 "${experiment}/resume_contract.py" --checkpoint-dir "${checkpoint_dir}" \
    --expected-step "${previous_step}" --manifest "${manifest}" \
    --git-sha "${EXPECTED_HEAD}" --target-revision "${target_revision}" \
    --drafter-revision "${drafter_revision}" --container-sha256 "${container_sha}" \
    --num-speculative-tokens "${NUM_SPECULATIVE_TOKENS}" \
    --validate-manifest
else
  test ! -e "${manifest}"
  test ! -d "${checkpoint_dir}"
  if [[ "${STAGE_MODE}" == smoke ]]; then
    test ! -e "${smoke_proof}"
  else
    python3 "${experiment}/resume_contract.py" \
      --checkpoint-dir "${checkpoint_dir}" --smoke-proof "${smoke_proof}" \
      --git-sha "${EXPECTED_HEAD}" --target-revision "${target_revision}" \
      --drafter-revision "${drafter_revision}" --container-sha256 "${container_sha}" \
      --num-speculative-tokens "${NUM_SPECULATIVE_TOKENS}" \
      --validate-smoke-proof
  fi
fi
if (( previous_step >= STAGE_MIN_STEP )); then exit 0; fi

mkdir -p "${scratch_root}" "${ray_root}" "${FINAL_DIR}/logs"
if [[ "${STAGE_MODE}" != smoke ]]; then mkdir -p "${checkpoint_dir}"; fi
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
  elif test -f "${scheduler_log}"; then
    tail -n 4000 "${scheduler_log}" > \
      "${FINAL_DIR}/logs/job-${SLURM_JOB_ID}-to-${STAGE_MIN_STEP}.log"
  fi
  exit "${status}"
}
trap archive EXIT

srun --ntasks=1 --container-image="${CONTAINER}" \
  --container-mounts="/home:/home,/raid:/raid,/lustre:/lustre" \
  --no-container-mount-home bash -lc "set -euo pipefail
mkdir -p '${mcore_overlay}'
cp -a '${mcore_source}/megatron' '${mcore_overlay}/'
export PYTHONPATH='${mcore_overlay}:${REMOTE_REPO}'
export HF_HOME='/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home'
export TMPDIR='${scratch_root}'
export RAY_TMPDIR='${ray_root}'
export UV_CACHE_DIR='${scratch_root}/cache/uv'
export UV_LINK_MODE=symlink
export NEMO_RL_VENV_DIR='${actor_venv_root}'
export TOKENIZERS_PARALLELISM=false
unset VIRTUAL_ENV
mkdir -p '${scratch_root}/cache/uv'
cd '${REMOTE_REPO}'
UV_PROJECT_ENVIRONMENT='${actor_venv_root}/${mcore_actor}' \
  uv sync --frozen --extra mcore --no-install-project \
  --no-install-package deep-ep
'${actor_venv_root}/${mcore_actor}/bin/python' -c 'import megatron.core'
echo 'MCORE_ACTOR_PREFLIGHT=complete'
UV_PROJECT_ENVIRONMENT='${actor_venv_root}/${vllm_actor}' \
  uv sync --frozen --extra vllm --no-install-project \
  --no-install-package deep-ep --no-install-package deep-gemm
uv pip install --python '${actor_venv_root}/${vllm_actor}/bin/python' \
  --no-deps '${openai_wheel}'
'${actor_venv_root}/${vllm_actor}/bin/python' -c $(printf '%q' "${preflight_code}")
UV_PROJECT_ENVIRONMENT='${actor_venv_root}/${rollout_actor}' \
  uv sync --frozen --extra vllm --no-install-project \
  --no-install-package deep-ep --no-install-package deep-gemm
uv pip install --python '${actor_venv_root}/${rollout_actor}/bin/python' \
  --no-deps '${openai_wheel}'
'${actor_venv_root}/${rollout_actor}/bin/python' -c $(printf '%q' "${preflight_code}")
UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv uv run --frozen --no-sync \
  examples/run_grpo.py --config '${experiment}/config.yaml' \
  grpo.max_num_steps='${run_max_steps}' grpo.val_period=1 \
  checkpointing.enabled='${checkpointing_enabled}' checkpointing.save_period=100 \
  checkpointing.checkpoint_must_save_by='${STAGE_DEADLINE}' \
  checkpointing.checkpoint_dir='${checkpoint_dir}' \
  policy.model_name='${TARGET_SNAPSHOT}' policy.tokenizer.name='${TARGET_SNAPSHOT}' \
  policy.draft.enabled=false policy.draft.model_name='${DRAFTER_SNAPSHOT}' \
  policy.draft.update_probe_enabled=false \
  policy.generation.vllm_kwargs.speculative_config.model='${DRAFTER_SNAPSHOT}' \
  policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens='${NUM_SPECULATIVE_TOKENS}' \
  logger.log_dir='${scratch_root}/logs' \
  logger.wandb.group='qwen3-8b-dapomath17k-fixed-control-matrix-v1' \
  logger.wandb.name='qwen3-8b-${ARM_NAME}-fixed-control-seed42' \
  logger.wandb.tags='[fixed-public-drafter,dspark,qwen3-8b,${ARM_NAME},cudagraph,seed42]' \
  logger.wandb.config.draft_training_enabled=false \
  logger.wandb.config.draft_refit_enabled=false \
  logger.wandb.config.k='${NUM_SPECULATIVE_TOKENS}' \
  +logger.wandb.id='${WANDB_RUN_ID}' +logger.wandb.resume='${WANDB_RESUME}' \
  +logger.wandb.config.git_sha='${EXPECTED_HEAD}' \
  +logger.wandb.config.stage_min_step='${STAGE_MIN_STEP}' \
  2>&1 | tee '${train_log}'
cp '${train_log}' '${evidence_log}'
if [[ '${STAGE_MODE}' == smoke ]]; then
  /opt/nemo_rl_venv/bin/python '${experiment}/validate_gate.py' \
    --wandb-run-path 'nvidia/sna-nemo-rl-online-drafter/${WANDB_RUN_ID}' \
    --log '${evidence_log}' 2>&1 | tee -a '${evidence_log}'
fi
"

grep -Fq "Capturing CUDA graphs (PIECEWISE)" "${train_log}"
grep -Fq "Graph capturing finished" "${train_log}"
if [[ "${STAGE_MODE}" == smoke ]]; then
  python3 "${experiment}/resume_contract.py" \
    --checkpoint-dir "${checkpoint_dir}" --smoke-proof "${smoke_proof}" \
    --git-sha "${EXPECTED_HEAD}" --wandb-run-id "${WANDB_RUN_ID}" \
    --target-revision "${target_revision}" --drafter-revision "${drafter_revision}" \
    --container-sha256 "${container_sha}" \
    --num-speculative-tokens "${NUM_SPECULATIVE_TOKENS}" --write-smoke-proof
  echo "WANDB_URL=https://wandb.ai/nvidia/sna-nemo-rl-online-drafter/runs/${WANDB_RUN_ID}"
  exit 0
fi
current_step="$(python3 "${experiment}/resume_contract.py" \
  --checkpoint-dir "${checkpoint_dir}" --print-latest-step)"
python3 "${experiment}/resume_contract.py" --checkpoint-dir "${checkpoint_dir}" \
  --expected-step "${current_step}" --previous-step "${previous_step}" \
  --required-min-step "${STAGE_MIN_STEP}"

if [[ "${STAGE_MODE}" == start ]]; then
  python3 "${experiment}/resume_contract.py" --checkpoint-dir "${checkpoint_dir}" \
    --expected-step "${current_step}" --manifest "${manifest}" \
    --git-sha "${EXPECTED_HEAD}" --wandb-run-id "${WANDB_RUN_ID}" \
    --target-revision "${target_revision}" --drafter-revision "${drafter_revision}" \
    --container-sha256 "${container_sha}" \
    --num-speculative-tokens "${NUM_SPECULATIVE_TOKENS}" --write-manifest
fi
echo "WANDB_URL=https://wandb.ai/nvidia/sna-nemo-rl-online-drafter/runs/${WANDB_RUN_ID}"
