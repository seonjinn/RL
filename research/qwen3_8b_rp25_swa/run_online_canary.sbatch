#!/usr/bin/env bash
#SBATCH --job-name=sna-q8-newdraft-online-canary
#SBATCH --account=nemotron_sw_post
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --segment=1
#SBATCH --time=02:00:00

set -euo pipefail
ulimit -c 0
expected_head=${1:?expected commit}
bundle=${2:?immutable source bundle}
bundle_sha=${3:?bundle sha256}
result_root=${4:?fresh Lustre result directory}
method=${5:?dflash or dspark}
[[ "${method}" == dflash || "${method}" == dspark ]]
[[ "${result_root}" == /lustre/* ]]
: "${WANDB_API_KEY:?Submit with the cluster W&B credential in the environment}"
base_archive=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/qwen3_8b_cadence_200step/20260824-b9f02264-fixed8-recovery1/source.tar
base_sha=a47aa60d295793ccd4800286df1faae53cb0de7ad555998839277a9695b79149
container=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260818_20260818_6296116.sqsh
container_sha=6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44
scratch_root=/raid/scratch/sna/q8-newdraft-online-${SLURM_JOB_ID}
source_root=${scratch_root}/source
test ! -e "${scratch_root}"
test ! -e "${result_root}/checkpoints"
mkdir -p "${source_root}" "${result_root}"
test "$(sha256sum "${base_archive}" | cut -d ' ' -f1)" = "${base_sha}"
test "$(sha256sum "${bundle}" | cut -d ' ' -f1)" = "${bundle_sha}"
test "$(sha256sum "${container}" | cut -d ' ' -f1)" = "${container_sha}"
tar -xf "${base_archive}" -C "${source_root}"
git -C "${source_root}" fetch "${bundle}" refs/heads/codex/q8-rp25-online-20260914
git -C "${source_root}" checkout --detach "${expected_head}"
test "$(git -C "${source_root}" rev-parse HEAD)" = "${expected_head}"
git -C "${source_root}" submodule status --recursive > "${result_root}/submodules.txt"
if grep -qv '^ ' "${result_root}/submodules.txt"; then exit 69; fi

/cm/local/apps/slurm/25.11/bin/srun --nodes=1 --ntasks=1 --mpi=pmix \
    --container-image="${container}" \
    --container-mounts="${source_root}:${source_root},${scratch_root}:${scratch_root},/lustre:/lustre" \
    --container-workdir="${source_root}" --container-remap-root \
    bash -s -- "${source_root}" "${scratch_root}" "${result_root}" "${method}" "${expected_head}" <<'INNER'
set -euo pipefail
source_root=$1
scratch_root=$2
result_root=$3
method=$4
expected_head=$5
export UV_CACHE_DIR="${scratch_root}/uv-cache" XDG_CACHE_HOME="${scratch_root}/xdg-cache"
export TRITON_CACHE_DIR="${scratch_root}/triton-cache" TORCH_EXTENSIONS_DIR="${scratch_root}/torch-extensions"
export TMPDIR="${scratch_root}/tmp" PYTHONPYCACHEPREFIX="${scratch_root}/pycache"
export NEMO_RL_VENV_DIR="${scratch_root}/actor-venvs"
export RAY_TMPDIR="/tmp/rq8-${SLURM_JOB_ID}"
export HF_HOME="${scratch_root}/hf-cache"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${source_root}:${source_root}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${source_root}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
export WANDB_RUN_ID="q8rp44-${method}-canary-${SLURM_JOB_ID}"
export WANDB_RESUME=never
mkdir -p "${TMPDIR}" "${HF_HOME}" "${RAY_TMPDIR}" "${NEMO_RL_VENV_DIR}"
target_source=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
draft_source=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/specdec_ptv23/ptv3_swa/sd2p3rp-q8b-base-ptv3rp25-${method}-b8-16n/exported-checkpoint-44000
cp -aL "${target_source}" "${scratch_root}/target"
cp -aL "${draft_source}" "${scratch_root}/draft"
printf 'target_source=%s\ndraft_source=%s\n' "${target_source}" "${draft_source}" > "${result_root}/input-paths.txt"
sha256sum "${scratch_root}/target/config.json" "${scratch_root}/draft/config.json" \
    "${scratch_root}/draft/model.safetensors" > "${result_root}/input-sha256.txt"
archive_logs() {
    status=$?
    trap - EXIT
    tar -cf "${result_root}/ray-logs-${SLURM_JOB_ID}.tar" -C "${RAY_TMPDIR}" . 2>/dev/null || true
    exit "${status}"
}
trap archive_logs EXIT
mcore_env=${NEMO_RL_VENV_DIR}/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
vllm_env=${NEMO_RL_VENV_DIR}/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker
rollout_env=${NEMO_RL_VENV_DIR}/nemo_rl.experience.sync_rollout_actor.SyncRolloutActor
UV_PROJECT_ENVIRONMENT="${mcore_env}" uv sync --frozen --extra mcore --no-install-project --no-install-package deep-ep
UV_PROJECT_ENVIRONMENT="${vllm_env}" uv sync --frozen --extra vllm --no-install-project \
    --no-install-package deep-ep --no-install-package deep-gemm
uv pip install --python "${vllm_env}/bin/python" --no-deps \
    'https://files.pythonhosted.org/packages/5b/e6/ca496976bacdd7f86d4fa69359c2d9b5a547d7acbf0ae5cac7bff107ff50/openai-2.25.0-py3-none-any.whl#sha256=7c7c01a0a4b69771a29913e28d06bc1e9cee8781b4d206bb56cb946d8d2fcb23'
ln -s "${vllm_env}" "${rollout_env}"
"${vllm_env}/bin/python" -c 'import importlib.metadata as m; assert m.version("vllm")=="0.25.1"; assert m.version("openai")=="2.25.0"; from vllm import LLM; print("VLLM_PREFLIGHT=PASS")'
export CUDNN_HOME="${mcore_env}/lib/python3.13/site-packages/nvidia/cudnn/lib"
export CUDNN_PATH="${CUDNN_HOME}" LD_LIBRARY_PATH="${CUDNN_HOME}:${LD_LIBRARY_PATH:-}"
"${mcore_env}/bin/python" -c 'import megatron.core, transformer_engine; print("MCORE_PREFLIGHT=PASS")'
mapfile -t overrides < <(/opt/nemo_rl_venv/bin/python -m research.qwen3_8b_rp25_swa.study \
    --arm "${method}-always" --result-dir "${result_root}" --canary)
test "${#overrides[@]}" -gt 30
recipe=$(/opt/nemo_rl_venv/bin/python -m research.qwen3_8b_rp25_swa.study \
    --arm "${method}-always" --result-dir "${result_root}" --canary --recipe)
printf '%s\n' "${overrides[@]}" > "${result_root}/overrides.txt"
printf 'head=%s\nrun_id=%s\n' "${expected_head}" "${WANDB_RUN_ID}" > "${result_root}/identity.txt"
/opt/nemo_rl_venv/bin/python examples/run_grpo.py --config "${recipe}" "${overrides[@]}" \
    ++policy.model_name="${scratch_root}/target" ++policy.tokenizer.name="${scratch_root}/target" \
    ++policy.draft.model_name="${scratch_root}/draft" \
    ++policy.generation.vllm_kwargs.speculative_config.model="${scratch_root}/draft" \
    ++logger.wandb.config.source_target="${target_source}" \
    ++logger.wandb.config.source_drafter="${draft_source}" \
    ++logger.wandb.config.source_head="${expected_head}" \
    2>&1 | tee "${result_root}/train.log"
printf 'process_exit=0\nreview_required=update_refit_next_rollout_and_checkpoint\n' > "${result_root}/process-completed.txt"
INNER
