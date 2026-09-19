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
method=${5:?dflash, dspark or none}
phase=${6:-canary}
arm=${7:-${method}-always}
case "${phase}" in
    canary|resume-check|production|production-300|smoke|long-context|graph-fap-default|graph-fap-8|graph-fap-32|graph-fap-64|graph-packed-default|graph-packed-8|graph-packed-32|graph-packed-64) ;;
    *) exit 64 ;;
esac
[[ "${method}" == dflash || "${method}" == dspark || "${method}" == none ]] || exit 64
if [[ "${method}" == none ]]; then
    [[ ( "${phase}" == production || "${phase}" == production-300 || "${phase}" == smoke || "${phase}" == long-context || "${phase}" == graph-fap-* || "${phase}" == graph-packed-* ) && "${arm}" == baseline ]] || exit 64
else
    case "${arm}" in
        "${method}-always"|"${method}-frozen"|"${method}-fixed-5"|"${method}-fixed-10"|"${method}-fixed-20") ;;
        *) exit 64 ;;
    esac
fi
if [[ ( "${phase}" == long-context || "${phase}" == graph-fap-* || "${phase}" == graph-packed-* ) && "${arm}" != baseline && "${arm}" != "${method}-frozen" ]]; then exit 64; fi
if [[ ( "${phase}" == graph-fap-default || "${phase}" == graph-packed-default ) && "${arm}" != baseline ]]; then exit 64; fi
[[ "${result_root}" == /lustre/* ]] || exit 64
: "${WANDB_API_KEY:?Submit with the cluster W&B credential in the environment}"
base_archive=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/qwen3_8b_cadence_200step/20260824-b9f02264-fixed8-recovery1/source.tar
base_sha=a47aa60d295793ccd4800286df1faae53cb0de7ad555998839277a9695b79149
container=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260818_20260818_6296116.sqsh
container_sha=6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44
scratch_root=/raid/scratch/sna/q8-newdraft-online-${SLURM_JOB_ID}
source_root=${scratch_root}/source
test ! -e "${scratch_root}"
if [[ "${phase}" == resume-check ]]; then
    test -s "${result_root}/checkpoints/step_2/cadence-checkpoint-receipt.json"
    test ! -e "${result_root}/checkpoints/step_4"
    output_root=${result_root}/attempts/resume-${SLURM_JOB_ID}
else
    test ! -e "${result_root}/checkpoints"
    output_root=${result_root}
fi
mkdir -p "${source_root}" "${output_root}"
test "$(sha256sum "${base_archive}" | cut -d ' ' -f1)" = "${base_sha}"
test "$(sha256sum "${bundle}" | cut -d ' ' -f1)" = "${bundle_sha}"
test "$(sha256sum "${container}" | cut -d ' ' -f1)" = "${container_sha}"
tar -xf "${base_archive}" -C "${source_root}"
git -C "${source_root}" fetch "${bundle}" refs/heads/codex/q8-rp25-online-20260914
git -C "${source_root}" checkout --detach "${expected_head}"
test "$(git -C "${source_root}" rev-parse HEAD)" = "${expected_head}"
git -C "${source_root}" submodule status --recursive > "${output_root}/submodules.txt"
if grep -qv '^ ' "${output_root}/submodules.txt"; then exit 69; fi
if [[ "${phase}" == resume-check ]]; then
    # terminal_closed writes exclusively: retain the previous terminal summaries.
    for summary in checkpoint-runtime.json schedule-runtime.json; do
        test ! -e "${output_root}/${summary}"
        if [[ -e "${result_root}/${summary}" ]]; then
            test -s "${result_root}/${summary}"
            mv "${result_root}/${summary}" "${output_root}/${summary}"
        fi
    done
fi

/cm/local/apps/slurm/25.11/bin/srun --nodes=1 --ntasks=1 --mpi=pmix \
    --container-image="${container}" \
    --container-mounts="${source_root}:${source_root},${scratch_root}:${scratch_root},/lustre:/lustre" \
    --container-workdir="${source_root}" --container-remap-root --no-container-mount-home \
    bash -s -- "${source_root}" "${scratch_root}" "${result_root}" "${method}" "${expected_head}" "${phase}" "${output_root}" "${arm}" <<'INNER'
set -euo pipefail
source_root=$1
scratch_root=$2
result_root=$3
method=$4
expected_head=$5
phase=$6
output_root=$7
arm=$8
unset VIRTUAL_ENV RAY_ADDRESS
# Diagnostic only: do not assume that an image venv survives host mounts.
ls -ld /opt/nemo_rl_venv /opt/nemo_rl_venv/bin/python 2>/dev/null || true
readlink /opt/nemo_rl_venv/bin/python || true
export NRL_FORCE_REBUILD_VENVS=false NEMO_RL_PY_EXECUTABLES_SYSTEM=0
export UV_CACHE_DIR="${scratch_root}/uv-cache" XDG_CACHE_HOME="${scratch_root}/xdg-cache"
export UV_PYTHON_INSTALL_DIR="${scratch_root}/uv-python"
export TRITON_CACHE_DIR="${scratch_root}/triton-cache" TORCH_EXTENSIONS_DIR="${scratch_root}/torch-extensions"
export TMPDIR="${scratch_root}/tmp" PYTHONPYCACHEPREFIX="${scratch_root}/pycache"
export NEMO_RL_VENV_DIR="${scratch_root}/actor-venvs"
export RAY_TMPDIR="/tmp/rq8-${SLURM_JOB_ID}"
export HF_HOME="${scratch_root}/hf-cache"
export TOKENIZERS_PARALLELISM=false
if [[ "${phase}" == graph-fap-* || "${phase}" == graph-packed-* ]]; then
    export NRL_Q8_CG_AUDIT=1
fi
export PYTHONPATH="${source_root}:${source_root}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${source_root}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
export WANDB_RUN_ID="q8rp44-${arm}-${phase}-${SLURM_JOB_ID}"
export WANDB_RESUME=never
mkdir -p "${TMPDIR}" "${HF_HOME}" "${RAY_TMPDIR}" "${NEMO_RL_VENV_DIR}"
target_source=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
draft_source=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/specdec_ptv23/ptv3_swa/sd2p3rp-q8b-base-ptv3rp25-${method}-b8-16n/exported-checkpoint-44000
source research/qwen3_8b_rp25_swa/stage_models.sh
archive_logs() {
    status=$?
    trap - EXIT
    tar -cf "${output_root}/ray-logs-${SLURM_JOB_ID}.tar" -C "${RAY_TMPDIR}" . 2>/dev/null || true
    exit "${status}"
}
trap archive_logs EXIT
mcore_env=${NEMO_RL_VENV_DIR}/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
vllm_env=${NEMO_RL_VENV_DIR}/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker
rollout_env=${NEMO_RL_VENV_DIR}/nemo_rl.experience.sync_rollout_actor.SyncRolloutActor
UV_PROJECT_ENVIRONMENT="${mcore_env}" uv sync --frozen --extra mcore --no-install-project --no-install-package deep-ep
export CUDNN_HOME="${mcore_env}/lib/python3.13/site-packages/nvidia/cudnn/lib"
export CUDNN_PATH="${CUDNN_HOME}" LD_LIBRARY_PATH="${CUDNN_HOME}:${LD_LIBRARY_PATH:-}"
driver_python=${mcore_env}/bin/python
test -x "${driver_python}"
bash research/qwen3_8b_rp25_swa/install_driver_project.sh "${driver_python}" "${source_root}"
driver=(uv run --no-project --no-sync --python "${driver_python}" python)
"${driver[@]}" -c 'import sys; import examples.run_grpo; print("DRIVER_PREFLIGHT=PASS", sys.executable)'
"${driver[@]}" -c 'from nemo_rl.data_plane.adapters.transfer_queue import _resolve_tq_pin; print("TQ_PIN_PREFLIGHT=PASS", _resolve_tq_pin())'
UV_PROJECT_ENVIRONMENT="${vllm_env}" uv sync --frozen --extra vllm --no-install-project \
    --no-install-package deep-ep --no-install-package deep-gemm
uv pip install --python "${vllm_env}/bin/python" --no-deps \
    'https://files.pythonhosted.org/packages/5b/e6/ca496976bacdd7f86d4fa69359c2d9b5a547d7acbf0ae5cac7bff107ff50/openai-2.25.0-py3-none-any.whl#sha256=7c7c01a0a4b69771a29913e28d06bc1e9cee8781b4d206bb56cb946d8d2fcb23'
ln -s "${vllm_env}" "${rollout_env}"
"${vllm_env}/bin/python" -c 'import importlib.metadata as m; assert m.version("vllm")=="0.25.1"; assert m.version("openai")=="2.25.0"; from vllm import LLM; print("VLLM_PREFLIGHT=PASS")'
"${mcore_env}/bin/python" -c 'import megatron.core, transformer_engine; print("MCORE_PREFLIGHT=PASS")'
bash research/qwen3_8b_rp25_swa/render_canary.sh "${driver_python}" "${arm}" "${result_root}" "--${phase}" "${output_root}"
mapfile -t overrides < "${output_root}/overrides.txt"
test "${#overrides[@]}" -gt 30
read -r recipe < "${output_root}/recipe.txt"
printf 'head=%s\nrun_id=%s\n' "${expected_head}" "${WANDB_RUN_ID}" > "${output_root}/identity.txt"
"${driver[@]}" examples/run_grpo.py --config "${recipe}" "${overrides[@]}" \
    "${model_overrides[@]}" \
    ++logger.wandb.config.source_target="${target_source}" \
    ++logger.wandb.config.source_drafter="${draft_source}" \
    ++logger.wandb.config.source_head="${expected_head}" \
    ++logger.log_dir="${output_root}/logs" \
    2>&1 | tee "${output_root}/train.log"
printf 'process_exit=0\nreview_required=update_refit_next_rollout_and_checkpoint\n' > "${output_root}/process-completed.txt"
INNER
