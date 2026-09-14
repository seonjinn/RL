#!/usr/bin/env bash
#SBATCH --job-name=sna-q8-rp25-swa-gate
#SBATCH --account=nemotron_sw_post
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --segment=1
#SBATCH --time=01:30:00

set -euo pipefail
ulimit -c 0
expected_head=${1:?expected commit}
bundle=${2:?immutable git bundle}
bundle_sha=${3:?bundle sha256}
result_root=${4:?durable result directory}
base_archive=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/qwen3_8b_cadence_200step/20260824-b9f02264-fixed8-recovery1/source.tar
base_sha=a47aa60d295793ccd4800286df1faae53cb0de7ad555998839277a9695b79149
container=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260818_20260818_6296116.sqsh
container_sha=6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44
scratch_root=/raid/scratch/sna/q8-swa-gate-${SLURM_JOB_ID}
source_root=${scratch_root}/source
test ! -e "${scratch_root}"
mkdir -p "${source_root}" "${result_root}"
test "$(sha256sum "${base_archive}" | cut -d ' ' -f1)" = "${base_sha}"
test "$(sha256sum "${bundle}" | cut -d ' ' -f1)" = "${bundle_sha}"
test "$(sha256sum "${container}" | cut -d ' ' -f1)" = "${container_sha}"
tar -xf "${base_archive}" -C "${source_root}"
git -C "${source_root}" fetch "${bundle}" refs/heads/codex/q8-rp25-online-20260914
git -C "${source_root}" checkout --detach "${expected_head}"
test "$(git -C "${source_root}" rev-parse HEAD)" = "${expected_head}"
test -z "$(git -C "${source_root}" status --porcelain --untracked-files=no)"
git -C "${source_root}" submodule status --recursive > "${result_root}/submodules-${SLURM_JOB_ID}.txt"
if grep -qv '^ ' "${result_root}/submodules-${SLURM_JOB_ID}.txt"; then
    exit 69
fi

srun --nodes=1 --ntasks=1 --mpi=pmix \
    --container-image="${container}" \
    --container-mounts="${source_root}:${source_root},${scratch_root}:${scratch_root},${result_root}:${result_root}" \
    --container-workdir="${source_root}" --container-remap-root \
    bash -s -- "${source_root}" "${scratch_root}" "${result_root}" "${expected_head}" <<'INNER'
set -euo pipefail
source_root=$1
scratch_root=$2
result_root=$3
expected_head=$4
export UV_PROJECT_ENVIRONMENT="${scratch_root}/venv"
export UV_CACHE_DIR="${scratch_root}/uv-cache"
export XDG_CACHE_HOME="${scratch_root}/xdg-cache"
export TRITON_CACHE_DIR="${scratch_root}/triton-cache"
export TORCH_EXTENSIONS_DIR="${scratch_root}/torch-extensions"
export TMPDIR="${scratch_root}/tmp"
export PYTHONPYCACHEPREFIX="${scratch_root}/pycache"
mkdir -p "${TMPDIR}"
uv sync --locked --extra mcore --group test --group dev
export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"
export PYTHONPATH="${source_root}:${source_root}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${source_root}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
cudnn_lib="${UV_PROJECT_ENVIRONMENT}/lib/python3.13/site-packages/nvidia/cudnn/lib"
export CUDNN_HOME="${cudnn_lib}"
export CUDNN_PATH="${cudnn_lib}"
export LD_LIBRARY_PATH="${cudnn_lib}:${LD_LIBRARY_PATH:-}"
uv run --no-sync python -c 'import torch, megatron.core, transformer_engine; print("SWA_GATE_RUNTIME", torch.__version__, torch.cuda.get_device_name()); assert torch.cuda.is_available()'
uv run --no-sync python -m pytest --noconftest -q --maxfail=0 \
    tests/unit/models/policy/test_draft_attention_config.py \
    tests/unit/models/megatron/test_dflash_model.py \
    tests/unit/models/megatron/test_dflash_training_provider.py \
    tests/unit/models/megatron/test_dspark_training_provider.py \
    tests/unit/models/megatron/test_draft_sliding_window_provider.py \
    --junitxml="${result_root}/provider-${SLURM_JOB_ID}.xml"
uv run --no-sync python -m pytest --noconftest -q --maxfail=0 \
    tests/unit/models/megatron/test_dflash_block_attention.py \
    -k 'sliding_window or fp32_forward_and_gradient_parity or cuda_public_geometry' \
    --junitxml="${result_root}/attention-${SLURM_JOB_ID}.xml"
printf 'head=%s\nresult=PASS\n' "${expected_head}" > "${result_root}/gate-${SLURM_JOB_ID}.txt"
INNER
