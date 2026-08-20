#!/usr/bin/env bash
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

readonly repo_root="${NRL_REPO_ROOT:?NRL_REPO_ROOT must be an absolute /home path}"
readonly commit="${NRL_COMMIT:?NRL_COMMIT must be a full Git SHA}"
readonly container="${NRL_CONTAINER:?NRL_CONTAINER must be an immutable image path}"
readonly arm="${NRL_ARM:?NRL_ARM must be baseline or dflash_k5}"
readonly range_start="${NRL_RANGE_START:-0}"
readonly range_end="${NRL_RANGE_END:-4}"
readonly batch_size="${NRL_BATCH_SIZE:-8}"
readonly final_root="${NRL_FINAL_ROOT:?NRL_FINAL_ROOT must be under /lustre}"
readonly target_model="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"  # pragma: allowlist secret
readonly draft_model="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/9b41424b7109f9c5413454f481b09a82b85333f4"  # pragma: allowlist secret
readonly manifest="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/datasets/dapomath17k-qwen3-8b-forced-long/first-1000-manifest.json"
readonly scratch_root="/raid/scratch/nrl-qwen3-8b-forced-long/${SLURM_JOB_ID}"
readonly config="${repo_root}/research/qwen3_8b_forced_long_rollout/config.yaml"
readonly runner="${repo_root}/research/qwen3_8b_forced_long_rollout/run_rollout.py"

[[ "${repo_root}" == /home/* ]] || { echo "NRL_REPO_ROOT must be under /home" >&2; exit 2; }
[[ "${final_root}" == /lustre/* ]] || { echo "NRL_FINAL_ROOT must be under /lustre" >&2; exit 2; }
[[ "${commit}" =~ ^[0-9a-f]{40}$ ]] || { echo "NRL_COMMIT must be a full Git SHA" >&2; exit 2; }
[[ "${arm}" == baseline || "${arm}" == dflash_k5 ]] || { echo "unsupported arm" >&2; exit 2; }
[[ "$(git -C "${repo_root}" rev-parse HEAD)" == "${commit}" ]] || { echo "checkout SHA mismatch" >&2; exit 2; }
[[ -z "$(git -C "${repo_root}" status --porcelain --untracked-files=all)" ]] || { echo "checkout is dirty" >&2; exit 2; }
git -C "${repo_root}" submodule foreach --recursive 'test -z "$(git status --porcelain --untracked-files=all)"' >/dev/null
for required_path in "${container}" "${target_model}" "${draft_model}" "${manifest}" "${config}" "${runner}"; do
  [[ -e "${required_path}" ]] || { echo "missing required path: ${required_path}" >&2; exit 2; }
done
: "${WANDB_API_KEY:?WANDB_API_KEY must be injected by the cluster secret environment}"

mkdir -p "${scratch_root}/logs" "${scratch_root}/ranks" "${final_root}"
exec > >(tee "${scratch_root}/logs/job.log") 2>&1

archive_results() {
  readonly status=$?
  trap - EXIT
  set +e
  for output_root in "${scratch_root}"/ranks/rank*/output; do
    [[ -d "${output_root}" ]] || continue
    gzip -f "${output_root}"/*.jsonl 2>/dev/null || true
  done
  artifact_paths=(logs)
  for output_root in "${scratch_root}"/ranks/rank*/output; do
    [[ -d "${output_root}" ]] || continue
    artifact_paths+=("${output_root#"${scratch_root}"/}")
  done
  readonly artifact="${final_root}/job-${SLURM_JOB_ID}-${arm}-${range_start}-${range_end}-exit${status}.tar.gz"
  tar -C "${scratch_root}" -czf "${artifact}" "${artifact_paths[@]}"
  sha256sum "${artifact}" > "${artifact}.sha256"
  echo "final_artifact=${artifact}"
  exit "${status}"
}
trap archive_results EXIT

echo "commit=${commit} arm=${arm} range=${range_start}:${range_end} batch_size=${batch_size}"
echo "container_sha256=6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44"

export NRL_TARGET_MODEL="${target_model}"
export NRL_DRAFT_MODEL="${draft_model}"
export HF_HOME="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home"
export HF_DATASETS_CACHE="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/datasets"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export UV_PROJECT_ENVIRONMENT="${scratch_root}/runtime"
export UV_CACHE_DIR="${scratch_root}/cache/uv"
mkdir -p "${UV_CACHE_DIR}"

srun \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=24 \
  --gpus-per-task=1 \
  --no-container-mount-home \
  --container-image="${container}" \
  --container-mounts=/home:/home,/raid:/raid,/lustre:/lustre \
  --container-workdir="${repo_root}" \
  bash -lc '
    set -euo pipefail
    unset VIRTUAL_ENV
    export UV_PROJECT_ENVIRONMENT='"${scratch_root}"'/runtime
    export UV_CACHE_DIR='"${scratch_root}"'/cache/uv
    cd '"${repo_root}"'
    uv sync \
      --frozen \
      --extra vllm \
      --no-install-project \
      --no-install-package deep-ep \
      --no-install-package deep-gemm
    test -x "${UV_PROJECT_ENVIRONMENT}/bin/python"
    "${UV_PROJECT_ENVIRONMENT}/bin/python" -c '\''
import importlib.metadata
from transformers import AutoTokenizer

version = importlib.metadata.version("vllm")
if version != "0.25.1":
    raise RuntimeError(f"expected vLLM 0.25.1, got {version}")
print(f"runtime_preflight=vllm-{version} transformers-AutoTokenizer-ok")
'\''
  '

srun \
  --no-container-mount-home \
  --container-image="${container}" \
  --container-mounts=/home:/home,/raid:/raid,/lustre:/lustre \
  --container-workdir="${repo_root}" \
  bash -lc '
    set -euo pipefail
    unset VIRTUAL_ENV
    export UV_PROJECT_ENVIRONMENT='"${scratch_root}"'/runtime
    rank=${SLURM_PROCID}
    rank_root='"${scratch_root}"'/ranks/rank${rank}
    mkdir -p "${rank_root}"/{output,cache,tmp,wandb}
    export XDG_CACHE_HOME="${rank_root}/cache/xdg"
    export UV_CACHE_DIR="${rank_root}/cache/uv"
    export TRITON_CACHE_DIR="${rank_root}/cache/triton"
    export TORCHINDUCTOR_CACHE_DIR="${rank_root}/cache/torchinductor"
    export VLLM_CONFIG_ROOT="${rank_root}/cache/vllm"
    export TMPDIR="${rank_root}/tmp"
    export WANDB_DIR="${rank_root}/wandb"
    cd '"${repo_root}"'
    '"${scratch_root}"'/runtime/bin/python '"${runner}"' \
      --config '"${config}"' \
      --arm '"${arm}"' \
      --start '"${range_start}"' \
      --end '"${range_end}"' \
      --rank "${rank}" \
      --world-size "${SLURM_NTASKS}" \
      --batch-size '"${batch_size}"' \
      --natural-probe \
      --output-dir "${rank_root}/output"
  '
