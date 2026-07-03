#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER="${CLUSTER:-auto}"
if [[ "${CLUSTER}" == "auto" ]]; then
  case "$(hostname)" in
    *lyris*) CLUSTER="lyris" ;;
    *ptyche*) CLUSTER="ptyche" ;;
    *)
      echo "Set CLUSTER=lyris or CLUSTER=ptyche" >&2
      exit 2
      ;;
  esac
fi

ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
case "${CLUSTER}" in
  lyris) PARTITION="${PARTITION:-gb200}" ;;
  ptyche) PARTITION="${PARTITION:-batch}" ;;
  *)
    echo "Unsupported cluster: ${CLUSTER}" >&2
    exit 2
    ;;
esac

LUSTRE_ROOT="${LUSTRE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${LUSTRE_ROOT}/containers/vllm-openai-v0.24.0-aarch64-ubuntu2404.sqsh}"
HF_HOME="${HF_HOME:-${LUSTRE_ROOT}/hf_home}"
ASSET_ROOT="${ASSET_ROOT:-${LUSTRE_ROOT}/vllm024-dynamicsd/assets}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_extended_assets}"
RUN_DIR="${ASSET_ROOT}/staging/${RUN_ID}"
SBATCH_FILE="${RUN_DIR}/submit.sbatch"
VLLM_COMMIT="ee0da84ab9e04ac7610e28580af62c365e898389"
ANGELSLIM_COMMIT="6a97dab2f17c0a3c031065329f092c4f61108a6f"
PARD_COMMIT="6f279bf3f1680e0b5d71c562ca5b91bdeef4c038"
QWEN8_REVISION="b968826d9c46dd6066d109eabc6255188de91218"
PARD_REVISION="f9f650fbab180c26498817718f0db5cae8f25136"
PARD2_REVISION="67a1516c8f6fc145cda99916799a0cbb3a4af135"
DFLASH_REVISION="9b41424b7109f9c5413454f481b09a82b85333f4"
DFLARE_REVISION="55e2c8d86d76ce1e79fa3b8642c7f80091285a82"
DRY_RUN="${DRY_RUN:-false}"
TEST_ONLY="${TEST_ONLY:-false}"
REQUIRE_GIT_PULL="${REQUIRE_GIT_PULL:-true}"

render_sbatch() {
  cat <<EOF
#!/usr/bin/env bash
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --segment=1
#SBATCH --time=02:00:00
#SBATCH --job-name=coreai_dlalgo_llm-vllm024.extended-assets
#SBATCH --output=${RUN_DIR}/slurm-%j.out

set -euo pipefail

test -s '${CONTAINER_IMAGE}'
mkdir -p '${ASSET_ROOT}/src' '${ASSET_ROOT}/python' '${HF_HOME}'

srun --ntasks=1 \\
  --container-image='${CONTAINER_IMAGE}' \\
  --container-mounts='/lustre:/lustre,${SCRIPT_DIR}:/workspace/experiment' \\
  --no-container-mount-home \\
  --container-remap-root \\
  --mpi=pmix \\
  bash -lc "set -euo pipefail
export HF_HOME='${HF_HOME}'
export HUGGINGFACE_HUB_CACHE='${HF_HOME}/hub'
export HF_DATASETS_CACHE='${HF_HOME}/datasets'
export PYTHONUNBUFFERED=1

python3 - <<'PY'
from huggingface_hub import snapshot_download

assets = (
    ('Qwen/Qwen3-8B', '${QWEN8_REVISION}'),
    ('amd/PARD-Qwen3-0.6B', '${PARD_REVISION}'),
    ('amd/PARD2-Qwen3-8B', '${PARD2_REVISION}'),
    ('z-lab/Qwen3-8B-DFlash-b16', '${DFLASH_REVISION}'),
    ('AngelSlim/Qwen3-8b-dflare', '${DFLARE_REVISION}'),
)
for repo_id, revision in assets:
    path = snapshot_download(repo_id=repo_id, revision=revision)
    print(f'staged={repo_id}@{revision} path={path}', flush=True)
PY

clone_pinned() {
  local repository=\"\$1\"
  local commit=\"\$2\"
  local destination=\"\$3\"
  if [[ ! -d \"\${destination}/.git\" ]]; then
    git clone --filter=blob:none \"\${repository}\" \"\${destination}\"
  fi
  git -C \"\${destination}\" fetch origin \"\${commit}\" --depth=1
  git -C \"\${destination}\" checkout --detach \"\${commit}\"
  test \"\$(git -C \"\${destination}\" rev-parse HEAD)\" = \"\${commit}\"
}

VLLM_SOURCE='${ASSET_ROOT}/src/vllm-${VLLM_COMMIT}'
ANGELSLIM_SOURCE='${ASSET_ROOT}/src/angelslim-${ANGELSLIM_COMMIT}'
PARD_SOURCE='${ASSET_ROOT}/src/pard-${PARD_COMMIT}'
clone_pinned https://github.com/vllm-project/vllm.git '${VLLM_COMMIT}' \"\${VLLM_SOURCE}\"
clone_pinned https://github.com/Tencent/AngelSlim.git '${ANGELSLIM_COMMIT}' \"\${ANGELSLIM_SOURCE}\"
clone_pinned https://github.com/AMD-AGI/PARD.git '${PARD_COMMIT}' \"\${PARD_SOURCE}\"

if ! grep -q 'method in (\"draft_model\", \"pard2\")' \"\${VLLM_SOURCE}/vllm/config/speculative.py\"; then
  patch -p1 -d \"\${VLLM_SOURCE}\" < /workspace/experiment/patches/vllm024_pard2_target_features.patch
fi
if ! grep -q 'output-json' \"\${ANGELSLIM_SOURCE}/tools/dflash_benchmark.py\"; then
  patch -p1 -d \"\${ANGELSLIM_SOURCE}\" < /workspace/experiment/patches/angelslim_benchmark_json.patch
fi

COMMON_SITE_VERSIONED='${ASSET_ROOT}/python/common-py312-arctic-0.1.1'
if [[ ! -f \"\${COMMON_SITE_VERSIONED}/.complete\" ]]; then
  mkdir -p \"\${COMMON_SITE_VERSIONED}\"
  CMAKE_BUILD_PARALLEL_LEVEL=16 python3 -m pip install \\
    --no-build-isolation --no-deps --no-cache-dir \\
    --target \"\${COMMON_SITE_VERSIONED}\" 'arctic-inference==0.1.1'
  PYTHONPATH=\"\${COMMON_SITE_VERSIONED}\" python3 -c 'import arctic_inference'
  touch \"\${COMMON_SITE_VERSIONED}/.complete\"
fi
ln -sfn \"\${COMMON_SITE_VERSIONED}\" '${ASSET_ROOT}/python/common'

PARD2_OVERLAY_VERSIONED='${ASSET_ROOT}/python/pard2-overlay-${VLLM_COMMIT}'
if [[ ! -f \"\${PARD2_OVERLAY_VERSIONED}/.complete\" ]]; then
  PARD2_OVERLAY_TMP=\"\${PARD2_OVERLAY_VERSIONED}.partial.\${SLURM_JOB_ID}\"
  rm -rf \"\${PARD2_OVERLAY_TMP}\"
  mkdir -p \"\${PARD2_OVERLAY_TMP}\"
  OVERLAY_TMP=\"\${PARD2_OVERLAY_TMP}\" python3 - <<'PY'
import os
import shutil
from pathlib import Path

import vllm

source = Path(vllm.__file__).resolve().parent
destination = Path(os.environ['OVERLAY_TMP']) / 'vllm'
shutil.copytree(source, destination)
PY
  for relative_path in \\
    vllm/config/speculative.py \\
    vllm/model_executor/models/qwen3.py \\
    vllm/v1/spec_decode/draft_model.py \\
    vllm/v1/spec_decode/llm_base_proposer.py \\
    vllm/v1/worker/gpu_model_runner.py; do
    install -D -m 0644 \\
      \"\${VLLM_SOURCE}/\${relative_path}\" \\
      \"\${PARD2_OVERLAY_TMP}/\${relative_path}\"
  done
  python3 -m compileall -q \"\${PARD2_OVERLAY_TMP}/vllm\"
  touch \"\${PARD2_OVERLAY_TMP}/.complete\"
  mv \"\${PARD2_OVERLAY_TMP}\" \"\${PARD2_OVERLAY_VERSIONED}\"
fi
ln -sfn \"\${PARD2_OVERLAY_VERSIONED}\" '${ASSET_ROOT}/python/pard2_overlay'

PYTHONPATH='${ASSET_ROOT}/python/pard2_overlay:${ASSET_ROOT}/python/common' python3 - <<'PY'
import inspect
import vllm
from vllm.config.speculative import SpeculativeConfig

assert vllm.__version__ == '0.24.0', vllm.__version__
assert 'pard2' in inspect.getsource(SpeculativeConfig.uses_draft_model)
print(f'validated_vllm={vllm.__version__}')
PY
PYTHONPATH=\"\${ANGELSLIM_SOURCE}\" python3 - <<'PY'
from angelslim.compressor.speculative.train.models.draft.qwen_dflash import QwenDFlashDraftModel
from angelslim.compressor.speculative.train.models.draft.qwen_dflare import QwenDFlareDraftModel

print(QwenDFlashDraftModel.__name__, QwenDFlareDraftModel.__name__)
PY
"
EOF
}

if [[ "${DRY_RUN}" != "true" && "${REQUIRE_GIT_PULL}" == "true" ]]; then
  git -C "${SCRIPT_DIR}" pull --ff-only
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  render_sbatch
  exit 0
fi

mkdir -p "${RUN_DIR}"
render_sbatch >"${SBATCH_FILE}"
if [[ "${TEST_ONLY}" == "true" ]]; then
  sbatch --test-only "${SBATCH_FILE}"
  exit 0
fi

job_id="$(sbatch --parsable "${SBATCH_FILE}")"
printf 'job_id=%s\nrun_dir=%s\n' "${job_id}" "${RUN_DIR}"
