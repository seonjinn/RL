#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR_OVERRIDE:-$(realpath "${SCRIPT_DIR}/../..")}
ACTION=${ACTION:-dry-run}
case "${ACTION}" in
    dry-run|test-only|submit) ;;
    *) echo "Unsupported ACTION: ${ACTION}" >&2; exit 2 ;;
esac

VLLM_GIT_URL=${VLLM_GIT_URL:-https://github.com/seonjinn/vllm.git}
VLLM_GIT_REF=${VLLM_GIT_REF:-cf856f2fb510f7de46e06fc6bdb6ca3a7fdfc5df}
VLLM_WHEEL=${VLLM_WHEEL:-https://github.com/vllm-project/vllm/releases/download/v0.25.1/vllm-0.25.1-cp38-abi3-manylinux_2_28_aarch64.whl}

ACCOUNT=${SLURM_ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-batch}
QOS=${QOS:-}
WALLTIME=${WALLTIME:-02:00:00}
WORK_ROOT=${WORK_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}
CONTAINER=${CONTAINER:-${WORK_ROOT}/containers/nemo_rl_nightly_20260711_vllm025_ffmpeg_20260713_1218.sqsh}
PREP_ROOT=${PREP_ROOT:-${WORK_ROOT}/experiments/qwen30b-mxfp8-linear-backends/prepare}

mkdir -p "${PREP_ROOT}"

COMMAND=$(cat <<EOF
set -euo pipefail
cd ${REPO_DIR}
export UV_PROJECT_ENVIRONMENT=
git submodule update --init --recursive --depth 1
existing_vllm_valid=false
if [[ -d 3rdparty/vllm/.git ]]; then
  actual=\$(git -C 3rdparty/vllm rev-parse HEAD)
  [[ "\${actual}" == "${VLLM_GIT_REF}" ]] || {
    echo "Existing custom vLLM commit is \${actual}" >&2
    exit 1
  }
  if [[ -x 3rdparty/vllm/.venv/bin/python ]] && \
      3rdparty/vllm/.venv/bin/python -c 'import vllm'; then
    existing_vllm_valid=true
  fi
fi
if [[ "\${existing_vllm_valid}" == true ]]; then
  printf 'export VLLM_GIT_REF=%s\nexport VLLM_PRECOMPILED_WHEEL_LOCATION=%s\n' \
    '${VLLM_GIT_REF}' '${VLLM_WHEEL}' > 3rdparty/vllm/nemo-rl.env
else
  if [[ -e 3rdparty/vllm ]]; then
    incomplete=3rdparty/vllm.incomplete.\${SLURM_JOB_ID:-\$\$}
    echo "Moving incomplete custom vLLM build to \${incomplete}"
    mv 3rdparty/vllm "\${incomplete}"
  fi
  bash tools/build-custom-vllm.sh ${VLLM_GIT_URL} ${VLLM_GIT_REF} ${VLLM_WHEEL}
fi
source 3rdparty/vllm/nemo-rl.env
export NRL_FORCE_REBUILD_VENVS=true
export SETUPTOOLS_SCM_PRETEND_VERSION=0.25.1
UV_PROJECT_ENVIRONMENT=${REPO_DIR}/3rdparty/vllm/.venv uv lock
3rdparty/vllm/.venv/bin/python - <<'PY'
import flashinfer
import vllm
from vllm.model_executor.kernels.linear import (
    FlashInferCutedslMxfp8LinearKernel,
    FlashInferCutlassMxfp8LinearKernel,
    FlashInferTrtllmMxfp8LinearKernel,
)

print(f"vLLM={vllm.__version__} path={vllm.__file__}")
print(f"FlashInfer={flashinfer.__version__}")
print(
    "MXFP8 kernels=",
    FlashInferCutlassMxfp8LinearKernel.__name__,
    FlashInferCutedslMxfp8LinearKernel.__name__,
    FlashInferTrtllmMxfp8LinearKernel.__name__,
)
PY
EOF
)

export CONTAINER
export MOUNTS=/lustre:/lustre
export COMMAND
export GPUS_PER_NODE=4
export BASE_LOG_DIR=${PREP_ROOT}

SBATCH_ARGS=(
    --nodes=1
    --gpus-per-node=4
    --exclusive
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --segment=1
    --time="${WALLTIME}"
    --job-name=q30-mx-vllm-prep
    --output="${PREP_ROOT}/slurm-%j.out"
)
if [[ -n "${QOS}" ]]; then
    SBATCH_ARGS+=(--qos="${QOS}")
fi

printf 'sbatch_args='; printf ' %q' "${SBATCH_ARGS[@]}"; printf '\n'
printf '%s\n' "${COMMAND}"

case "${ACTION}" in
    dry-run) ;;
    test-only) sbatch --test-only "${SBATCH_ARGS[@]}" "${REPO_DIR}/ray.sub" ;;
    submit) sbatch "${SBATCH_ARGS[@]}" "${REPO_DIR}/ray.sub" ;;
esac
