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
VLLM_GIT_REF=${VLLM_GIT_REF:-a76062edee3a3ac23d47a93c7ce466f06a19111f}
VLLM_WHEEL=${VLLM_WHEEL:-https://github.com/vllm-project/vllm/releases/download/v0.25.1/vllm-0.25.1-cp38-abi3-manylinux_2_28_aarch64.whl}

ACCOUNT=${SLURM_ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-batch}
QOS=${QOS:-}
WALLTIME=${WALLTIME:-02:00:00}
WORK_ROOT=${WORK_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}
CONTAINER=${CONTAINER:-${WORK_ROOT}/containers/nemo_rl_nightly_20260711_vllm025_ffmpeg_20260713_1218.sqsh}
PREP_ROOT=${PREP_ROOT:-${WORK_ROOT}/experiments/qwen30b-mxfp8-linear-backends/prepare}
VLLM_VENV_BASE_ROOT=${VLLM_VENV_BASE_ROOT:-${WORK_ROOT}/.cache/nemo-rl-vllm0251-worker-venvs}

mkdir -p "${PREP_ROOT}"

COMMAND=$(cat <<EOF
set -euo pipefail
cd ${REPO_DIR}
export UV_PROJECT_ENVIRONMENT=
source ${REPO_DIR}/experiments/mxfp8_linear_backend_model_matrix/provenance.sh
PREPARATION_LOCK=\$(git rev-parse --path-format=absolute --git-path \
  mxfp8-vllm-preparation.lock)
exec 8>"\${PREPARATION_LOCK}"
flock 8
git submodule update --init --recursive
assert_preparation_scope_clean() {
  if [[ -n "\$(git status --porcelain --untracked-files=all -- . \
      ":(exclude)pyproject.toml" ":(exclude)uv.lock" \
      ":(exclude)3rdparty/vllm")" ]]; then
    echo "Preparation found disallowed NeMo-RL source changes" >&2
    return 1
  fi
}
assert_preparation_scope_clean
existing_vllm_valid=false
if [[ -d 3rdparty/vllm/.git ]]; then
  actual=\$(git -C 3rdparty/vllm rev-parse HEAD)
  if [[ "\${actual}" != "${VLLM_GIT_REF}" ]]; then
    echo "Replacing custom vLLM commit \${actual} with ${VLLM_GIT_REF}"
  elif ! mxfp8_assert_vllm_tracked_state 3rdparty/vllm; then
    echo "Preserving polluted custom vLLM checkout before rebuilding" >&2
  else
    if mxfp8_vllm_build_state_matches 3rdparty/vllm && \
        [[ -x 3rdparty/vllm/.venv/bin/python ]] && \
        3rdparty/vllm/.venv/bin/python -c 'import vllm'; then
      printf 'export VLLM_GIT_REF=%s\nexport VLLM_PRECOMPILED_WHEEL_LOCATION=%s\n' \
        '${VLLM_GIT_REF}' '${VLLM_WHEEL}' > 3rdparty/vllm/nemo-rl.env
    fi
    if mxfp8_vllm_reuse_state_valid \
        3rdparty/vllm '${VLLM_GIT_REF}' '${VLLM_WHEEL}'; then
      existing_vllm_valid=true
    else
      echo "Existing custom vLLM checkout is not reusable"
    fi
  fi
fi
if [[ "\${existing_vllm_valid}" == true ]]; then
  printf 'export VLLM_GIT_REF=%s\nexport VLLM_PRECOMPILED_WHEEL_LOCATION=%s\n' \
    '${VLLM_GIT_REF}' '${VLLM_WHEEL}' > 3rdparty/vllm/nemo-rl.env
else
  if [[ -e 3rdparty/vllm ]]; then
    incomplete=${PREP_ROOT}/vllm.incomplete.\${SLURM_JOB_ID:-\$\$}
    echo "Moving incomplete custom vLLM build to \${incomplete}"
    mv 3rdparty/vllm "\${incomplete}"
  fi
  bash tools/build-custom-vllm.sh ${VLLM_GIT_URL} ${VLLM_GIT_REF} ${VLLM_WHEEL}
fi
source 3rdparty/vllm/nemo-rl.env
export NRL_FORCE_REBUILD_VENVS=true
export SETUPTOOLS_SCM_PRETEND_VERSION=0.25.1
CONTAINER_RAY_VERSION=\$(python3 -c 'import ray; print(ray.__version__)')
echo "Pinning NeMo-RL driver Ray to container version \${CONTAINER_RAY_VERSION}"
UV_PROJECT_ENVIRONMENT=${REPO_DIR}/3rdparty/vllm/.venv \
  uv add --frozen --bounds exact "ray[default]==\${CONTAINER_RAY_VERSION}"
uv pip install --python 3rdparty/vllm/.venv/bin/python poetry-dynamic-versioning poetry pybind11
uv pip install --python 3rdparty/vllm/.venv/bin/python "ray[default]==\${CONTAINER_RAY_VERSION}"
3rdparty/vllm/.venv/bin/python - <<'PY'
from pathlib import Path

import tomlkit


pyproject_path = Path("pyproject.toml")
document = tomlkit.parse(pyproject_path.read_text())
uv_config = document["tool"]["uv"]
uv_config["environments"] = ["python_version == '3.13' and sys_platform == 'linux' and platform_machine == 'aarch64'"]
pyproject_path.write_text(tomlkit.dumps(document))
PY
UV_PROJECT_ENVIRONMENT=${REPO_DIR}/3rdparty/vllm/.venv uv lock --no-build-isolation --refresh-package ray
python3 - "\${CONTAINER_RAY_VERSION}" <<'PY'
import sys
import tomllib
from pathlib import Path

expected = sys.argv[1]
with Path("uv.lock").open("rb") as lock_file:
    lock = tomllib.load(lock_file)
locked = {
    package["version"]
    for package in lock["package"]
    if package["name"] == "ray"
}
if locked != {expected}:
    raise SystemExit(
        f"Ray lock mismatch: expected {expected}, found {sorted(locked)}"
    )
print(f"Ray lock verified: {expected}")
PY
export NRL_VENV_BOOTSTRAP_PACKAGES='--torch-backend cu130 torch==2.11.0 numpy setuptools setuptools-rust setuptools-scm'
export NRL_VENV_NO_BUILD_ISOLATION_PACKAGES=vllm
VLLM_ENVIRONMENT_KEY=\$(mxfp8_vllm_environment_key \
  "${REPO_DIR}" "${REPO_DIR}/3rdparty/vllm" "${CONTAINER}" \
  "\${NRL_VENV_BOOTSTRAP_PACKAGES}" \
  "\${NRL_VENV_NO_BUILD_ISOLATION_PACKAGES}")
VLLM_ENVIRONMENT_ROOT=${VLLM_VENV_BASE_ROOT}/\${VLLM_ENVIRONMENT_KEY}
validate_vllm_environment() {
  local environment_root=\$1
  local expected_key=\$2
  local require_ready=\${3:-true}
  local prepared_environment_key
  if [[ "\${require_ready}" == true ]]; then
    prepared_environment_key=\$(cat "\${environment_root}/READY" 2>/dev/null || true)
    [[ "\${prepared_environment_key}" == "\${expected_key}" ]] || return 1
  fi
  [[ -x "\${environment_root}/vllm-canonical/bin/python" ]] || return 1
  NEMO_RL_VENV_DIR="\${environment_root}" \
    "\${environment_root}/vllm-canonical/bin/python" - <<'PY'
import os
from pathlib import Path

from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
)
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES

root = Path(os.environ["NEMO_RL_VENV_DIR"])
canonical = (root / "vllm-canonical").resolve()
missing = []
for actor_fqn, py_executable in sorted(ACTOR_ENVIRONMENT_REGISTRY.items()):
    if py_executable != PY_EXECUTABLES.VLLM:
        continue
    alias = root / actor_fqn
    if not alias.is_symlink() or alias.resolve() != canonical:
        missing.append(actor_fqn)
if missing:
    raise SystemExit(f"Missing or invalid vLLM actor aliases: {missing}")
PY
}
mkdir -p "${VLLM_VENV_BASE_ROOT}"
exec 9>"${VLLM_VENV_BASE_ROOT}/.\${VLLM_ENVIRONMENT_KEY}.lock"
flock 9
if validate_vllm_environment \
    "\${VLLM_ENVIRONMENT_ROOT}" "\${VLLM_ENVIRONMENT_KEY}"; then
  echo "Reusing prepared vLLM environment \${VLLM_ENVIRONMENT_KEY}"
else
  rm -rf "\${VLLM_ENVIRONMENT_ROOT}"
  mkdir -p "\${VLLM_ENVIRONMENT_ROOT}"
  export NEMO_RL_VENV_DIR=\${VLLM_ENVIRONMENT_ROOT}
  UV_PROJECT_ENVIRONMENT=${REPO_DIR}/3rdparty/vllm/.venv \
    uv run --frozen --extra vllm python - <<'PY'
import os
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
)
from nemo_rl.utils.venvs import create_local_venv
from pathlib import Path

root = Path(os.environ["NEMO_RL_VENV_DIR"])
canonical_python = Path(create_local_venv(
    PY_EXECUTABLES.VLLM,
    "vllm-canonical",
))
canonical = canonical_python.parent.parent
actor_names = sorted(
    actor_fqn
    for actor_fqn, py_executable in ACTOR_ENVIRONMENT_REGISTRY.items()
    if py_executable == PY_EXECUTABLES.VLLM
)
required = {
    "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker",
    "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker",
}
missing = required.difference(actor_names)
if missing:
    raise RuntimeError(f"Missing required vLLM actor registrations: {sorted(missing)}")
for actor_name in actor_names:
    alias = root / actor_name
    alias.symlink_to(canonical.name, target_is_directory=True)
    print(f"Prepared vLLM actor alias: {actor_name} -> {canonical.name}")
PY
  \${VLLM_ENVIRONMENT_ROOT}/vllm-canonical/bin/python - <<'PY'
import flashinfer
import ray
import vllm

print(f"Prepared vLLM={vllm.__version__}")
print(f"Prepared FlashInfer={flashinfer.__version__}")
print(f"Prepared Ray={ray.__version__}")
PY
  validate_vllm_environment \
    "\${VLLM_ENVIRONMENT_ROOT}" "\${VLLM_ENVIRONMENT_KEY}" false
  printf '%s\n' \${VLLM_ENVIRONMENT_KEY} > \${VLLM_ENVIRONMENT_ROOT}/READY.tmp
  mv \${VLLM_ENVIRONMENT_ROOT}/READY.tmp \${VLLM_ENVIRONMENT_ROOT}/READY
fi
flock -u 9
export NEMO_RL_VENV_DIR=\${VLLM_ENVIRONMENT_ROOT}
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
mxfp8_assert_vllm_tracked_state 3rdparty/vllm
mxfp8_vllm_build_state_matches 3rdparty/vllm
assert_preparation_scope_clean
flock -u 8
EOF
)

export CONTAINER
export MOUNTS=/lustre:/lustre
export COMMAND
export GPUS_PER_NODE=4
export BASE_LOG_DIR=${PREP_ROOT}

SBATCH_ARGS=(
    --nodes=1
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
