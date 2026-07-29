#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -eou pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(realpath "$SCRIPT_DIR/..")"

usage() {
  echo "Usage: $0 <GIT_URL> <IMMUTABLE_GIT_COMMIT> <VLLM_PRECOMPILED_WHEEL_LOCATION>" >&2
  exit 2
}

if [[ $# -ne 3 || -z "$1" || -z "$2" || -z "$3" ]]; then
  usage
fi

GIT_URL="$1"
GIT_REF="$2"
VLLM_PRECOMPILED_WHEEL_LOCATION="$3"
if [[ ! "$GIT_REF" =~ ^[0-9a-f]{40}$ ]]; then
  echo "[ERROR] GIT_REF must be an immutable 40-character Git commit." >&2
  exit 2
fi

export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_LOCATION
export VLLM_VERSION_OVERRIDE=0.20.2

OLD_UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-}"
OLD_VIRTUAL_ENV="${VIRTUAL_ENV:-}"
if [[ -n "$OLD_UV_PROJECT_ENVIRONMENT" && -n "$OLD_VIRTUAL_ENV" ]]; then
  UV_PROJECT_PYTHON="$OLD_UV_PROJECT_ENVIRONMENT/bin/python"
  VIRTUAL_ENV_PYTHON="$OLD_VIRTUAL_ENV/bin/python"
  if [[ ! -x "$UV_PROJECT_PYTHON" || ! -x "$VIRTUAL_ENV_PYTHON" ]]; then
    echo "[ERROR] Both configured root environment selectors must contain an executable bin/python." >&2
    exit 2
  fi
  UV_PROJECT_PYTHON="$(realpath "$UV_PROJECT_PYTHON")"
  VIRTUAL_ENV_PYTHON="$(realpath "$VIRTUAL_ENV_PYTHON")"
  if [[ "$UV_PROJECT_PYTHON" != "$VIRTUAL_ENV_PYTHON" ]]; then
    echo "[ERROR] UV_PROJECT_ENVIRONMENT and VIRTUAL_ENV select different Python environments." >&2
    exit 2
  fi
  ROOT_PYTHON="$UV_PROJECT_PYTHON"
elif [[ -n "$OLD_VIRTUAL_ENV" ]]; then
  ROOT_PYTHON="$OLD_VIRTUAL_ENV/bin/python"
elif [[ -n "$OLD_UV_PROJECT_ENVIRONMENT" ]]; then
  ROOT_PYTHON="$OLD_UV_PROJECT_ENVIRONMENT/bin/python"
else
  ROOT_PYTHON="$REPO_ROOT/.venv/bin/python"
fi
if [[ ! -x "$ROOT_PYTHON" ]]; then
  echo "[ERROR] Root project Python is not executable at $ROOT_PYTHON." >&2
  exit 2
fi
ROOT_PYTHON="$(realpath "$ROOT_PYTHON")"

BUILD_DIR=$(realpath "$SCRIPT_DIR/../3rdparty/vllm")
if [[ -e "$BUILD_DIR" ]]; then
  echo "[ERROR] $BUILD_DIR already exists. Please remove or move it before running this script."
  exit 1
fi

echo "Building vLLM from:"
echo "  vLLM Git URL: $GIT_URL"
echo "  vLLM requested commit: $GIT_REF"
echo "  vLLM wheel location: $VLLM_PRECOMPILED_WHEEL_LOCATION"
echo "  vLLM version override: $VLLM_VERSION_OVERRIDE"

# Clone the repository
echo "Cloning repository..."
# When running inside Docker with --mount=type=ssh, the known_hosts file is empty.
# Skip host key verification for internal builds (only applies to SSH URLs).
GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  git clone --no-checkout "$GIT_URL" "$BUILD_DIR"
cd "$BUILD_DIR"
git checkout --detach "$GIT_REF"
RESOLVED_VLLM_COMMIT="$(git rev-parse HEAD)"
if [[ "$RESOLVED_VLLM_COMMIT" != "$GIT_REF" ]]; then
  echo "[ERROR] Requested vLLM commit $GIT_REF resolved to $RESOLVED_VLLM_COMMIT." >&2
  exit 1
fi
echo "  vLLM resolved commit: $RESOLVED_VLLM_COMMIT"

# Create a new Python environment using uv
echo "Creating Python environment..."
# Preserve caller environment selectors without letting them redirect custom vLLM installs.
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
VLLM_VENV="$BUILD_DIR/.venv"
VLLM_PYTHON="$VLLM_VENV/bin/python"
uv venv "$VLLM_VENV"

# Install dependencies
echo "Installing dependencies..."
uv pip install --python "$VLLM_PYTHON" --upgrade pip
uv pip install --python "$VLLM_PYTHON" numpy setuptools setuptools_scm

# Install vLLM using precompiled wheel
echo "Installing vLLM with precompiled wheel..."
uv pip install --python "$VLLM_PYTHON" --no-build-isolation -e .

echo "Build completed successfully!"
echo "The built vLLM is available in: $BUILD_DIR"

echo "Updating repo pyproject.toml to point vLLM to local clone..."

PYPROJECT_TOML="$REPO_ROOT/pyproject.toml"
if [[ ! -f "$PYPROJECT_TOML" ]]; then
  echo "[ERROR] pyproject.toml not found at $PYPROJECT_TOML. This script must be run from the repo root and pyproject.toml must exist."
  exit 1
fi

cd "$REPO_ROOT"

if [[ -n "$OLD_UV_PROJECT_ENVIRONMENT" ]]; then
  # Preserve an explicitly configured project environment for the root project.
  export UV_PROJECT_ENVIRONMENT="$OLD_UV_PROJECT_ENVIRONMENT"
else
  unset UV_PROJECT_ENVIRONMENT
fi
if [[ -n "$OLD_VIRTUAL_ENV" ]]; then
  export VIRTUAL_ENV="$OLD_VIRTUAL_ENV"
else
  unset VIRTUAL_ENV
fi

uv run --no-project --with packaging --with tomlkit \
  python tools/configure_custom_vllm.py "$PYPROJECT_TOML"

# Ensure build deps and re-lock
uv pip install --python "$ROOT_PYTHON" setuptools_scm
uv lock

# Write to a file that a docker build will use to set the necessary env vars
{
  printf 'export VLLM_GIT_URL=%q\n' "$GIT_URL"
  printf 'export VLLM_GIT_REF=%q\n' "$GIT_REF"
  printf 'export VLLM_GIT_COMMIT=%q\n' "$RESOLVED_VLLM_COMMIT"
  printf 'export VLLM_USE_PRECOMPILED=%q\n' "$VLLM_USE_PRECOMPILED"
  printf 'export VLLM_PRECOMPILED_WHEEL_LOCATION=%q\n' "$VLLM_PRECOMPILED_WHEEL_LOCATION"
  printf 'export VLLM_VERSION_OVERRIDE=%q\n' "$VLLM_VERSION_OVERRIDE"
} >"$BUILD_DIR/nemo-rl.env"

cat <<EOF
[INFO] pyproject.toml updated. NeMo RL is now configured to use the local vLLM at 3rdparty/vllm.
[INFO] Verify this new vllm version by running:

VLLM_USE_PRECOMPILED=$VLLM_USE_PRECOMPILED \\
VLLM_PRECOMPILED_WHEEL_LOCATION=$VLLM_PRECOMPILED_WHEEL_LOCATION \\
VLLM_VERSION_OVERRIDE=$VLLM_VERSION_OVERRIDE \\
  uv run --locked --extra vllm python -c \\
  'import flashinfer, vllm; print(vllm.__version__, flashinfer.__version__, vllm.__file__)'

[INFO] For more information on this custom install, visit https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/use-custom-vllm.md
[IMPORTANT] Remember to set the shell variable 'VLLM_PRECOMPILED_WHEEL_LOCATION' when running NeMo RL apps with this custom vLLM to avoid re-compiling.
EOF
