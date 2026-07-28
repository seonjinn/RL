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
if [[ ! "$GIT_REF" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "[ERROR] GIT_REF must be an immutable 40-character Git commit." >&2
  exit 2
fi

export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_LOCATION
export VLLM_VERSION_OVERRIDE=0.20.2

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
if [[ "${RESOLVED_VLLM_COMMIT,,}" != "${GIT_REF,,}" ]]; then
  echo "[ERROR] Requested vLLM commit $GIT_REF resolved to $RESOLVED_VLLM_COMMIT." >&2
  exit 1
fi
echo "  vLLM resolved commit: $RESOLVED_VLLM_COMMIT"

# Create a new Python environment using uv
echo "Creating Python environment..."
# Pop the project environment set by user to not interfere with the one we create for the vllm repo
OLD_UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-}"
unset UV_PROJECT_ENVIRONMENT
uv venv

# Install dependencies
echo "Installing dependencies..."
uv pip install --upgrade pip
uv pip install numpy setuptools setuptools_scm

# Install vLLM using precompiled wheel
echo "Installing vLLM with precompiled wheel..."
uv pip install --no-build-isolation -e .

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
  export VIRTUAL_ENV="$OLD_UV_PROJECT_ENVIRONMENT"
else
  unset UV_PROJECT_ENVIRONMENT
fi

uv run --no-project --with packaging --with tomlkit \
  python tools/configure_custom_vllm.py "$PYPROJECT_TOML"

# Ensure build deps and re-lock
uv pip install setuptools_scm
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
