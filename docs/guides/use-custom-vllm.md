# Experiment with Custom vLLM

This guide explains how to use an immutable custom vLLM commit with an
ABI-matching precompiled wheel. The Git commit and wheel URL are required
inputs: moving branches and automatically discovered nightly wheels are not
reproducible build provenance.

## Clone and Build Your Custom vLLM

The MXFP8 adaptive rollout integration is developed from:

```sh
VLLM_URL=https://github.com/seonjinn/vllm.git
VLLM_BRANCH=sna/mxfp8-adaptive-v0.20.2-nemorl
git ls-remote "$VLLM_URL" "refs/heads/$VLLM_BRANCH"
```

The approved immutable build commit for this integration is
`77d5e10eec8f5cc217d16a9230f2955cf8553cee`. The branch may advance beyond
that commit, so use `ls-remote` only to confirm the published branch and do not
pass the branch name or its newer head to the build script.

The following wheel is the vLLM 0.20.2 Linux aarch64 release wheel used by the
GB200 CUDA 13.0 build. For another platform, supply a vLLM 0.20.2 wheel built
for the same Python, PyTorch, CUDA, architecture, and C++ ABI as NeMo RL.

```sh
VLLM_COMMIT=77d5e10eec8f5cc217d16a9230f2955cf8553cee
VLLM_WHEEL=https://github.com/vllm-project/vllm/releases/download/v0.20.2/vllm-0.20.2-cp38-abi3-manylinux_2_35_aarch64.whl

bash tools/build-custom-vllm.sh \
  "$VLLM_URL" \
  "$VLLM_COMMIT" \
  "$VLLM_WHEEL"
```

This script does the following:

1. Clones and checks out the exact vLLM commit in detached-HEAD mode.
2. Builds the editable source with `VLLM_USE_PRECOMPILED=1`,
   `VLLM_PRECOMPILED_WHEEL_LOCATION`, and
   `VLLM_VERSION_OVERRIDE=0.20.2`.
3. Replaces all vLLM requirements with one editable
   `3rdparty/vllm` source.
4. Regenerates `uv.lock`.
5. Records the URL, requested commit, resolved commit, wheel URL, and build
   environment in `3rdparty/vllm/nemo-rl.env`.

Commit the updated `pyproject.toml` and `uv.lock` together so other builds use
the same dependency graph.

## Verify Your Custom vLLM in Isolation

Verify the immutable checkout, lock, versions, and import source:

```sh
test "$(git -C 3rdparty/vllm rev-parse HEAD)" = "$VLLM_COMMIT"
uv lock --check
source 3rdparty/vllm/nemo-rl.env
uv run --locked --extra vllm python -c \
  'import flashinfer, vllm; print(vllm.__version__, flashinfer.__version__, vllm.__file__)'
```

The output must report vLLM `0.20.2`, FlashInfer `0.6.8.post1`, and a vLLM
module path below `3rdparty/vllm`.

## Running NeMo RL Apps with Custom vLLM

Source the recorded build environment before running a NeMo RL application:

```sh
source 3rdparty/vllm/nemo-rl.env
export NRL_FORCE_REBUILD_VENVS=true
uv pip install setuptools_scm
uv run examples/run_grpo.py
```

## Re-building the NeMo RL Docker Image

Using a custom vllm may require you to rebuild the docker image. The two most common reasons are:

1. The `ray` version was changed, so you **must** rebuild the image to allow `ray.sub` to start the ray cluster with the same version as the application.
2. Many dependencies changed and add a large overhead when `NRL_FORCE_REBUILD_VENVS=true` is set to rebuild venvs, so you wish to cache the dependencies in the image to avoid re-build/re-pulling wheels.

The Docker build requires the same three explicit dependency inputs. It creates
`3rdparty/vllm` and sources `3rdparty/vllm/nemo-rl.env` automatically.

```sh
docker buildx build \
  --build-arg BUILD_CUSTOM_VLLM=1 \
  --build-arg BUILD_CUSTOM_VLLM_URL="$VLLM_URL" \
  --build-arg BUILD_CUSTOM_VLLM_REF="$VLLM_COMMIT" \
  --build-arg BUILD_CUSTOM_VLLM_PRECOMPILED_WHEEL_LOCATION="$VLLM_WHEEL" \
  --target release \
  --build-context nemo-rl=. \
  -f docker/Dockerfile \
  --tag <registry>/nemo-rl:latest \
  --push \
  .
```

After the image is built, verify the recorded commit and installed packages:

```sh
docker run --rm <registry>/nemo-rl:latest bash -lc \
  'source 3rdparty/vllm/nemo-rl.env &&
   test "$(git -C 3rdparty/vllm rev-parse HEAD)" = "$VLLM_GIT_COMMIT" &&
   python -c "import flashinfer, vllm; print(vllm.__version__, flashinfer.__version__, vllm.__file__)"'
```

### SSH Setup for Private Repositories

If your custom vLLM is hosted in a **private repository** (e.g., internal GitLab), you need to set up SSH agent forwarding for Docker to clone it during the build.

#### Prerequisites
1. Your SSH key must be registered on the Git server (GitLab/GitHub)
2. The key must **not be expired** - check your Git server's SSH key settings
3. The key must be loaded into your local ssh-agent

#### Step 1: Verify your SSH key works

```sh
# For GitLab (adjust host/port as needed)
ssh -T git@gitlab.example.com -p 12051

# You should see: "Welcome to GitLab, @username!"
# If you see "Your SSH key has expired", renew it on the server
```

#### Step 2: Load your SSH key into the agent

```sh
# Check if an ssh-agent is already running
echo $SSH_AUTH_SOCK

# If empty, start one (this also sets SSH_AUTH_SOCK which `docker buildx` expects to be set when using `--ssh default`)
eval "$(ssh-agent -s)"

# Clear any old/expired keys from the agent
ssh-add -D

# Add your SSH key (use the key registered on your Git server)
ssh-add ~/.ssh/id_ed25519

# Verify it's loaded
ssh-add -l
```

#### Step 3: Run the Docker build with SSH forwarding

```sh
docker buildx build \
  --build-arg BUILD_CUSTOM_VLLM=1 \
  --build-arg BUILD_CUSTOM_VLLM_URL="$VLLM_URL" \
  --build-arg BUILD_CUSTOM_VLLM_REF="$VLLM_COMMIT" \
  --build-arg BUILD_CUSTOM_VLLM_PRECOMPILED_WHEEL_LOCATION="$VLLM_WHEEL" \
  --target release \
  --build-context nemo-rl=. \
  -f docker/Dockerfile \
  --ssh default \
  --tag <registry>/nemo-rl:latest \
  --push \
  .
```

## Running Applications with a Custom vLLM Container

When using a container built with custom vLLM, **use the frozen environment workflow** (bare `python`) instead of `uv run` with `NRL_FORCE_REBUILD_VENVS=true`.

```sh
# Recommended: use bare python (frozen environment)
python examples/run_grpo.py

# NOT recommended with custom vLLM containers:
# uv run examples/run_grpo.py
# or
# NRL_FORCE_REBUILD_VENVS=true uv run examples/run_grpo.py
```

### Why Not Use `uv run` or Rebuild Venvs?

Rebuilding worker virtual environments (via `uv run` or `NRL_FORCE_REBUILD_VENVS=true`) requires having the custom vLLM compiled locally. However, compiling vLLM requires a container environment with the correct CUDA toolchain—creating a chicken-and-egg problem.

The container already has vLLM built and cached in the frozen environments. Using bare `python` leverages these pre-built environments directly, avoiding the need to recompile vLLM at runtime.

> [!TIP]
> For more details on frozen environments and how they differ from `uv run`, see the [Dependency Management](../design-docs/dependency-management.md#frozen-environments) documentation.
