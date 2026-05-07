# Build the Super Omni VLM Container

This is the working guide for building a NeMo RL Super container. You can build it on a desktop or a Colossus machine.

Before you start, put this in `~/.bashrc`:

```bash
export CI_JOB_TOKEN="<your GitLab job token>"
export USERNAME="<your NVIDIA alias>"
```

## Step 1: Clone the right branch

It's best to start by doing a fresh clone:

```bash
git clone --branch YOUR_BRANCH --single-branch --recurse-submodules git@gitlab-master.nvidia.com:jseppanen/nemo-rl.git nemo-rl-super
```
Run the next steps inside the `nemo-rl-super` folder.

## Step 2: Build vLLM
Make sure this script already has the right vLLM version baked in. If not, ask your agent to change it to the version you want.

```bash
bash tools/build-custom-vllm.sh
```

## Step 3: Build the Docker image

```bash
COMMIT_PIN="$(git rev-parse --short HEAD)"
CONTAINER_NAME="super-omni-$(date +%Y%m%d)-${COMMIT_PIN}"

docker buildx build \
  --progress=plain \
  --build-context nemo-rl=. \
  --target release \
  -f docker/Dockerfile \
  --build-arg SKIP_SGLANG_BUILD=1 \
  --build-arg NEMO_GYM_PREFETCH_CONFIGS="" \
  --build-arg MAX_JOBS=16 \
  --build-arg NVTE_BUILD_THREADS_PER_JOB=1 \
  --secret id=CI_JOB_TOKEN,env=CI_JOB_TOKEN \
  --tag "nemo-rl:${CONTAINER_NAME}" \
  --load .
```

## Step 4: Run a smoke test

```bash
docker run --rm -it nemo-rl:${CONTAINER_NAME} bash -lc "$(cat tools/postbuild_container_smoke.sh)"
```

## Step 5: Export the Enroot image and rsync it
Make sure to change the destination to where you want the image to be synced.

```bash
IMAGE_TAG="nemo-rl:${CONTAINER_NAME}"
OUT="/tmp/local-aroshanghias/containers/${CONTAINER_NAME}.sqsh"

mkdir -p "$(dirname "$OUT")"
enroot import -o "$OUT" "dockerd://${IMAGE_TAG}"

ls -lh "$OUT"

rsync -avhP "$OUT" $USERNAME@cw-dfw-cs-001-vscode-01:/lustre/fs1/portfolios/coreai/users/aroshanghias/containers/
```

## Troubleshooting Notes

- `401 Unauthorized` during `uv lock` means `CI_JOB_TOKEN` was missing or
invalid for the private FlashInfer package index.
- `git upload-pack: not our ref` during submodule update after pulling means
the superproject changed `.gitmodules` URLs. Pull with
`submodule.recurse=false`, run `git submodule sync --recursive`, update the
top-level Megatron-LM and Megatron-Bridge submodules, then run the recursive
update.
- `Dependency mismatch between Megatron-LM-workspace/Megatron-LM/pyproject.toml vs Megatron-LM-workspace/setup.py::CACHED_DEPENDENCIES` means the wrapper
dependency cache is stale for the pinned Megatron-LM submodule. Sync
`CACHED_DEPENDENCIES` to the submodule's default plus `dev` dependencies.
- `Because megatron-core depends on av<16.0.0 and nemo-rl depends on av>=17.0.0` means the current branch needs the top-level `av>=17.0.0` override
in `[tool.uv].override-dependencies`, followed by a lockfile refresh.
- `The lockfile at uv.lock needs to be updated, but --locked was provided`
after dependency metadata changes means `uv.lock` must be regenerated. If the
host lacks CUDA and vLLM metadata fails with `CUDA_HOME is not set`, regenerate
the lock in the Dockerfile's CUDA base image as shown in section 2.
- `3rdparty/vllm already exists` means the vendoring script will not overwrite
an existing checkout. Remove it only when intentionally retrying the vLLM
setup.
- The Docker build still needs a `CI_JOB_TOKEN` BuildKit secret because the
hermetic stage also resolves private FlashInfer packages. Use `env=CI_JOB_TOKEN`
for an exported environment variable or `src="$HOME/secrets"` for a token file.
- `SKIP_SGLANG_BUILD=1` is intentional for this image path.
- FlashInfer cubin predownload is intentionally quiet in `docker/Dockerfile`.
On success it suppresses thousands of per-file download logs; on failure it
prints the last 100 log lines.
- If the Docker log ever prints the token value, rotate the token and ensure
the Dockerfile disables xtrace while reading `/run/secrets/CI_JOB_TOKEN`.
