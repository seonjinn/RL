# Build the Super Ultra RL v0.17 Container

This note records the working path for building a NeMo RL Super container with
the vendored Ultra RL v0.17 vLLM checkout and Omni media dependencies.

The expected result is a Docker image like:

```bash
nemo-rl:super-ultra-rl-v017-<date>-<commit>-omni-media
```

and an Enroot image like:

```bash
/lustre/fs1/portfolios/coreai/users/aroshanghias/containers/nemo-rl-super-ultra-rl-v017-<date>-<commit>-omni-media.sqsh
```

## Source Setup

Use the branch that already vendors `3rdparty/vllm`.

```bash
cd /lustre/fs1/portfolios/coreai/users/aroshanghias/nemo-rl-super
```

Add the Omni media dependencies to the main project dependencies:

```bash
uv add librosa soundfile av
```

If the Megatron-Bridge submodule pin was updated, keep
`3rdparty/Megatron-Bridge-workspace/setup.py::CACHED_DEPENDENCIES` in sync with
`3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/pyproject.toml`. The working
dependency list for this build was:

```python
CACHED_DEPENDENCIES = [
    "transformers>=4.57.1",
    "datasets",
    "omegaconf>=2.3.0",
    "tensorboard>=2.19.0",
    "typing-extensions",
    "rich",
    "wandb>=0.19.10",
    "six>=1.17.0",
    "regex>=2024.11.6",
    "pyyaml>=6.0.2",
    "tqdm>=4.67.1",
    "hydra-core>1.3,<=1.3.2",
    "megatron-core[dev,mlm]>=0.15.0a0,<0.17.0",
    "qwen-vl-utils",
    "transformer-engine[pytorch]>=2.9.0a0,<2.10.0",
    "mamba-ssm",
    "nvidia-resiliency-ext",
    "causal-conv1d",
    "flash-linear-attention",
    "timm",
    "open-clip-torch>=3.2.0",
]
```

Refresh the lockfile:

```bash
uv lock
```

## Dockerfile Setup

In `docker/Dockerfile`, the `hermetic` stage needs access to the vLLM
precompiled wheel location before `uv sync` runs. Under the existing hermetic
stage build args:

```dockerfile
ARG BUILD_CUSTOM_VLLM_PRECOMPILED_WHEEL_LOCATION
ENV VLLM_PRECOMPILED_WHEEL_LOCATION=${BUILD_CUSTOM_VLLM_PRECOMPILED_WHEEL_LOCATION}
```

In the same `RUN` block, install the build group plus the vLLM metadata build
helpers immediately after `uv venv --seed`:

```dockerfile
uv venv --seed
uv pip install --python /opt/nemo_rl_venv --group build setuptools-scm numpy
```

## Build the Docker Image

Use the vendored `3rdparty/vllm` checkout. Do not clone vLLM during the Docker
build.

```bash
TAG="$(date +%Y%m%d)-$(git rev-parse --short HEAD)-omni-media"

docker buildx build \
  --progress=plain \
  --build-context nemo-rl=. \
  --target release \
  -f docker/Dockerfile \
  --build-arg BUILD_CUSTOM_VLLM_PRECOMPILED_WHEEL_LOCATION="https://github.com/vllm-project/vllm/releases/download/v0.17.0/vllm-0.17.0-cp38-abi3-manylinux_2_31_x86_64.whl" \
  --build-arg SKIP_SGLANG_BUILD=1 \
  --build-arg NEMO_GYM_PREFETCH_CONFIGS="" \
  --build-arg MAX_JOBS=1 \
  --build-arg NVTE_BUILD_THREADS_PER_JOB=1 \
  --secret id=CI_JOB_TOKEN,env=CI_JOB_TOKEN \
  --tag "nemo-rl:super-ultra-rl-v017-${TAG}" \
  --load .
```

Confirm the image tag:

```bash
docker images | grep 'nemo-rl.*super-ultra-rl-v017'
```

## Export the `.sqsh`

Convert the loaded Docker image to Enroot format:

```bash
IMAGE_TAG="nemo-rl:super-ultra-rl-v017-${TAG}"
OUT="/lustre/fs1/portfolios/coreai/users/aroshanghias/containers/nemo-rl-super-ultra-rl-v017-${TAG}.sqsh"

mkdir -p "$(dirname "$OUT")"
enroot import -o "$OUT" "dockerd://${IMAGE_TAG}"

ls -lh "$OUT"
```

## Validate the Container

Run a quick import probe through Slurm:

```bash
srun -A coreai_dlalgo_nemorl -p batch_short -N 1 --time=00:10:00 \
  --ntasks-per-node=1 --gres=gpu:1 --no-container-mount-home \
  --container-image "$OUT" \
  --container-mounts "/lustre/fsw/:/lustre/fsw/,/lustre/fs1:/lustre/fs1" \
  bash -lc '
    cd /lustre/fs1/portfolios/coreai/users/aroshanghias/nemo-rl-super
    export PYTHONPATH=/lustre/fs1/portfolios/coreai/users/aroshanghias/nemo-rl-super:/lustre/fs1/portfolios/coreai/users/aroshanghias/nemo-rl-super/3rdparty/vllm${PYTHONPATH:+:$PYTHONPATH}
    python - <<'"'"'PY'"'"'
import importlib.util
for name in ["vllm", "cbor2", "librosa", "soundfile", "av", "soxr", "audioread"]:
    spec = importlib.util.find_spec(name)
    print(f"{name:10s}", "FOUND" if spec else "MISSING", getattr(spec, "origin", None))
PY
  '
```

Then run the Gate 6 image smoke:

```bash
srun -A coreai_dlalgo_nemorl -p batch_short -N 1 --time=00:30:00 \
  -J nrl-gate6-smoke --ntasks-per-node=1 --gres=gpu:1 \
  --no-container-mount-home \
  --container-image "$OUT" \
  --container-mounts "/lustre/fsw/:/lustre/fsw/,/lustre/fs1:/lustre/fs1" \
  bash -lc '
    cd /lustre/fs1/portfolios/coreai/users/aroshanghias/nemo-rl-super
    export NRL_IGNORE_VERSION_MISMATCH=1
    export HF_HOME=/lustre/fs1/portfolios/coreai/users/aroshanghias/tmp/hf-home-gate6-runtime-ultra-rl-v017
    export PYTHONPATH=/lustre/fs1/portfolios/coreai/users/aroshanghias/nemo-rl-super:/lustre/fs1/portfolios/coreai/users/aroshanghias/nemo-rl-super/3rdparty/vllm${PYTHONPATH:+:$PYTHONPATH}
    python gate6_runtime_smoke.py 2>&1
  '
```

A successful smoke prints `SMOKE_OK`.
