# Qwen3.8-27B DFlash2 static smoke

This standalone smoke compares autoregressive vLLM serving with the published
Qwen3.8-27B DFlash2 draft. Both arms use the same pinned target revision, TP1
BF16 server settings, prompts, sampling parameters, seeds, concurrency, and
exactly 20 requests. The DFlash2 arm adds only `--speculative-config`.

This is static rollout only. Draft refit and online draft training remain
disabled and fail closed.

## Pinned contract

- Target: `Qwen/Qwen3.8-27B` at
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Draft: `incoai/Qwen3.8-27B-DFlash2` at
  `dedf8df68adfb1afeaf7b7480c0a0243108177b4`
- vLLM DFlash2 implementation: upstream merge commit
  `b389ac29465b33f9e9c534df221ea3c129e9793f` or a descendant
- vLLM speculative method: `dflash`, with seven speculative tokens
- Required runtime switch: `VLLM_USE_V2_MODEL_RUNNER=1`

The launcher checks the installed runtime before starting either arm. It
requires the upstream Qwen3 DFlash2 model and V2 speculator modules, so the
current NeMo-RL vLLM 0.25.1 pin is intentionally rejected. The contract was
verified against [vLLM PR #52816](https://github.com/vllm-project/vllm/pull/52816),
the [speculative decoding docs](https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/README.md),
and [SpeculativeConfig](https://github.com/vllm-project/vllm/blob/main/vllm/config/speculative.py).

## Container requirements

Use an immutable pyxis/enroot `.sqsh` containing a vLLM build from commit
`b389ac29465b33f9e9c534df221ea3c129e9793f` or a pinned descendant. It must
include Python 3.12, PyYAML, CUDA-compatible PyTorch, FlashInfer, and the CUDA
runtime and development headers required by FlashInfer JIT (including
`curand.h`). The host NVIDIA driver must support the image's CUDA version.
The smoke requests one H200/GB200-class GPU and serves at TP1.

Keep the checkout in `/home`, the immutable container and durable results in
`/lustre`, and caches and temporary files in `/raid/scratch`. `run.slurm`
enforces those path classes and mounts them into the container.

## Local validation

The manifests are launcher inputs, not `vllm serve --config` files. Inspect the
resolved commands without loading a model:

```bash
python research/qwen3_8_27b_dflash2_static/launch.py \
  --manifest research/qwen3_8_27b_dflash2_static/baseline.yaml \
  --output-dir /tmp/not-created --dry-run
python research/qwen3_8_27b_dflash2_static/launch.py \
  --manifest research/qwen3_8_27b_dflash2_static/dflash2.yaml \
  --output-dir /tmp/not-created --dry-run
```

Every completed run writes `summary.json` for machine aggregation and
`server.log`. The summary records the manifest hash and contents, exact server
command, runtime fingerprint, request-level latency and token counts,
container image, and SLURM job ID. Failures also write a JSON summary before
the launcher exits nonzero.

## Proposed SLURM submission

Do not submit from a dirty or unpushed checkout. After committing, pushing,
and pulling that commit on the cluster, set concrete paths and first validate
each arm with `--test-only`:

```bash
export DFLASH2_IMAGE=/lustre/<user>/containers/vllm-b389ac294.sqsh
export DFLASH2_REPO=/home/<user>/Nemo-RL
export DFLASH2_RESULTS=/lustre/<user>/results/qwen3_8_27b_dflash2_static
mkdir -p "${DFLASH2_RESULTS}"

sbatch --test-only --account=<ACCOUNT> --partition=<PARTITION> \
  --export=ALL,ARM=baseline,CONTAINER_IMAGE="${DFLASH2_IMAGE}",REPO_ROOT="${DFLASH2_REPO}",OUTPUT_ROOT="${DFLASH2_RESULTS}" \
  --output="${DFLASH2_RESULTS}/slurm-%x-%j.out" \
  --error="${DFLASH2_RESULTS}/slurm-%x-%j.err" \
  research/qwen3_8_27b_dflash2_static/run.slurm
sbatch --test-only --account=<ACCOUNT> --partition=<PARTITION> \
  --export=ALL,ARM=dflash2,CONTAINER_IMAGE="${DFLASH2_IMAGE}",REPO_ROOT="${DFLASH2_REPO}",OUTPUT_ROOT="${DFLASH2_RESULTS}" \
  --output="${DFLASH2_RESULTS}/slurm-%x-%j.out" \
  --error="${DFLASH2_RESULTS}/slurm-%x-%j.err" \
  research/qwen3_8_27b_dflash2_static/run.slurm
```

After both checks succeed, remove `--test-only` from those exact commands to
submit. `run.slurm` passes `--request-count 20`; the Python launcher separately
rejects any value outside 1 through 20. Per cluster policy, monitor both jobs
with one filtered scheduler query at intervals of at least 60 seconds for the
first five minutes.
