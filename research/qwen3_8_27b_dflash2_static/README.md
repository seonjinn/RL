# Qwen3.8-27B DFlash2 static smoke

This standalone smoke compares autoregressive vLLM serving with the published
Qwen3.8-27B DFlash2 draft. Both arms use the same pinned target revision, TP1
BF16 server settings, prompts, sampling parameters, seeds, concurrency, and
exactly 20 requests. The DFlash2 arm adds only `--speculative-config`.

This is static rollout only. Draft refit and online draft training remain
disabled and fail closed.

The standalone comparison above is not a training smoke. The separate
`grpo.yaml` and `run_grpo.slurm` path runs NeMo-RL GRPO for exactly one or
twenty optimizer steps. It keeps the DFlash2 checkpoint static while NeMo-RL
refits only target-model weights after policy updates.

## Pinned contract

- Target: `Qwen/Qwen3.8-27B` at
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Draft: `incoai/Qwen3.8-27B-DFlash2` at
  `dedf8df68adfb1afeaf7b7480c0a0243108177b4`
- vLLM image: `docker.io/vllm/vllm-openai:nightly-f94666b60d4c58ec0807d22c837cfae322a1dde9`
- vLLM source commit: `f94666b60d4c58ec0807d22c837cfae322a1dde9`
- Multi-architecture index digest:
  `sha256:f50b406f696712019a673e317a0db6e029c430cf81ec7bdea2ebd7111e55aef7`
- Linux ARM64 manifest digest:
  `sha256:4db6d42b66ad393faa3da7341db580f443b7aeb9a7de5597cd11b724eabff6f6`
- DFlash2 merge ancestor:
  `b389ac29465b33f9e9c534df221ea3c129e9793f`
- vLLM speculative method: `dflash`, with seven speculative tokens
- Required runtime switch: `VLLM_USE_V2_MODEL_RUNNER=1`

The launcher checks the installed runtime and the exact staged image contract
before starting either arm. It requires the upstream Qwen3 DFlash2 model and
V2 speculator modules, so the current NeMo-RL vLLM 0.25.1 pin is intentionally
rejected. The source commit is a descendant of the DFlash2 merge commit. The
contract was verified against [vLLM PR #52816](https://github.com/vllm-project/vllm/pull/52816),
the [speculative decoding docs](https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/README.md),
and [SpeculativeConfig](https://github.com/vllm-project/vllm/blob/main/vllm/config/speculative.py).

## Container requirements

Use the concrete immutable pyxis/enroot `.sqsh` produced by
`stage_enroot_image.sbatch` from the exact image above. Do not pass its mutable
convenience symlink to `run.slurm`. The staged image must have its adjacent
`.metadata.txt` sidecar; runtime submission rejects a missing, extra-field, or
mismatched contract. The smoke requests one GB200-class ARM64 host and GPU and
serves at TP1.

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

Verify the published registry metadata without downloading the image:

```bash
python research/qwen3_8_27b_dflash2_static/image_contract.py verify-registry
```

The check requires the exact index digest, the published Linux ARM64 manifest
digest, both ARM64 image revision labels at the pinned source commit, and the
official vLLM source label.

## Proposed image staging

On an OCI-HSG ARM64 login/compute environment, first check scheduling. The
source tag and all digests are loaded from `image_contract.json`; supplying a
different `SOURCE_IMAGE` or digest fails before import.

```bash
export DFLASH2_CONTAINER_DIR=/lustre/<user>/containers
export DFLASH2_REPO=/home/<user>/Nemo-RL

sbatch --test-only --account=<ACCOUNT> --partition=<PARTITION> \
  --export=ALL,CONTAINER_DIR="${DFLASH2_CONTAINER_DIR}",REPO_ROOT="${DFLASH2_REPO}" \
  --output="${DFLASH2_CONTAINER_DIR}/stage-dflash2-%j.out" \
  --error="${DFLASH2_CONTAINER_DIR}/stage-dflash2-%j.err" \
  research/qwen3_8_27b_dflash2_static/stage_enroot_image.sbatch
```

After the test-only check, remove `--test-only` to stage. The job prints the
concrete output path, shaped as
`vllm-openai-nightly-f94666b_YYYYMMDD_JOBID.sqsh`, and creates an adjacent
validated metadata sidecar. Use that printed concrete path below.

## Proposed SLURM submission

Do not submit from a dirty or unpushed checkout. After committing, pushing,
and pulling that commit on the cluster, set concrete paths and first validate
each arm with `--test-only`:

```bash
export DFLASH2_IMAGE=/lustre/<user>/containers/vllm-openai-nightly-f94666b_YYYYMMDD_JOBID.sqsh
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

## NeMo-RL GRPO feasibility spike

The training path uses the target snapshot ending in
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` and the draft snapshot ending in
`dedf8df68adfb1afeaf7b7480c0a0243108177b4`. Both must be staged under
`/lustre`; repository or mutable model names are rejected by the launcher.
The recipe is text-only (`language_model_only: true`), uses four GPUs for both
DTensor policy training and TP4 vLLM generation, explicitly disables
`policy.draft`, and supplies the published drafter only through vLLM's static
`speculative_config`.

The official f946 vLLM image is sufficient for standalone serving but is not
assumed to include NeMo-RL, Ray, and Automodel. Before model loading, run the
one-GPU actor preflight against the candidate runtime image. It imports the
actual NeMo-RL `VllmGenerationWorker`, checks that Ray selects the current
system interpreter for that actor, imports the installed vLLM package, checks
both DFlash2 modules and V2 runner mode, and writes `runtime_fingerprint.json`.

The official image cannot be used as the NeMo-RL training image unchanged. Its
published filesystem uses Python 3.12 and Torch 2.13, while this NeMo-RL
checkout requires Python 3.13 and Torch 2.11; the image also does not include
Ray. A production training runtime must therefore rebuild vLLM at source
commit `f94666b60d4c58ec0807d22c837cfae322a1dde9` against the NeMo-RL locked
Python/Torch environment. Copying the official image's compiled extensions
into a NeMo-RL nightly is not a supported compatibility shortcut. The actor
preflight intentionally rejects either unmodified image.

```bash
export DFLASH2_RUNTIME_IMAGE=/lustre/<user>/containers/<nemo-rl-plus-f946>.sqsh
export DFLASH2_REPO=/home/<user>/RL-dflash2
export DFLASH2_RUNTIME_RESULTS=/lustre/<user>/results/dflash2-runtime-preflight

sbatch --test-only --account=<ACCOUNT> --partition=<PARTITION> \
  --export=ALL,CONTAINER_IMAGE="${DFLASH2_RUNTIME_IMAGE}",REPO_ROOT="${DFLASH2_REPO}",OUTPUT_ROOT="${DFLASH2_RUNTIME_RESULTS}" \
  --output="${DFLASH2_RUNTIME_RESULTS}/slurm-%x-%j.out" \
  --error="${DFLASH2_RUNTIME_RESULTS}/slurm-%x-%j.err" \
  research/qwen3_8_27b_dflash2_static/runtime_preflight.slurm
```

Only after that preflight passes, submit the one-step smoke. `SMOKE_STEPS`
accepts only `1` or `20`; begin with `1`, then use the same command with `20`
only after the one-step run exits successfully.

```bash
export DFLASH2_TARGET=/lustre/<user>/models/Qwen3.8-27B/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
export DFLASH2_DRAFT=/lustre/<user>/models/Qwen3.8-27B-DFlash2/dedf8df68adfb1afeaf7b7480c0a0243108177b4
export DFLASH2_GRPO_RESULTS=/lustre/<user>/results/dflash2-grpo

sbatch --test-only --account=<ACCOUNT> --partition=<PARTITION> \
  --export=ALL,CONTAINER_IMAGE="${DFLASH2_RUNTIME_IMAGE}",REPO_ROOT="${DFLASH2_REPO}",TARGET_SNAPSHOT="${DFLASH2_TARGET}",DRAFT_SNAPSHOT="${DFLASH2_DRAFT}",OUTPUT_ROOT="${DFLASH2_GRPO_RESULTS}",SMOKE_STEPS=1 \
  --output="${DFLASH2_GRPO_RESULTS}/slurm-%x-%j.out" \
  --error="${DFLASH2_GRPO_RESULTS}/slurm-%x-%j.err" \
  research/qwen3_8_27b_dflash2_static/run_grpo.slurm
```

Every training attempt writes `training_summary.json` with its kind set to
`nemo_rl_grpo_training`, the requested optimizer-step count, both revisions,
the exact `examples/run_grpo.py` command, and final return code. This output is
separate from the standalone launcher's 20-request `summary.json`.
