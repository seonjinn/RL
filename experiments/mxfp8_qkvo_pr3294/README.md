# PR 3294 MXFP8 QKVO Refit A/B

This experiment measures the refit performance impact of NeMo-RL PR #3294
with the existing MoE-only quantization scope and with Q/K/V/O projections
included in MXFP8 quantization.

## Matrix

| Arm | Recipe | PR refit optimizations |
| --- | --- | --- |
| `moe-baseline` | standard MXFP8 rollout | off |
| `moe-optimized` | standard MXFP8 rollout | on |
| `qkvo-baseline` | QKVO-inclusive MXFP8 rollout | off |
| `qkvo-optimized` | QKVO-inclusive MXFP8 rollout | on |

All arms use Qwen3-30B-A3B, 4 nodes with 4 GB200 GPUs per node, GBS 2048,
seed 42, real importance sampling, vLLM TP1, sleep level 1, and a fixed 4 GiB
refit buffer. The pinned reference-policy swap stays disabled so the comparison
isolates the refit path.

The optimized arms enable trainer-side prequantization, persistent IPC staging
buffers, slim post-refit offload, batched MoE shuffle, and cached weight-loader
replay. The baseline arms explicitly disable all five paths on the same PR
commit.

The launcher runs the local checkout with the nightly container's prebuilt
Python and Ray environments. It prepends the checkout to `PYTHONPATH`, disables
worker-venv rebuilds, and verifies the Ray version and imported NeMo-RL source
before starting the driver. This prevents `uv run` from replacing the
container's Ray version with the older version pinned by the PR branch.

## Run

AWS-DFW:

```bash
ACTION=test-only MAX_STEPS=20 ./experiments/mxfp8_qkvo_pr3294/submit_suite.sh
ACTION=submit MAX_STEPS=20 ./experiments/mxfp8_qkvo_pr3294/submit_suite.sh
```

OCI-HSG:

```bash
BASE=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna \
SLURM_ACCOUNT=coreai_dlalgo_nemorl \
PARTITION=batch \
ACTION=test-only \
MAX_STEPS=20 \
./experiments/mxfp8_qkvo_pr3294/submit_suite.sh
```

Lyris:

```bash
BASE=/lustre/fsw/coreai_dlalgo_llm/users/sna \
SLURM_ACCOUNT=coreai_dlalgo_llm \
PARTITION=gb200 \
USE_GRES=0 \
SLURM_NETWORK=sharp \
CONTAINER_MOUNTS=/lustre:/lustre,/project:/project \
JOB_PREFIX=coreai_dlalgo_llm-mxfp8.pr3294 \
ACTION=test-only \
MAX_STEPS=20 \
./experiments/mxfp8_qkvo_pr3294/submit_suite.sh
```

Logs and manifests are written below:

```text
/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/sna/experiments/mxfp8-qkvo-pr3294-ab
```

The same relative result path is used below the selected `BASE` on other
clusters.
