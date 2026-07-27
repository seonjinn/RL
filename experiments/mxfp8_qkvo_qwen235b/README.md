# Qwen3-235B-A22B MXFP8 QKVO A/B

This experiment extends the Qwen3-30B and Nemotron Nano QKVO comparison to
Qwen3-235B-A22B. It measures both the quantization-scope effect and the refit
optimization effect using five arms.

See [PLAN.md](PLAN.md) for the controlled setup and reporting criteria.

## Runtime

The suite targets Lyris:

- 16 nodes x 4 GB200 GPUs
- `coreai_dlalgo_llm` account
- `gb200` partition with `sharp` networking
- 4-hour walltime
- 20 GRPO steps
- no checkpoint saving

The validated 235B recipes use vLLM TP8 for BF16 and TP4 for MXFP8. Both
MoE-only and QKVO MXFP8 arms use TP4, so the primary quantization-scope A/B is
topology-matched.

The launcher replaces the recipe's HybridEP dispatcher with `alltoall` because
HybridEP is unavailable in the provisioned worker environment.

The canonical 235B test guards are retained: `NCCL_NVLS_ENABLE=0`,
`RAY_CGRAPH_get_timeout=2400`, and a 2400-second vLLM distributed timeout.
TensorBoard remains disabled as specified by the 235B parent recipe.

At submission, the container symlink is resolved to its immutable squashfs
filename and the exact repository SHA is passed to every job. A queued job
fails before allocation setup if the shared checkout moves to another commit.

The Lyris host's `api.wandb.ai` entry in `~/.netrc` is the canonical W&B
credential source. It overrides stale values loaded from `.nemo_rl_tokens`;
the driver verifies the credential before model initialization.

## Dry Run

```bash
BASE=/lustre/fsw/coreai_dlalgo_llm/users/sna \
ACTION=test-only \
MAX_STEPS=20 \
./experiments/mxfp8_qkvo_qwen235b/submit_suite.sh
```

## Submit

```bash
BASE=/lustre/fsw/coreai_dlalgo_llm/users/sna \
ACTION=submit \
MAX_STEPS=20 \
./experiments/mxfp8_qkvo_qwen235b/submit_suite.sh
```

Select a subset with a comma-separated arm filter:

```bash
ARM_FILTER=moe-optimized,qkvo-optimized \
ACTION=submit \
./experiments/mxfp8_qkvo_qwen235b/submit_suite.sh
```

Logs, manifests, metadata, and TensorBoard files are stored under:

```text
/lustre/fsw/coreai_dlalgo_llm/users/sna/experiments/mxfp8-qkvo-qwen235b
```

W&B project:

```text
nvidia/sna-mxfp8-qkvo-qwen235b
```

## GCP-NRT B200

The same 64-GPU matrix maps to 8 nodes x 8 B200 GPUs on GCP-NRT. The profile
sets `--gpus-per-node=8` and overrides NeMo-RL's cluster node, GPU, and segment
counts to 8.

Dry-run all five arms:

```bash
ACTION=test-only MAX_STEPS=20 \
  ./experiments/mxfp8_qkvo_qwen235b/submit_gcp_nrt.sh
```

Submit all five arms:

```bash
ACTION=submit MAX_STEPS=20 \
  ./experiments/mxfp8_qkvo_qwen235b/submit_gcp_nrt.sh
```

The GCP-NRT profile uses:

- account `coreai_chef_posttrain`
- partition `batch`
- 8 nodes x 8 B200 GPUs
- `/lustre:/lustre` container mount
- the local Qwen3-235B Hugging Face cache under `.cache/huggingface`
- a 120-minute GPU-idle reaper exemption for model loading and autotuning

GCP-NRT does not expose the Slurm `--segment` option. The application-side
`cluster.segment_size=8` matches prior completed Qwen3-235B jobs on the same
8-node B200 allocation.

The GCP-NRT profile skips recursive submodule initialization. The current
GCP-NRT Lustre is inode-constrained, while the pinned container already
provides the runtime dependencies; the sparse experiment checkout supplies the
NeMo-RL source, recipes, and launch files.
