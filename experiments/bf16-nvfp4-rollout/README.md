# BF16 Training with NVFP4 Rollout

This experiment validates a plain BF16 Qwen3-30B-A3B Megatron policy with real
ModelOpt NVFP4 generation. It covers W4A16 and calibrated W4A4 through both the
legacy colocated IPC refit and non-colocated NCCL-Reshard refit paths.

The checked-in artifacts are static launch and validation definitions. No GPU
jobs have been submitted and no runtime result is claimed here.

## Smoke matrix

| Mode | `REFIT_TRANSPORT` | Placement | W&B name suffix |
| --- | --- | --- | --- |
| W4A16 | `null` | Two colocated 8-GPU B200 nodes | `-legacy` |
| W4A4 | `null` | Two colocated 8-GPU B200 nodes | `-legacy` |
| W4A16 | `nccl_reshard` | One 8-GPU train node and one 8-GPU generation node | `-nccl-reshard` |
| W4A4 | `nccl_reshard` | One 8-GPU train node and one 8-GPU generation node | `-nccl-reshard` |

The B200 launch maps the recipe's 16-GPU world from 4x4 to 2x8. The NCCL
variant splits that allocation into one 8-GPU training node and one 8-GPU
generation node, then overrides Megatron expert parallelism from 16 to 8. It
always sets `policy.generation.colocated.enabled=false` and uses application
segment size 1 for the one-node training cluster.

## Required environment

- `HF_HOME` and `HF_DATASETS_CACHE`: cluster-visible Hugging Face caches.
- `CONTAINER`, `ACCOUNT`, and `PARTITION=batch`: standard `tools/launch` inputs.
- `WANDB_PROJECT_OVERRIDE`: optional; defaults to `sna-bf16-nvfp4-rollout`.
- `NVFP4_CALIBRATION_ARTIFACT`: required by every W4A4 run. The path must be
  visible inside the container and point to a provenance-validated safetensors
  artifact.

Generate the W4A4 artifact with the standalone exporter before launching W4A4:

```bash
uv run --no-sync examples/modelopt/export_nvfp4_calibration.py \
  --model Qwen/Qwen3-30B-A3B \
  --model-revision <immutable-model-revision> \
  --quant-cfg examples/modelopt/quant_configs/nvfp4_experts.yaml \
  --dataset <calibration-dataset> \
  --sample-count <sample-count> \
  --sequence-length <sequence-length> \
  --seed <seed> \
  --output <cluster-visible-path>/qwen3-30ba3b-nvfp4-calibration.safetensors
```

Record the exact command, model revision, dataset parameters, file size, and
SHA256 in the future `report.md`. Do not substitute a dummy artifact.

## Launch commands

Run only from a committed revision. `tools/launch` creates the code snapshot
used by the job. The W4A4 examples include the artifact in `EXTRA_ENV` so the
snapshot continuation command remains reproducible.

```bash
# W4A16 legacy
EXTRA_ENV="REFIT_TRANSPORT=null" \
CONTAINER="$CONTAINER" ACCOUNT="$ACCOUNT" PARTITION=batch \
tools/launch tests/test_suites/llm/performance/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a16-rollout.sh

# W4A16 NCCL-Reshard
EXTRA_ENV="REFIT_TRANSPORT=nccl_reshard" \
CONTAINER="$CONTAINER" ACCOUNT="$ACCOUNT" PARTITION=batch \
tools/launch tests/test_suites/llm/performance/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a16-rollout.sh

# W4A4 legacy
EXTRA_ENV="REFIT_TRANSPORT=null NVFP4_CALIBRATION_ARTIFACT=$NVFP4_CALIBRATION_ARTIFACT" \
CONTAINER="$CONTAINER" ACCOUNT="$ACCOUNT" PARTITION=batch \
tools/launch tests/test_suites/llm/performance/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a4-rollout.sh

# W4A4 NCCL-Reshard
EXTRA_ENV="REFIT_TRANSPORT=nccl_reshard NVFP4_CALIBRATION_ARTIFACT=$NVFP4_CALIBRATION_ARTIFACT" \
CONTAINER="$CONTAINER" ACCOUNT="$ACCOUNT" PARTITION=batch \
tools/launch tests/test_suites/llm/performance/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a4-rollout.sh
```

Each script runs two steps with checkpointing disabled. Logs and metrics are
written beneath a transport-specific directory next to the script, and each
default W&B name includes the quantization mode and transport.

## Pass criteria

A run passes only when the script confirms the real ModelOpt worker and expected
W4A16 or W4A4 method, exactly two refit timing records, and exactly two GRPO
`train/loss` records. It rejects QARL/fake-quant selection, incomplete reloads,
manifest failures, NaN or infinity values, NCCL plan disagreement, and weight
refit exceptions.

Follow [PLAN.md](PLAN.md) for submission, monitoring, and result capture.
