# BF16 Training with NVFP4 Rollout

This experiment validates a plain BF16 Qwen3-30B-A3B Megatron policy with real
ModelOpt NVFP4 generation. It covers W4A16 and calibrated W4A4 through both the
legacy colocated IPC refit and non-colocated NCCL-Reshard refit paths.

The W4A16 and calibrated W4A4 legacy and NCCL-Reshard two-step smokes completed
on GCP-NRT. See `REPORT.md` for job IDs, W&B links, metrics, and artifact
provenance.

For GCP-NRT, use `submit_gcp_nrt.sh`. It creates a unique committed code
snapshot, uses `--gpus-per-node=8`, and supports scheduler preflight with
`ACTION=test-only` before `ACTION=submit`.

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
- `MOUNTS=/lustre:/lustre`: required so model caches and calibration artifacts
  remain visible after `ray.sub` disables the container home mount.
- `WANDB_API_KEY`: required because the smoke scripts force W&B online.
- `CODE_SNAPSHOT_DIRNAME`: set to a new commit/run-specific directory for each
  campaign. Reusing a snapshot can reuse stale code and completed metrics.
- `WANDB_PROJECT_OVERRIDE`: optional; defaults to `sna-bf16-nvfp4-rollout`.
- `NVFP4_CALIBRATION_ARTIFACT`: required by every W4A4 run. The path must be
  visible inside the container and point to a provenance-validated safetensors
  artifact.

First create a fresh W4A4 snapshot with `DRYRUN=2`. Generate the artifact from
that exact snapshot because its metadata records the resolved absolute quant
config path, which the rollout validates.

```bash
export QWEN_REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
export CAMPAIGN="$(git rev-parse --short HEAD)-$(date +%Y%m%d-%H%M%S)"
export CODE_SNAPSHOT_DIRNAME="code_snapshots_nvfp4/${CAMPAIGN}"

DRYRUN=2 MOUNTS=/lustre:/lustre \
  tools/launch \
  tests/test_suites/llm/performance/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a4-rollout.sh

export W4A4_SNAPSHOT="$PWD/$CODE_SNAPSHOT_DIRNAME/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a4-rollout"
export NVFP4_CALIBRATION_ARTIFACT="$W4A4_SNAPSHOT/artifacts/qwen3-30ba3b-w4a4.safetensors"
mkdir -p "$(dirname "$NVFP4_CALIBRATION_ARTIFACT")"

cd "$W4A4_SNAPSHOT"
uv run --no-sync examples/modelopt/export_nvfp4_calibration.py \
  --model Qwen/Qwen3-30B-A3B \
  --model-revision "$QWEN_REV" \
  --quant-cfg "$W4A4_SNAPSHOT/examples/modelopt/quant_configs/nvfp4_experts.yaml" \
  --dataset cnn_dailymail \
  --sample-count 16 \
  --sequence-length 512 \
  --seed 42 \
  --output "$NVFP4_CALIBRATION_ARTIFACT"

stat -c '%n %s bytes' "$NVFP4_CALIBRATION_ARTIFACT"
sha256sum "$NVFP4_CALIBRATION_ARTIFACT"
```

Record the exact command, model revision, dataset parameters, file size, and
SHA256 in the future `report.md`. Do not substitute a dummy artifact.

## Launch commands

Run only from a committed revision. Export the common launch inputs and a fresh
campaign snapshot first:

```bash
export HF_HOME=<cluster-visible-hf-cache>
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export CONTAINER=<nemo-rl-sqsh>
export ACCOUNT=<slurm-account>
export PARTITION=batch
export MOUNTS=/lustre:/lustre
export WANDB_API_KEY=<key>
export CODE_SNAPSHOT_DIRNAME="code_snapshots_nvfp4/$(git rev-parse --short HEAD)-$(date +%Y%m%d-%H%M%S)"
```

Use `DRYRUN=2` first and inspect the generated `continue.sh`. The W4A4 run must
reuse the same snapshot in which its artifact was generated.

```bash
# W4A16 legacy
SCHEDULER_SEGMENT_SIZE=2 \
EXTRA_ENV="REFIT_TRANSPORT=null SCHEDULER_SEGMENT_SIZE=2" \
tools/launch tests/test_suites/llm/performance/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a16-rollout.sh

# W4A16 NCCL-Reshard
SCHEDULER_SEGMENT_SIZE=1 \
EXTRA_ENV="REFIT_TRANSPORT=nccl_reshard SCHEDULER_SEGMENT_SIZE=1" \
tools/launch tests/test_suites/llm/performance/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a16-rollout.sh

# W4A4 legacy
SCHEDULER_SEGMENT_SIZE=2 \
EXTRA_ENV="REFIT_TRANSPORT=null SCHEDULER_SEGMENT_SIZE=2 NVFP4_CALIBRATION_ARTIFACT=$NVFP4_CALIBRATION_ARTIFACT" \
tools/launch tests/test_suites/llm/performance/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a4-rollout.sh

# W4A4 NCCL-Reshard
SCHEDULER_SEGMENT_SIZE=1 \
EXTRA_ENV="REFIT_TRANSPORT=nccl_reshard SCHEDULER_SEGMENT_SIZE=1 NVFP4_CALIBRATION_ARTIFACT=$NVFP4_CALIBRATION_ARTIFACT" \
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
