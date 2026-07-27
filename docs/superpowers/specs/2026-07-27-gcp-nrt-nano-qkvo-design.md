# GCP-NRT Nano QKVO Experiment Design

## Goal

Run a matched five-arm Nemotron 3 Nano GRPO suite on GCP-NRT B200 and preserve
the existing Lyris 16-GPU experiment semantics.

## Experiment Matrix

| Arm | Rollout precision | Quantization scope | Refit optimization |
| --- | --- | --- | --- |
| `bf16` | BF16 | none | off |
| `moe-baseline` | MXFP8 | MoE only | off |
| `moe-optimized` | MXFP8 | MoE only | on |
| `qkvo-baseline` | MXFP8 | MoE and QKVO | off |
| `qkvo-optimized` | MXFP8 | MoE and QKVO | on |

All arms use 20 GRPO steps, real importance sampling, global batch size 16,
seed 42, vLLM TP1, trainer TP2/PP2/CP2/EP8, and checkpointing disabled.

## GCP-NRT Mapping

- Hardware: 2 B200 nodes with 8 GPUs per node, preserving the 16-GPU total
  used by the Lyris 4-node by 4-GPU measurements.
- Scheduler: `batch`, `--gpus-per-node=8`, four-hour wall time, no Slurm
  `--segment` or `--network` option.
- Idle policy: 120-minute OccupiedIdleGPUsJobReaper exemption for model load
  and MXFP8 autotuning.
- Container, model, and source checkout are existing Lustre artifacts. The
  submit path does not initialize submodules because the GCP filesystem is
  nearly out of inodes.
- W&B project: `nvidia/sna-mxfp8-qkvo-nano-gcp-nrt`.

## Launcher Design

The shared launcher validates exactly 16 allocated GPUs while accepting either
4x4 or 2x8 physical topology. A small `submit_gcp_nrt.sh` profile supplies the
GCP paths and scheduler options. Each submitted job receives the expected
repository SHA and exits before launch if the checkout changes after
submission.

The application still receives the full allocation as
`cluster.segment_size`; only the unsupported Slurm `--segment` option is
omitted on GCP-NRT.

## Validation

Local tests use fake `git`, `readlink`, and `sbatch` executables to inspect the
exact scheduler request without submitting jobs. Remote validation runs
`ACTION=test-only` before the five jobs are submitted. The submitted jobs are
monitored for at least five minutes for allocation shape, startup failures,
W&B initialization, tracebacks, and OOMs.
