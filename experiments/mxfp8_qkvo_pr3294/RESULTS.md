# Qwen3-30B-A3B MXFP8 QKVO Results

## Methodology

- Hardware: 4 Lyris nodes, 4 GB200 GPUs per node
- Workload: synchronous GRPO, 20 steps, GBS 2048, real importance sampling
- Parallelism: trainer TP1/EP16; vLLM TP1
- Scope: standard MXFP8 ignores q/k/v/o; QKVO sets the ignored list to empty
- Aggregation: arithmetic mean over W&B steps 3-19 inclusive
- Software: exact commits, submodules, and immutable container are recorded in
  `PROVENANCE.md`

BF16 uses the recipe's Triton MoE path, while MXFP8 uses the supported ModelOpt
FlashInfer path. Generated sequence lengths vary across runs, so throughput is
the primary cross-precision metric. All main metrics contain 17 samples;
transfer/update contains 16 logged samples within the same requested window.

## Step Time

All values are seconds per step.

| Arm | E2E | Generation | Logprob | Policy training | Refit | Transfer/update |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 196.18 | 70.34 | 37.77 | 79.99 | 3.97 | 1.30 |
| MoE baseline | 187.18 | 54.10 | 37.42 | 78.79 | 12.77 | 10.57 |
| MoE optimized | 182.12 | 53.90 | 37.49 | 80.19 | 6.77 | 4.35 |
| QKVO baseline | 189.43 | 55.05 | 37.56 | 79.90 | 12.77 | 10.67 |
| QKVO optimized | 182.54 | 55.59 | 37.23 | 79.29 | 6.75 | 4.34 |

## Throughput

All values are tokens/s/GPU.

| Arm | E2E | Generation | Logprob | Policy training |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 2123.70 | 5916.80 | 11018.03 | 5202.95 |
| MoE baseline | 2219.94 | 7675.97 | 11088.26 | 5264.12 |
| MoE optimized | 2280.67 | 7710.10 | 11076.12 | 5175.38 |
| QKVO baseline | 2200.95 | 7571.37 | 11082.49 | 5208.87 |
| QKVO optimized | 2281.41 | 7490.94 | 11178.98 | 5246.38 |

## Findings

- The refit optimization reduced total refit time by 47.0% for standard MXFP8
  and 47.2% for QKVO-inclusive MXFP8.
- Transfer/update time fell by 58.9% for standard MXFP8 and 59.3% for
  QKVO-inclusive MXFP8.
- Optimized standard MXFP8 improved generation throughput by 30.3% and E2E
  throughput by 7.4% relative to BF16.
- Optimized QKVO improved generation throughput by 26.6% and E2E throughput by
  7.4% relative to BF16.
- Adding QKVO to optimized MXFP8 changed generation throughput by -2.84% and
  E2E throughput by +0.03% relative to optimized standard MXFP8. This is not a
  speedup in these point estimates.

The result is consistent with QKVO projections being small compared with the
expert weights and their attention-projection shapes not benefiting enough
from the current MXFP8 linear kernels. Kernel profiling is required to
establish the exact cause.

## Runs

| Arm | SLURM job | W&B |
| --- | ---: | --- |
| BF16 | 2503698 | [f9ld5mh8](https://wandb.ai/nvidia/sna-mxfp8-qkvo-refit-pr3294/runs/f9ld5mh8) |
| MoE baseline | 2503506 | [8dr0kzml](https://wandb.ai/nvidia/sna-mxfp8-qkvo-refit-pr3294/runs/8dr0kzml) |
| MoE optimized | 2503507 | [9sjcdtn8](https://wandb.ai/nvidia/sna-mxfp8-qkvo-refit-pr3294/runs/9sjcdtn8) |
| QKVO baseline | 2503508 | [gso4vqja](https://wandb.ai/nvidia/sna-mxfp8-qkvo-refit-pr3294/runs/gso4vqja) |
| QKVO optimized | 2503509 | [daod8kce](https://wandb.ai/nvidia/sna-mxfp8-qkvo-refit-pr3294/runs/daod8kce) |

The machine-readable aggregations are in `report_steps3_19.csv` and
`transfer_steps3_19.csv`; per-arm configuration is in `run_matrix.csv`.
