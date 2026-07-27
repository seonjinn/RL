# Nemotron 3 Nano MXFP8 QKVO Results

## Methodology

- Hardware: 4 Lyris nodes, 4 GB200 GPUs per node
- Model: Nemotron 3 Nano checkpoint documented in `README.md`
- Workload: synchronous GRPO, 20 steps, GBS 16, real importance sampling
- Parallelism: trainer TP2/PP2/CP2/EP8; vLLM TP1
- Scope: standard MXFP8 ignores q/k/v/o; QKVO sets the ignored list to empty
- Aggregation: arithmetic mean over W&B steps 3-19 inclusive
- Software: exact commit, submodules, and immutable container are recorded in
  `PROVENANCE.md`

Step 1 initialization and MXFP8 autotuning are excluded. Generated sequence
lengths vary across runs, so throughput is the primary cross-precision metric.
All main metrics contain 17 samples; transfer/update contains 16 logged samples
within the same requested window.

## Step Time

All values are seconds per step.

| Arm | E2E | Generation | Logprob | Policy training | Refit | Transfer/update |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 36.43 | 20.37 | 3.58 | 3.48 | 3.88 | 0.99 |
| MoE baseline | 37.57 | 18.64 | 3.53 | 3.64 | 7.30 | 4.36 |
| MoE optimized | 34.70 | 18.46 | 3.37 | 3.30 | 5.24 | 2.15 |
| QKVO baseline | 37.33 | 18.76 | 3.45 | 3.36 | 7.34 | 4.36 |
| QKVO optimized | 35.67 | 19.01 | 3.50 | 3.64 | 5.28 | 2.22 |

## Throughput

All values are tokens/s/GPU.

| Arm | E2E | Generation | Logprob | Policy training |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 103.25 | 176.31 | 1023.01 | 1288.05 |
| MoE baseline | 87.64 | 176.12 | 950.18 | 1162.43 |
| MoE optimized | 102.23 | 188.76 | 1050.42 | 1320.76 |
| QKVO baseline | 91.20 | 179.82 | 996.49 | 1274.60 |
| QKVO optimized | 102.08 | 188.05 | 1049.48 | 1292.12 |

## Findings

- The refit optimization reduced total refit time by 28.3% for standard MXFP8
  and 28.0% for QKVO-inclusive MXFP8.
- Transfer/update time fell by 50.6% for standard MXFP8 and 49.1% for
  QKVO-inclusive MXFP8.
- Optimized standard MXFP8 was 7.1% faster than BF16 in generation throughput,
  but 1.0% slower in E2E throughput.
- Optimized QKVO was 6.7% faster than BF16 in generation throughput, but 1.1%
  slower in E2E throughput.
- Adding QKVO to optimized MXFP8 changed generation throughput by -0.38% and
  E2E throughput by -0.15% relative to optimized standard MXFP8. This is not a
  speedup in these point estimates.

The result is consistent with the attention projections being a small fraction
of Nano's MoE weights and their GEMM shapes not offsetting MXFP8 linear
quantization and dispatch overhead. Kernel profiling is required to establish
the exact cause.

## Correctness Diagnostics

| Arm | Mean token product error | Maximum token product error | Mean maximum sequence error | Maximum sequence error | Mean generation KL |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 1.017 | 1.025 | 1.022 | 1.028 | 0.0009 |
| MoE baseline | 1.050 | 1.087 | 1.096 | 1.494 | 0.0047 |
| MoE optimized | 1.047 | 1.069 | 1.065 | 1.129 | 0.0048 |
| QKVO baseline | 2.441 | 23.756 | 12.031 | 177.545 | 0.0064 |
| QKVO optimized | 1.300 | 3.050 | 3.477 | 16.018 | 0.0067 |

No sequence was masked because this performance recipe does not set a logprob
error threshold. Both QKVO runs show substantially larger probability-product
outliers than standard MXFP8, so QKVO is not correctness-validated or
recommended from this experiment. Mean rollout reward ranged from 0.713 to
0.743 across the five 17-step windows, but this small performance run is not an
accuracy evaluation.

## Runs

| Arm | SLURM job | W&B |
| --- | ---: | --- |
| BF16 | 2504133 | [60ayvb27](https://wandb.ai/nvidia/sna-mxfp8-qkvo-nano/runs/60ayvb27) |
| MoE baseline | 2504134 | [87340q06](https://wandb.ai/nvidia/sna-mxfp8-qkvo-nano/runs/87340q06) |
| MoE optimized | 2504135 | [l2zoecxw](https://wandb.ai/nvidia/sna-mxfp8-qkvo-nano/runs/l2zoecxw) |
| QKVO baseline | 2504136 | [o7oz28v8](https://wandb.ai/nvidia/sna-mxfp8-qkvo-nano/runs/o7oz28v8) |
| QKVO optimized | 2504137 | [ondgz3au](https://wandb.ai/nvidia/sna-mxfp8-qkvo-nano/runs/ondgz3au) |

The machine-readable aggregations are in `report_steps3_19.csv`,
`transfer_steps3_19.csv`, `correctness_steps3_19.csv`, and `run_matrix.csv`.
