# Qwen3-235B-A22B GCP-NRT MXFP8 QKVO Results

## Methodology

- Hardware: 8 GCP-NRT nodes, 8 B200 GPUs per node
- Workload: synchronous GRPO, 20 steps, GBS 512, real importance sampling
- Parallelism: trainer TP2/PP4/CP2/EP16
- vLLM parallelism: TP8 for BF16; TP4 for all MXFP8 arms
- Scope: standard MXFP8 quantizes MoE weights; QKVO also quantizes Q/K/V/O
- Aggregation: arithmetic mean over W&B `_step=3..19` inclusive
- Checkpoint saving: disabled
- Software: commit `ef3f029ae33da686a901d4956c6cb9230bb93b75`

All five jobs completed 20 steps with exit code 0. Main timing and throughput
metrics contain 17 samples. Transfer/update and reward contain 16 samples
because those metrics are not logged on one step in the requested window.

Generated sequence lengths vary slightly across runs, so throughput is the
primary cross-arm metric. BF16 and MXFP8 use different vLLM TP settings;
MoE-only versus QKVO MXFP8 is the topology-matched scope comparison.

## Step Time

All values are seconds per step.

| Arm | E2E | Generation | Logprob | Policy training | Refit | Transfer/update |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 376.26 | 200.95 | 52.61 | 74.23 | 21.08 | 6.54 |
| MoE baseline | 321.34 | 130.81 | 50.38 | 72.55 | 42.04 | 22.33 |
| MoE optimized | 318.16 | 135.13 | 52.34 | 75.22 | 29.08 | 7.16 |
| QKVO baseline | 342.17 | 140.90 | 53.37 | 76.52 | 43.79 | 22.63 |
| QKVO optimized | 314.60 | 139.16 | 51.52 | 74.14 | 22.61 | 7.16 |

## Throughput

All values are tokens/s/GPU.

| Arm | E2E | Generation | Logprob | Policy training |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 124.06 | 228.71 | 877.36 | 620.92 |
| MoE baseline | 145.51 | 349.61 | 910.94 | 631.81 |
| MoE optimized | 146.82 | 338.38 | 875.78 | 609.04 |
| QKVO baseline | 138.60 | 328.53 | 870.73 | 606.08 |
| QKVO optimized | 150.64 | 332.32 | 900.67 | 625.52 |

## Findings

- The optimized MoE path reduced transfer/update by 67.9% and total refit by
  30.8% versus the MoE baseline. E2E step time improved by 1.0%.
- The optimized QKVO path reduced transfer/update by 68.3% and total refit by
  48.4% versus the QKVO baseline. E2E step time improved by 8.1%.
- Versus BF16, optimized MoE reduced E2E step time by 15.4% and increased E2E
  throughput by 18.3%.
- Versus BF16, optimized QKVO reduced E2E step time by 16.4% and increased E2E
  throughput by 21.4%.
- Against optimized MoE, optimized QKVO improved E2E throughput by 2.6%, but
  generation throughput was 1.8% lower. This does not establish a QKVO kernel
  speedup.

Mean reward was 0.5664 for BF16, 0.5662/0.5659 for MoE baseline/optimized, and
0.5521/0.5535 for QKVO baseline/optimized. This short performance run is not an
accuracy evaluation, so the QKVO scope is not correctness-validated here.

## Jobs And Runs

| Arm | SLURM job | Elapsed | W&B |
| --- | ---: | ---: | --- |
| BF16 | 474462 | 02:27:45 | [w425ns70](https://wandb.ai/nvidia/sna-mxfp8-qkvo-qwen235b-gcp-nrt/runs/w425ns70) |
| MoE baseline | 474463 | 02:16:13 | [auf703va](https://wandb.ai/nvidia/sna-mxfp8-qkvo-qwen235b-gcp-nrt/runs/auf703va) |
| MoE optimized | 474464 | 02:11:32 | [55dkfjpv](https://wandb.ai/nvidia/sna-mxfp8-qkvo-qwen235b-gcp-nrt/runs/55dkfjpv) |
| QKVO baseline | 474465 | 02:19:29 | [ljul0jic](https://wandb.ai/nvidia/sna-mxfp8-qkvo-qwen235b-gcp-nrt/runs/ljul0jic) |
| QKVO optimized | 474466 | 02:11:06 | [dwvwsdus](https://wandb.ai/nvidia/sna-mxfp8-qkvo-qwen235b-gcp-nrt/runs/dwvwsdus) |

Machine-readable values are in `report_steps3_19.csv` and
`transfer_reward_steps3_19.csv`.
