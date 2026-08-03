# BF16 Training to MXFP8 Rollout NCCL-Reshard Results

## Objective

Measure whether transform-aware NCCL-Reshard reduces weight-refit time for BF16 Megatron training with MXFP8 vLLM rollout. This experiment is isolated from PR 3294 so its established legacy-refit results remain unchanged.

## Setup

| Item | Value |
|---|---|
| Cluster | GCP-NRT, B200, 8 GPUs/node |
| Model | Qwen3-30B-A3B |
| Nodes | 4 total: 2 training + 2 generation |
| Training storage | BF16 |
| Rollout storage | MoE-only MXFP8; QKVO remains BF16 |
| Trainer parallelism | TP1, PP1, EP16 |
| vLLM parallelism | TP1, PP1, EP1 |
| Batch | 64 prompts x 32 generations = GBS 2048 |
| Sequence length | 4096 |
| Dataset | OpenMathInstruct-2 |
| Steps | 20; primary window is steps 3-20 |
| Importance sampling | Enabled |
| Checkpointing | Disabled |
| Source | `45cfb89164d949ab2ea7cd86e6d6c7404ff7c529` |

The matched A/B holds trainer-side MXFP8 prequantization constant and changes only the transport:

- Legacy: packed collective refit with `refit_transport=null`.
- NCCL-Reshard: exact shard transfer for paired E4M3 values and E8M0 scales, with unsupported tensors using the miscellaneous broadcast path.

## Correctness Gate

Job `486926` completed two GRPO steps with exit code `0:0`.

- W&B: <https://wandb.ai/nvidia/sna-pr3294-nccl-mxfp8-prequant/runs/jtuxxyl8>
- Step 1 generation KL error: `0.0054`
- Step 2 generation KL error: `0.0044`
- Step 2 transfer/update: `1.10 s`
- Step 2 E2E: `10.57 s`
- Reshard payload: `27.84 GiB`; miscellaneous broadcast: `2.87 GiB`; reshard fraction: `90.7%`

## Performance A/B

| Arm | Job | W&B | Status |
|---|---:|---|---|
| Legacy collective | `486954` | <https://wandb.ai/nvidia/sna-pr3294-nccl-mxfp8-prequant/runs/9wlo72ky> | Completed, `0:0` |
| NCCL-Reshard prequant | `486955` | <https://wandb.ai/nvidia/sna-pr3294-nccl-mxfp8-prequant/runs/3gme19dv> | Completed, `0:0` |

Results below use the 18 steady-state steps from step 3 through step 20.

| Metric | Legacy mean | NCCL mean | Change | Legacy median | NCCL median |
|---|---:|---:|---:|---:|---:|
| Transfer/update | 4.796 s | 0.787 s | -83.6%, 6.10x faster | 4.130 s | 0.730 s |
| Total refit | 4.796 s | 0.788 s | -83.6% | 4.130 s | 0.730 s |
| E2E step time | 175.61 s | 168.23 s | -4.21% | 174.54 s | 167.16 s |
| E2E throughput | 1179.61 tok/s/GPU | 1231.90 tok/s/GPU | +4.43% | 1181.56 | 1235.83 |
| Generation | 48.51 s | 48.38 s | -0.25% | 48.46 s | 48.64 s |
| Policy training | 82.76 s | 80.45 s | -2.80% | 81.30 s | 80.32 s |
| Policy and reference logprobs | 36.89 s | 37.11 s | +0.60% | 36.93 s | 37.44 s |
| Mean rollout reward | 0.52796 | 0.52919 | +0.00123 | 0.52950 | 0.53005 |
| Generation KL error | 0.003994 | 0.003983 | -0.000011 | 0.004000 | 0.004000 |

The paired mean refit reduction was `4.009 s/step` with a 95% confidence interval of `[3.442, 4.575] s`. The paired E2E reduction was `7.388 s/step`, with a 95% confidence interval of `[5.967, 8.809] s`. Reward and generation-KL confidence intervals included zero:

- Reward delta: `+0.001233`, 95% CI `[-0.002045, +0.004512]`.
- Generation KL delta: `-0.000011`, 95% CI `[-0.000035, +0.000012]`.

The direct refit timer is the primary transport metric. E2E measurements also contain generation-length and node variation, although the paired E2E and throughput results both favored the NCCL arm.

## Residual Optimizations After NCCL-Reshard

This A/B holds trainer-side prequantization and the NCCL exact-transfer path
constant. It isolates the receiver-side work that remains after transport:

- Baseline: scalar MoE shuffle and uncached loader-route lookup.
- Optimized: batched MoE shuffle and identity-validated loader-route caching.

| Arm | Job | W&B | Status |
|---|---:|---|---|
| NCCL receiver baseline | `487298` | <https://wandb.ai/nvidia/sna-pr3294-nccl-mxfp8-prequant/runs/mzr8x55g> | Completed, `0:0` |
| NCCL receiver optimized | `487299` | <https://wandb.ai/nvidia/sna-pr3294-nccl-mxfp8-prequant/runs/8c2n3oj7> | Completed, `0:0` |

Results use the 18 steady-state steps from step 3 through step 20.

| Metric | NCCL baseline | NCCL optimized | Change |
|---|---:|---:|---:|
| Transfer/update | 4.138 s | 0.886 s | -78.6%, 4.67x faster |
| Total refit | 4.138 s | 0.887 s | -78.6% |
| E2E step time | 175.72 s | 172.10 s | -2.06% |
| E2E throughput | 1178.97 tok/s/GPU | 1205.04 tok/s/GPU | +2.21% |
| Generation | 48.52 s | 48.59 s | +0.13% |
| Policy training | 84.43 s | 83.13 s | -1.55% |
| Policy and reference logprobs | 37.06 s | 37.85 s | +2.14% |
| Mean rollout reward | 0.52802 | 0.52713 | -0.00090 |
| Generation KL error | 0.003981 | 0.003974 | -0.000008 |

The paired refit reduction was `3.252 s/step`, with a 95% confidence interval
of `[3.094, 3.409] s`. The paired E2E reduction was `3.619 s/step`, with a 95%
confidence interval of `[1.722, 5.516] s`. Reward and generation-KL confidence
intervals included zero:

- Reward delta: `-0.000895`, 95% CI `[-0.004112, +0.002322]`.
- Generation KL delta: `-0.000008`, 95% CI `[-0.000021, +0.000005]`.

The relative refit improvement remains comparable to the pre-NCCL PR 3294
result (`9.67 s -> 2.98 s`, -69.2%). The absolute saving is smaller because
NCCL exact-transfer has already removed most transport overhead: the post-NCCL
receiver optimization saves `3.25 s/step`, compared with `6.69 s/step` in the
historical pre-NCCL measurement. Consequently, its E2E impact is smaller even
though its relative refit reduction is larger.

## Runtime Boundary

The staged nightly image does not contain an importable `nccl.m2n` module or `libnccl_m2n.so`. The NCCL arm therefore uses NeMo-RL's `xferdtensor_python (exact-transfer)` implementation over NCCL communicators. These results validate the transform-aware NCCL-Reshard algorithm and its Python exact-transfer fallback; they are not compiled native-M2N measurements.

## Generic Transform Contract Validation

The generalized transform registry and ordered-component protocol were
revalidated on GCP-NRT at source
`fbe22cc3dcb10b9edf26cb4234341a9485cd22d9`.

An initial functional run, job `488544`, exposed a metadata boundary error:
the already transformed FP8 wire tensor was passed back to the codec as if it
were the logical BF16 source. The fix keeps the immutable source shape, dtype,
and format separate from the current wire-component metadata, validates the
wire value and scale against the codec outputs, and preserves source shapes
when individual experts are grouped.

Validation results:

| Gate | Job | Result |
|---|---:|---|
| Core transform and synchronizer tests | `488631` | 147 passed |
| Megatron transform handshake tests | `488597` | 6 passed |
| vLLM mixed-plan agreement test | `488598` | 1 passed |
| Four-node functional smoke | `488645` | 2/2 steps, `COMPLETED 0:0` |

The functional smoke used Qwen3-30B-A3B on four B200 nodes split into two
BF16 trainer nodes and two MoE-only MXFP8 generation nodes. It used trainer
TP1/EP16, generation TP1/EP1, GBS 16, sequence length 512, real importance
sampling, and checkpointing disabled. The NCCL path transferred `27.84 GiB`
through exact reshard and `2.87 GiB` through miscellaneous broadcast.

Step 2 completed with `0.62 s` transfer/update, `10.05 s` E2E step time,
`0.00436` generation KL error, `1.033` token-mult probability error, and no
NaN/Inf metrics. W&B: <https://wandb.ai/nvidia/sna-pr3294-nccl-mxfp8-prequant/runs/dpsenun7>.

## Conclusion

The transform-aware exact shard-transfer path reduced steady-state refit time by 83.6% and improved E2E throughput by 4.43% without a measurable reward or generation-KL regression. On top of that path, the remaining receiver-side PR 3294 optimizations reduced refit by another 78.6% and improved E2E throughput by 2.21%. Native M2N must be packaged and measured separately before claiming compiled NCCL-M2N performance.
