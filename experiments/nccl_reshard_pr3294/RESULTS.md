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

## vLLM 0.25 Component Factorial Ablation

The receiver-side result was repeated as a full 2x2 factorial ablation at
source `5152b5e569ef2cf4dc242aa66be4e50303d29c3d`. All four arms used
trainer-side prequantization and NCCL-Reshard; only batched MoE shuffle and
loader-route caching changed. The setup otherwise matched the performance A/B:
Qwen3-30B-A3B, 4 B200 nodes split into 2 training and 2 generation nodes,
MoE-only MXFP8 rollout, GBS 2048, sequence length 4096, real importance
sampling, and checkpointing disabled. vLLM was `0.25.1`.

| Arm | Shuffle | Cache | Job | W&B | Status |
|---|---:|---:|---:|---|---|
| Baseline | Off | Off | `488867` | [aszejvw3](https://wandb.ai/nvidia/sna-pr3294-nccl-reshard-ablation/runs/aszejvw3) | 20/20, `0:0` |
| Batched shuffle | On | Off | `488868` | [xumk6oxd](https://wandb.ai/nvidia/sna-pr3294-nccl-reshard-ablation/runs/xumk6oxd) | 20/20, `0:0` |
| Loader cache | Off | On | `488869` | [5q5kvr8h](https://wandb.ai/nvidia/sna-pr3294-nccl-reshard-ablation/runs/5q5kvr8h) | 20/20, `0:0` |
| Optimized | On | On | `488870` | [z8tu865o](https://wandb.ai/nvidia/sna-pr3294-nccl-reshard-ablation/runs/z8tu865o) | 20/20, `0:0` |

Results are arithmetic means over Steps 3-20. Every metric has 18 valid
observations and no missing steps.

| Metric | Baseline | Shuffle only | Cache only | Both |
|---|---:|---:|---:|---:|
| Transfer/update | 3.772 s | 0.727 s | 3.988 s | 0.734 s |
| Total refit | 3.772 s | 0.727 s | 3.988 s | 0.734 s |
| E2E step time | 168.939 s | 169.538 s | 171.192 s | 169.860 s |
| E2E throughput | 1228.4 tok/s/GPU | 1222.4 | 1211.2 | 1219.5 |
| Generation | 48.377 s | 48.807 s | 48.487 s | 48.514 s |
| Policy training | 80.357 s | 83.232 s | 82.196 s | 83.389 s |
| Policy/reference logprobs | 34.576 s | 34.829 s | 34.789 s | 35.391 s |

| Component contrast | Refit delta | Interpretation |
|---|---:|---|
| Add shuffle without cache | -3.045 s (-80.7%, 5.19x faster) | Dominant refit optimization |
| Add shuffle with cache | -3.254 s (-81.6%) | Same benefit when cache is enabled |
| Add cache without shuffle | +0.216 s (+5.7%) | No refit benefit in this path |
| Add cache with shuffle | +0.007 s (+1.0%) | Effect is negligible |

The shuffle-only paired refit delta versus baseline was `-3.045 s/step`, with
a 95% confidence interval of `[-3.145, -2.946] s`. The optimized paired delta
was `-3.038 s/step`, with a 95% confidence interval of
`[-3.121, -2.955] s`. Batched MoE shuffle therefore accounts for effectively
all measured receiver-side refit improvement on this NCCL-Reshard/vLLM 0.25
path; loader-route caching adds no measurable benefit.

E2E time did not improve in this four-job run even though refit fell by about
3 seconds. Policy training was 2.9-3.0 seconds slower on the shuffle-enabled
node allocations, an unrelated component that offsets the direct refit gain.
The shuffle-only E2E delta was `+0.600 s`, with a 95% confidence interval of
`[-0.959, +2.158] s`; it is not distinguishable from zero. The direct refit
timer is the causal metric for this component ablation. A prior two-arm run on
different nodes measured a 2.21% E2E throughput gain, so E2E impact should be
repeated or controlled on identical nodes before making a headline claim.

Correctness indicators remained aligned:

| Metric | Baseline | Shuffle only | Cache only | Both |
|---|---:|---:|---:|---:|
| Mean rollout reward | 0.52889 | 0.52995 | 0.52851 | 0.52827 |
| Generation KL error | 0.003984 | 0.003988 | 0.003983 | 0.003983 |
| Median token-mult probability error | 1.0374 | 1.0610 | 1.0375 | 1.0388 |

Paired reward and generation-KL confidence intervals versus baseline included
zero for every arm. Token-mult probability error had sparse, large outliers in
all arms, so its arithmetic mean is not representative; the medians remained
near `1.04`.

The exact run manifest is in `factorial_runs_5152b5e56.csv`, and the canonical
W&B Steps 3-20 export is in `factorial_results_5152b5e56.csv`.

## Cumulative Prequantization Ablation

The cumulative ablation at source `5c50597f3ec684e455a6d5f64daeb48ed6122e22`
adds trainer prequantization, batched MoE shuffle, and loader-route caching in
that order. The first four arms use the legacy collective so the
prequantization delta keeps transport constant. The fifth arm changes only the
transport to NCCL-Reshard. A no-prequantization NCCL arm is not valid because
prequantization is the BF16-to-MXFP8 wire transform required by the current
NCCL-Reshard contract.

All five two-step correctness gates completed with exit code `0:0` on the same
four-node B200 topology. Step 2 follows an optimizer update and therefore
validates a non-initial weight refit.

| Cumulative arm | Job | W&B | Step 2 transfer/update | Generation KL | Reward |
|---|---:|---|---:|---:|---:|
| Receiver-quant baseline | `489123` | [ntm29zax](https://wandb.ai/nvidia/sna-pr3294-cumulative-ablation/runs/ntm29zax) | 8.75 s | 0.0044 | 0.25 |
| + Trainer prequantization | `489124` | [2wmh5qco](https://wandb.ai/nvidia/sna-pr3294-cumulative-ablation/runs/2wmh5qco) | 6.77 s | 0.0044 | 0.25 |
| + Batched MoE shuffle | `489125` | [dkz8kpwt](https://wandb.ai/nvidia/sna-pr3294-cumulative-ablation/runs/dkz8kpwt) | 4.22 s | 0.0044 | 0.25 |
| + Loader-route cache | `489126` | [9w3ccpt0](https://wandb.ai/nvidia/sna-pr3294-cumulative-ablation/runs/9w3ccpt0) | 5.93 s | 0.0044 | 0.25 |
| + NCCL-Reshard | `489127` | [gop80kln](https://wandb.ai/nvidia/sna-pr3294-cumulative-ablation/runs/gop80kln) | 0.65 s | 0.0044 | 0.25 |

These tiny-batch gates establish correctness, not performance. The loader-cache
row in particular is dominated by short-run variance. The reportable 20-step
runs use GBS 2048, sequence length 4096, real importance sampling, and compare
Steps 3-20:

| Cumulative arm | Job | Status at submission |
|---|---:|---|
| Receiver-quant baseline | `489157` | Running |
| + Trainer prequantization | `489158` | Queued |
| + Batched MoE shuffle | `489159` | Queued |
| + Loader-route cache | `489160` | Queued |
| + NCCL-Reshard | `489161` | Queued |

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

The full-batch follow-up, job `488732`, completed 20/20 steps with exit code
`0:0` and no traceback. Over Steps 3-20 it averaged `0.730 s` total refit,
`167.74 s` E2E step time, and `1236.0 tokens/s/GPU`. The run used GBS 2048,
sequence length 4096, and real importance sampling. W&B:
<https://wandb.ai/nvidia/sna-pr3294-nccl-mxfp8-prequant/runs/ongveks5>.

## Conclusion

The transform-aware exact shard-transfer path reduced steady-state refit time
by 83.6% and improved E2E throughput by 4.43% without a measurable reward or
generation-KL regression. On top of that path, the component factorial shows
that batched MoE shuffle accounts for the receiver-side refit gain: it reduced
refit by 80.7%, while loader-route caching added no benefit. The earlier
two-arm run measured a 2.21% E2E throughput gain, but the four-arm run did not
reproduce that E2E result because unrelated policy-training time differed by
node allocation. Native M2N and a topology-controlled E2E repeat remain before
making broader performance claims.
