# vLLM 0.25.1 Drafter Matrix Live Report

## Status

The matrix, strict result collector, and Lyris submission workflow are under
validation. No result below is considered reportable until its exact matched
baseline and candidate both complete steps 2-20.

| Field | Controlled value |
|---|---|
| Branch | `sna/nemorl-vllm0251-drafter-matrix-20260716` |
| vLLM | official 0.25.1 |
| Cluster | Lyris GB200, `coreai_dlalgo_llm`, `--segment=<nodes>` |
| CUDA Graph | enabled, `FULL_AND_PIECEWISE`, native sizing |
| Sampling | temperature 1.0, top-p 1.0 |
| Checkpoint saving | disabled |
| W&B project | `nemo-rl-vllm0251-drafter-matrix` |
| Final window | steps 2-20 inclusive |

## Lyris Preflight Inventory

Read-only inspection on 2026-07-16 found all three target snapshots, all three
base EAGLE3 snapshots, and the shared PARD snapshot complete at their pinned
revisions. Staging job `2402456` then completed in 1 minute 58 seconds with
exit code 0 and validated all eight immutable matrix drafter snapshots,
including exact DFlash and distinct Qwen32/Qwen235 Thinking EAGLE3 snapshots.
Qwen30 has no separate Thinking row because its
previously inspected Thinking alias at revision
`a7ec796dd65236f1ecd4ed2958a7f0689e5da5cf` has the same config blob
`4e11c4dbb9b0bd911748a6f567d41f57c3dcdbe3` and model LFS SHA-256
`d2d6e2e63e09dc755053ae5c98cdececae3611ae5e202d4fa5411126dd3b1dfa`
as the selected reasoning-enabled base checkpoint. The available image is
`nemo_rl_nightly_20260715.sqsh`; the older derived image referenced by the
previous experiment has been removed.

At that snapshot, partition `gb200` had no idle nodes: 244 were allocated and
the remainder were in maintenance/down/drain states. FairShare for user `sna`
under `coreai_dlalgo_llm` was 0.793651. Scheduler preflight may run, but smoke
jobs can remain pending until nodes return.

## Completed Smoke Wave

All jobs below passed an exact-topology scheduler preflight before submission.
They are two-step configuration/runtime gates, not reportable performance
results. Promotion to steps 2-20 waits for matched baseline and candidate gates.

| Model | Variant | Job | W&B |
|---|---|---:|---|
| Qwen3-30B-A3B | baseline | `2402479` | [run](https://wandb.ai/nvidia/nemo-rl-vllm0251-drafter-matrix/runs/97yhgm50) |
| Qwen3-30B-A3B | EAGLE3 K1 / K3 / K5 | `2402481` / `2402483` / `2402485` | [K1](https://wandb.ai/nvidia/nemo-rl-vllm0251-drafter-matrix/runs/49w01pin) / [K3](https://wandb.ai/nvidia/nemo-rl-vllm0251-drafter-matrix/runs/c7vd4qvt) / [K5](https://wandb.ai/nvidia/nemo-rl-vllm0251-drafter-matrix/runs/3gskmira) |
| Qwen3-32B | baseline | `2402487` | [run](https://wandb.ai/nvidia/nemo-rl-vllm0251-drafter-matrix/runs/5x7gpll3) |
| Qwen3-32B | EAGLE3 K1 / K3 / K5 | `2402489` / `2402491` / `2402493` | [K1](https://wandb.ai/nvidia/nemo-rl-vllm0251-drafter-matrix/runs/zaf7lrpt) / [K3](https://wandb.ai/nvidia/nemo-rl-vllm0251-drafter-matrix/runs/h01cbwzk) / [K5](https://wandb.ai/nvidia/nemo-rl-vllm0251-drafter-matrix/runs/mlhdehtm) |
| Qwen3-32B | Thinking EAGLE3 K1 / K3 / K5 | `2402620` / `2402622` / `2402624` | [K1](https://wandb.ai/nvidia/nemo-rl-vllm0251-drafter-matrix/runs/bo48pxvo) / [K3](https://wandb.ai/nvidia/nemo-rl-vllm0251-drafter-matrix/runs/tqex9iq5) / [K5](https://wandb.ai/nvidia/nemo-rl-vllm0251-drafter-matrix/runs/khslxp6o) |
| Qwen3-235B-A22B | baseline | `2402495` | failed: vLLM rendezvous `EADDRINUSE` |
| Qwen3-235B-A22B | EAGLE3 K1 / K3 / K5 | `2402497` / `2402500` / `2402502` | K1 reproduced the port failure; K3/K5 cancelled |
| Qwen3-235B-A22B | Thinking EAGLE3 K1 / K3 / K5 | `2402626` / `2402630` / `2402632` | cancelled before allocation pending the baseline fix |

### Preliminary Step-2 Gate

These values are one post-initialization step only. They select final20
candidates but are not final performance claims. Time speedups are matched
baseline time divided by candidate time.

| Model | Drafter | K | E2E time | E2E speedup | Gen time | Gen speedup | Acceptance | Mean accepted | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-30B-A3B | none | - | 179.5s | 1.00x | 73.2s | 1.00x | n/a | n/a | baseline |
| Qwen3-30B-A3B | base / Thinking-equivalent | 1 | 156.8s | 1.15x | 50.4s | 1.45x | 81.5% | 1.82 | pass |
| Qwen3-30B-A3B | base / Thinking-equivalent | 3 | 148.5s | 1.21x | 43.8s | 1.67x | 64.4% | 2.93 | promote |
| Qwen3-30B-A3B | base / Thinking-equivalent | 5 | 203.4s | 0.88x | 94.2s | 0.78x | 51.1% | 3.56 | reject |
| Qwen3-32B | none | - | 247.9s | 1.00x | 100.0s | 1.00x | n/a | n/a | baseline |
| Qwen3-32B | base | 1 | 256.1s | 0.97x | 109.9s | 0.91x | 69.2% | 1.69 | same-K control |
| Qwen3-32B | base | 3 | 264.6s | 0.94x | 118.3s | 0.85x | 45.1% | 2.35 | reject |
| Qwen3-32B | base | 5 | 287.9s | 0.86x | 139.6s | 0.72x | 31.3% | 2.57 | reject |
| Qwen3-32B | Thinking | 1 | 235.4s | 1.05x | 89.6s | 1.12x | 79.8% | 1.80 | promote |
| Qwen3-32B | Thinking | 3 | 243.6s | 1.02x | 96.0s | 1.04x | 62.6% | 2.88 | pass |
| Qwen3-32B | Thinking | 5 | 249.2s | 0.99x | 102.7s | 0.97x | 49.7% | 3.48 | reject |

Qwen3-30B does not demonstrate an independent Thinking-checkpoint effect:
the selected base repository and the Thinking alias resolve to identical model
weights. Qwen3-32B shows a public-checkpoint effect: at matched K1, the
Thinking head changed generation time from 109.9s to 89.6s and acceptance from
69.2% to 79.8% in this smoke. This is not an isolated effect of Thinking
training because the public heads also differ in implementation and size.

The performance recipes leave `chat_template_kwargs` unset. For all three
pinned target tokenizer snapshots, the chat template suppresses reasoning only
when `enable_thinking` is explicitly false, so the effective default remains
thinking-enabled. This preserves the original recipe behavior but should not be
confused with an explicit `enable_thinking=true` experiment override.

## Final20 Promotion Wave

All promoted jobs use the same performance recipes and CUDA Graph controls as
their smoke gates. Final comparisons average steps 2-20. To avoid a
duplicate allocation, the first five steps of each final20 run serve as the
in-place smoke5 gate; the jobs were also monitored for more than five minutes
without an early runtime error. Qwen3-32B base K1 is not a performance
promotion; it is retained solely as the same-K public-checkpoint control for
Thinking K1.

| Model | Variant | Job | W&B | State |
|---|---|---:|---|---|
| Qwen3-30B-A3B | baseline | `2404968` | [run](https://wandb.ai/nvidia/nemo-rl-vllm0251-drafter-matrix/runs/m6jqtwb0) | completed (`0:0`) |
| Qwen3-30B-A3B | EAGLE3 K3 | `2405075` | [run](https://wandb.ai/nvidia/nemo-rl-vllm0251-drafter-matrix/runs/lzg06xnm) | completed (`0:0`) |
| Qwen3-32B | baseline | `2405077` | [run](https://wandb.ai/nvidia/nemo-rl-vllm0251-drafter-matrix/runs/zy22udd0) | completed (`0:0`) |
| Qwen3-32B | base EAGLE3 K1 | `2405076` | [run](https://wandb.ai/nvidia/nemo-rl-vllm0251-drafter-matrix/runs/on935s9p) | completed (`0:0`) |
| Qwen3-32B | Thinking EAGLE3 K1 | `2405078` | [run](https://wandb.ai/nvidia/nemo-rl-vllm0251-drafter-matrix/runs/0rgf8mxc) | completed (`0:0`) |
| Qwen3-32B | Thinking EAGLE3 K3 | `2409618` | [run](https://wandb.ai/nvidia/nemo-rl-vllm0251-drafter-matrix/runs/8zf8g77s) | completed (`0:0`) |
| Qwen3-235B-A22B | baseline | `2409727` | [run](https://wandb.ai/nvidia/nemo-rl-vllm0251-drafter-matrix/runs/pgzi0h7u) | completed (`0:0`) |
| Qwen3-235B-A22B | Thinking EAGLE3 K3 | `2409729` | [run](https://wandb.ai/nvidia/nemo-rl-vllm0251-drafter-matrix/runs/xjr83tib) | completed (`0:0`) |
| Qwen3-235B-A22B | Thinking EAGLE3 K5 | `2409731` | [run](https://wandb.ai/nvidia/nemo-rl-vllm0251-drafter-matrix/runs/kblsp1fm) | completed (`0:0`) |
| Qwen3-235B-A22B | NVIDIA EAGLE3 K3 reproduction | `2411704` | [run](https://wandb.ai/nvidia/nemo-rl-vllm0251-drafter-matrix/runs/7ychqvya) | running; step 1 started |
| Qwen3-235B-A22B | NVIDIA EAGLE3 K5 reproduction | `2411706` | [run](https://wandb.ai/nvidia/nemo-rl-vllm0251-drafter-matrix/runs/9x0cy42u) | running; CUDA Graph capture active |

### Final20 Results

Every row below averages the complete step 2-20 window (19 points). Baselines
are matched by model and immutable runner configuration. Logged throughput is
averaged directly rather than reconstructed from averaged time.

| Model | Variant | E2E time | E2E time speedup | Gen time | Gen time speedup | Gen ratio | E2E tok/s/GPU | E2E throughput | Gen tok/s/GPU | Gen throughput | Acceptance | Mean accepted | Reward |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B | baseline | 180.92s | 1.000x | 69.74s | 1.000x | 38.5% | 2303.35 | 1.000x | 5958.07 | 1.000x | n/a | n/a | 0.5271 |
| Qwen3-30B-A3B | EAGLE3 K3 | 154.48s | 1.171x | 43.88s | 1.589x | 28.4% | 2692.91 | 1.169x | 9489.28 | 1.593x | 64.8% | 2.94 | 0.5251 |
| Qwen3-32B | baseline | 262.36s | 1.000x | 109.61s | 1.000x | 41.8% | 1610.78 | 1.000x | 3850.38 | 1.000x | n/a | n/a | 0.5242 |
| Qwen3-32B | base EAGLE3 K1 | 255.69s | 1.026x | 103.44s | 1.060x | 40.5% | 1653.10 | 1.026x | 4099.08 | 1.065x | 69.4% | 1.69 | 0.5248 |
| Qwen3-32B | Thinking EAGLE3 K1 | 245.08s | 1.071x | 94.22s | 1.163x | 38.4% | 1722.97 | 1.070x | 4484.79 | 1.165x | 80.0% | 1.80 | 0.5268 |
| Qwen3-32B | Thinking EAGLE3 K3 | 259.71s | 1.010x | 107.82s | 1.017x | 41.5% | 1626.36 | 1.010x | 3925.38 | 1.019x | 63.0% | 2.89 | 0.5232 |

The Qwen3-32B Thinking head is the stronger K1 checkpoint in this matched
window. Its reward remains aligned with the baseline while generation and E2E
throughput improve by 16.5% and 7.0%, respectively. Qwen3-30B-A3B K3 delivers
the largest final20 gain in this wave: 59.3% generation-throughput and 16.9%
E2E-throughput improvement.

### Qwen3-235B Final20 Results

All three rows are complete over steps 2-20. Throughput is averaged directly
from the logged per-GPU metrics.

| Variant | Window | E2E time | E2E time speedup | Gen time | Gen time speedup | Gen ratio | E2E tok/s/GPU | E2E throughput | Gen tok/s/GPU | Gen throughput | Acceptance | Mean accepted | Prepare-for-gen avg/max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | steps 2-20 (19/19) | 315.32s | 1.000x | 150.92s | 1.000x | 52.8% | 156.68 | 1.000x | 298.57 | 1.000x | n/a | n/a | 77.39s / 304.67s |
| Thinking EAGLE3 K3 | steps 2-20 (19/19) | 597.46s | 0.528x | 417.78s | 0.361x | 69.9% | 80.78 | 0.516x | 108.02 | 0.362x | 48.2% | 2.45 | 102.69s / 868.40s |
| Thinking EAGLE3 K5 | steps 2-20 (19/19) | 707.19s | 0.446x | 443.17s | 0.341x | 62.7% | 73.35 | 0.468x | 103.48 | 0.347x | 35.4% | 2.77 | 187.28s / 1065.21s |

The current Qwen3-235B Thinking runs are regressions despite nonzero
acceptance. Drafter identity is therefore not a sufficient explanation. The
effective 16-node performance recipe supplies only CUDA Graph capture sizes
`[1,2,4,8,16,32,64]`; a controlled vLLM 0.25.1 ablation elsewhere in this
repository slowed generation by 2.425x when coverage was reduced while
acceptance remained unchanged. Jobs `2411704` and `2411706` reproduce NVIDIA
K3/K5 under the exact current configuration to isolate checkpoint cost. A
second native-capture comparison is required to reproduce the earlier high
throughput cohort.

The first matched non-warmup Step 2 from the NVIDIA-checkpoint reproductions
already argues against drafter identity as the primary regression source. These
single-step rows are diagnostic only; the running jobs must finish before they
replace the final20 table above.

| Drafter / K | Step 2 E2E | E2E speedup | Step 2 gen | Gen speedup | E2E tok/s/GPU | E2E throughput | Gen tok/s/GPU | Gen throughput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 199.04s | 1.000x | 131.24s | 1.000x | 155.42 | 1.000x | 235.73 | 1.000x |
| Thinking K3 | 365.80s | 0.544x | 297.88s | 0.441x | 83.56 | 0.538x | 102.61 | 0.435x |
| NVIDIA K3 | 382.69s | 0.520x | 315.55s | 0.416x | 80.31 | 0.517x | 97.40 | 0.413x |
| Thinking K5 | 426.10s | 0.467x | 364.34s | 0.360x | 71.92 | 0.463x | 84.12 | 0.357x |
| NVIDIA K5 | 521.73s | 0.382x | 458.11s | 0.286x | 59.03 | 0.380x | 67.22 | 0.285x |

## Qwen3-32B K2 And DynamicSD Smoke Wave

The isolated matrix now includes fixed Thinking EAGLE3 K2 and a validated
DynamicSD K0-K3 path. Fixed K2 and K3 use native vLLM 0.25.1 without a runtime
patch. DynamicSD alone applies the source-guarded autoregressive-drafter CUDA
Graph fix in its run-specific venv and preserves `FULL_AND_PIECEWISE` with
native capture sizing.

The checked-in schedule uses K3 for scheduler batch sizes 1-127 and K1 for
128-256. It is a smoke seed derived from a historical vLLM 0.24 standalone
profile, not a reportable calibration. The launcher rejects this seed for
`final20`; promotion requires a matched vLLM 0.25.1 NeMo-RL profile marked
`calibrated`. vLLM 0.25.1 exposes total draft/accepted counters but no native
selected-K histogram, so K-specific fractions require separate scheduler-side
instrumentation and are not inferred from accepted-token position counters.

| Model | Variant | Phase | Job/W&B | State |
|---|---|---|---|---|
| Qwen3-32B | Thinking EAGLE3 K2 | smoke2 | `2406250` / [W&B](https://wandb.ai/nvidia/nemo-rl-vllm0251-drafter-matrix/runs/e9pterbv) | completed (`0:0`) |
| Qwen3-32B | Thinking EAGLE3 K3 | smoke2 | `2406253` / [W&B](https://wandb.ai/nvidia/nemo-rl-vllm0251-drafter-matrix/runs/7hwdrdoe) | completed (`0:0`) |
| Qwen3-32B | Thinking DynamicSD K0-K3 seed | smoke2 | `2406255` / [W&B](https://wandb.ai/nvidia/nemo-rl-vllm0251-drafter-matrix/runs/krwc1su5) | completed (`0:0`) |

All three submissions passed their exact `sbatch --test-only` shape before
submission and started on separate four-node segments. At 5 minutes 11 seconds,
all remained running with W&B connected, all 16 generation workers initialized,
and EAGLE3 CUDA Graph capture reached. DynamicSD logged successful application
of the vLLM 0.25.1 patch on all four nodes and resolved the exact seed ranges.
No traceback, OOM, NCCL watchdog, or CUDA Graph downgrade was observed. The
repeated `git diff main` message is W&B source-diff metadata noise and did not
stop initialization. All three jobs completed both steps in about 15 minutes.
The allocator emitted `CUDA Error: invalid argument` during process teardown,
after metrics and the early-stop message; SLURM still recorded successful
`0:0` exits.

### K2 And DynamicSD Preliminary Step-2 Gate

The exact runner-matched baseline is job `2402487`. These are single step-2
observations, so they validate direction only and are not final20 claims.
Throughput values are logged per-GPU values; time speedup is baseline time
divided by candidate time, and throughput speedup is candidate throughput
divided by baseline throughput.

| Variant | E2E time | E2E speedup | Gen time | Gen speedup | E2E tok/s/GPU | E2E throughput | Gen tok/s/GPU | Gen throughput | Acceptance | Mean accepted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 247.93s | 1.000x | 100.05s | 1.000x | 1692.53 | 1.000x | 4194.35 | 1.000x | n/a | n/a |
| Thinking EAGLE3 K2 | 248.78s | 0.997x | 101.74s | 0.983x | 1683.02 | 0.994x | 4115.47 | 0.981x | 70.7% | 2.41 |
| Thinking EAGLE3 K3 | 243.64s | 1.018x | 97.60s | 1.025x | 1720.37 | 1.016x | 4294.77 | 1.024x | 62.6% | 2.88 |
| Thinking DynamicSD K0-K3 seed | 237.29s | 1.045x | 90.50s | 1.105x | 1767.04 | 1.044x | 4633.11 | 1.105x | 78.6% | 1.82 |

The DynamicSD row is promising but provisional: its aggregate acceptance
mixes K3 and K1 intervals, and vLLM does not expose a selected-K histogram.
The current seed remains ineligible for final20 until a matched vLLM 0.25.1
NeMo-RL calibration is produced and allowlisted.

### K0-K5 Offline Calibration

The reportable DynamicSD path declares global K5 and profiles fixed K0-K5
before final20. This avoids the vLLM behavior that silently clamps a schedule
entry above the configured global K. The grid uses Qwen3-32B TP2, Thinking
EAGLE3 draft TP1, temperature/top-p 1.0/1.0, max model length 4096, 256
generated profiling tokens, max batched tokens 16384, and native
`FULL_AND_PIECEWISE` CUDA Graph sizing.

The batch-size points are `1,4,16,32,64,128,192,256`; every K/BS point runs
twenty batches on deterministic OpenMathInstruct-2 prompts. K5 separately
collects position-level acceptance. The schedule uses the vLLM PR #32374
criterion `accepted_length / median_ITL` and linearly interpolates only between
measured batch-size points. Missing cells cannot produce a calibrated
artifact, and schema-v2 final20 remains blocked until the reviewed schedule
SHA-256 is allowlisted.

The OpenMathInstruct-2 source is pinned to revision
`469216e3f46f4dacf476b382e192485ea51a143e`. Profiling jobs materialize a
per-job locked vLLM 0.25.1 environment under `/tmp` and leave the container's
base `/opt/nemo_rl_venv` unchanged.

All six Lyris jobs passed exact scheduler preflight and completed with exit
code `0:0`. Each K produced all eight batch-size cells, for 48/48 complete
cells. The shutdown-only `EngineDeadError` messages occurred after result
files were written and did not invalidate the jobs.

| K | Job | Cells | State |
|---:|---:|---:|---|
| 0 | `2407016` | 8/8 | completed |
| 1 | `2407017` | 8/8 | completed |
| 2 | `2407018` | 8/8 | completed |
| 3 | `2407019` | 8/8 | completed |
| 4 | `2407020` | 8/8 | completed |
| 5 | `2407021` | 8/8 | completed |

The K5 acceptance pass observed 70,214 draft events and 351,070 draft
tokens. Positional acceptance was 80.75%, 65.07%, 52.59%, 42.22%, and
33.79%, giving expected accepted lengths of 1.000, 1.807, 2.458, 2.984,
3.406, and 3.744 for K0 through K5. Goodput is expected accepted length
divided by measured median ITL.

| Scheduler BS | K0 ITL | Selected K | Selected ITL | Expected accepted | Goodput vs K0 |
|---:|---:|---:|---:|---:|---:|
| 1 | 7.505 ms | 5 | 9.102 ms | 3.744 | 3.087x |
| 4 | 7.739 ms | 5 | 9.656 ms | 3.744 | 3.001x |
| 16 | 7.717 ms | 5 | 10.485 ms | 3.744 | 2.756x |
| 32 | 7.965 ms | 5 | 12.802 ms | 3.744 | 2.329x |
| 64 | 8.601 ms | 3 | 12.976 ms | 2.984 | 1.978x |
| 128 | 9.636 ms | 1 | 12.456 ms | 1.807 | 1.398x |
| 192 | 11.917 ms | 1 | 17.812 ms | 1.807 | 1.209x |
| 256 | 13.029 ms | 1 | 20.872 ms | 1.807 | 1.128x |

The zero-margin calibrated schedule is `BS 1-34 -> K5`, `35-75 -> K3`,
`76-85 -> K2`, and `86-256 -> K1`. Its raw-profile SHA-256 is
`6d888c2198dd2f592bcc146329cbeafa53f5eb44ee63cb59cb250a87272d479d`;
the schedule SHA-256 is
`8cdfed304302f45e04e72cd219cb0be26c23c30b509c010fe9081d0c6da5fc14`.
The checked-in artifacts are
`calibration/qwen32_thinking_k5_vllm0251_profile.json` and
`calibration/qwen32_thinking_k5_vllm0251_schedule.json`.

The runtime lookup key is the number of requests assigned tokens in the
current scheduler step, not total rollout size. The serving profile therefore
approximates this key with fixed concurrency. K2 at BS4 and K4 at BS32 show
isolated latency discontinuities; neither changes the selected K at those
measured points, but the narrow interpolated K2 range must be boundary-checked.
Before promotion, a NeMo-RL smoke must record scheduler-selected K and actual
drafter width so a parsed schedule alone is not treated as proof that the
drafting work changed.

Source inspection found a specific MRv2 risk in vLLM 0.25.1: the scheduler
records `num_spec_tokens_to_schedule`, while the EAGLE3 autoregressive
speculator loops over the global `num_speculative_steps` and returns that
global-width tensor. The run-scoped patch now forwards the selected K into the
autoregressive speculator, limits the decode loop and returned tensor to that
K, supports K0 by bypassing drafting, and records selected, requested, and
returned widths. Baseline and fixed-K paths remain unchanged.

Telemetry smoke job `2407523` completed both NeMo-RL steps with exit code
`0:0` ([W&B](https://wandb.ai/nvidia/nemo-rl-vllm0251-drafter-matrix/runs/3e4mav6e)).
The scheduler selected every intended range during the draining rollout:
K5 at BS1-34, K3 at BS35-75, K2 at BS76-85, and K1 at BS86-256. The
autoregressive speculator nevertheless returned width 5 for every observed
range. Unique scheduler batch-size observations contained 34 K5-to-width-5,
41 K3-to-width-5, 10 K2-to-width-5, and 171 K1-to-width-5 cases. This confirms
that the vLLM 0.25.1 MRv2 scheduler changes target verification width but does
not reduce EAGLE3 drafting work. The final20 gate therefore failed for a
specific implementation reason rather than a profiling or configuration
error.

Corrected smoke job `2412001` completed all five steps with exit code `0:0`
([W&B](https://wandb.ai/nvidia/nemo-rl-vllm0251-drafter-matrix/runs/azohuw4p)).
All four generation nodes applied the source-guarded patch, retained
`FULL_AND_PIECEWISE` CUDA Graph capture, and observed exact selected/requested/
returned-width parity for K0, K1, K2, K3, and K5. Steps 2-5 averaged:

| Window | E2E time | Gen time | E2E tok/s/GPU | Gen tok/s/GPU | E2E vs baseline | Gen vs baseline | E2E throughput | Gen throughput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DynamicSD steps 2-5 | 250.76s | 96.25s | 1688.59 | 4405.40 | 1.046x | 1.139x | 1.048x | 1.144x |

This passes the correctness and smoke performance gates. It is 3.6-12.2%
better than fixed K3 across the four timing/throughput comparisons, but remains
1.8-2.3% behind the fixed-K1 final20 result. The schedule is therefore approved
for final20 measurement, not yet promoted as the best Qwen3-32B policy.

The boundary launcher is ready for twelve isolated fresh-server cells:
BS34/35 with K3/K5, BS75/76 with K2/K3, and BS85/86 with K1/K2. Each cell
retains full benchmark arrays and a cell-specific server log. These jobs remain
unsubmitted until the MRv2 variable-width patch passes GPU unit and runtime
telemetry tests; otherwise they would refine a schedule that the drafter does
not execute.

| Stage | State | Required evidence |
|---|---|---|
| Pure derivation and schema | complete | focused tests, Ruff, Pyright |
| K0-K5 Lyris profile | complete | 48/48 cells and K5 position acceptance |
| Derived schedule review | complete | immutable profile and schedule hashes |
| Boundary spot check | launcher ready | exact checks near BS35, BS76, and BS86 |
| DynamicSD runtime smoke | complete (`2412001`) | K0/K1/K2/K3/K5 selected, requested, and returned widths match |
| MRv2 variable-width patch | GPU validated | CUDA Graph retained; fixed-K and baseline paths unchanged |
| DynamicSD final20 | approved for submission | checked-in schedule SHA-256 allowlisted after smoke gate |

## Qwen3-235B Port Failure

Baseline job `2402495` and base EAGLE3 K1 job `2402497` failed at the same
vLLM TP=8 engine initialization boundary. In each cross-node RayExecutorV2
engine, vLLM probed the fixed NeMo-RL `VLLM_PORT` for both its TCPStore and
remote MessageQueue; the MessageQueue bound it before TCPStore initialization,
producing `torch.distributed.DistNetworkError: EADDRINUSE` on ports such as
`7000` and `7100`. This reproduces without a drafter, so it is not evidence
against either Qwen3-235B checkpoint.

The isolated retry delegates rendezvous allocation to vLLM 0.25.1 by setting
`NRL_DISABLE_VLLM_PORT_OVERRIDE=1` only for Qwen3-235B. A later retry exposed a
separate host-memory cgroup OOM during the initial HF-to-Megatron conversion:
the default cache contained only 59 of 128 shards and no completion marker,
while hard NUMA memory binding concentrated policy-load and vLLM sleep-mode CPU
backing memory on individual NUMA nodes. The corrected path disables only hard
NUMA memory binding, retains CPU affinity, and reuses a validated complete
128-shard Megatron cache. It does not change model, sampling, or training code.

Corrected baseline smoke job `2409674` reached step 2/2 without the previous
port, host OOM, or NCCL failure. The matched Qwen3-235B baseline, Thinking K3,
and Thinking K5 final20 jobs are `2409727`, `2409729`, and `2409731`; all use
the original 16-node performance recipe, five-hour limit, CUDA Graph enabled,
and checkpoint saving disabled.

## Applicability And Run Ledger

`planned` means the exact checkpoint/config is defined but has not yet passed
the current branch's two-step runtime gate. Job IDs and W&B links are added only
after real submission.

| Model | Variant family | Candidates | Runner | State | Job/W&B | Reason or gate |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B | baseline | MRv2, MRv1 | mixed | final20 complete | `2404968` | MRv2 control complete; MRv1 remains planned |
| Qwen3-30B-A3B | EAGLE3 | K1, K3, K5 | MRv2 | K3 final20 complete | `2405075` | K1/K3/K5 smoke complete |
| Qwen3-30B-A3B | DFlash | K3, K5 | MRv2 | planned | pending | exact head; draft FlashAttention |
| Qwen3-30B-A3B | draft/PARD | draft K1/K5; PARD K5/K16 | MRv1 | planned | pending | shared AMD 0.6B drafter; sequential/parallel split |
| Qwen3-30B-A3B | suffix/ngram | suffix K32; ngram K5; ngram-gpu K5 | MRv1 | planned | pending | checkpoint-free proposers |
| Qwen3-32B | baseline | MRv2, MRv1 | mixed | final20 complete | `2405077` | MRv2 control complete; MRv1 remains planned |
| Qwen3-32B | EAGLE3 | K1, K3, K5 | MRv2 | K1 control final20 complete | `2405076` | K1 retained for same-K checkpoint comparison |
| Qwen3-32B | EAGLE3 Thinking | K1, K2, K3, K5 | MRv2 | K1/K3 final20 complete | `2405078`, `2409618` | K3 is nearly neutral over steps 2-20 |
| Qwen3-32B | EAGLE3 Thinking DynamicSD | K0-K5 | MRv2 | corrected smoke complete; final20 approved | `2407523`, `2412001` | final20 schedule allowlisted after exact width parity and CUDA Graph validation |
| Qwen3-32B | DFlash | K3, K5 | MRv2 | planned | pending | exact head; draft FlashAttention |
| Qwen3-32B | draft/PARD | draft K1/K5; PARD K5/K16 | MRv1 | planned | pending | shared AMD 0.6B drafter; sequential/parallel split |
| Qwen3-32B | suffix/ngram | suffix K32; ngram K5; ngram-gpu K5 | MRv1 | planned | pending | checkpoint-free proposers |
| Qwen3-235B-A22B | baseline | MRv2, MRv1 | mixed | corrected smoke and MRv2 final20 complete | `2409674`, `2409727` | vLLM owns rendezvous ports; complete Megatron cache avoids repeated conversion |
| Qwen3-235B-A22B | EAGLE3 | K1, K3, K5 | MRv2 | K3/K5 reproduction running | `2411704`, `2411706` | exact current-config NVIDIA checkpoint comparison |
| Qwen3-235B-A22B | EAGLE3 Thinking | K1, K3, K5 | MRv2 | K3/K5 final20 complete | `2409729`, `2409731` | both regress against matched baseline `2409727` |
| Qwen3-235B-A22B | DFlash | K3, K5 | MRv2 | unsupported | n/a | no exact public Qwen3-235B DFlash checkpoint |
| Qwen3-235B-A22B | draft/PARD | draft K1/K5; PARD K5/K16 | MRv1 | planned | pending | shared AMD 0.6B drafter; sequential/parallel split |
| Qwen3-235B-A22B | suffix/ngram | suffix K32; ngram K5; ngram-gpu K5 | MRv1 | planned | pending | checkpoint-free proposers |

## Explicitly Out Of Scope

| Method | State | Reason |
|---|---|---|
| Native MTP | unsupported | controlled Qwen3 targets do not embed MTP heads |
| DSpark | unsupported | no exact target-specific checkpoint |
| Medusa | unsupported | no exact target-specific checkpoint |
| `mlp_speculator` | unsupported | vLLM 0.25.1 MRv1 runtime gap |
| hidden-state extraction/custom class | excluded | not standalone acceleration proposers |
| PARD-2 | separate experiment | requires non-upstream method/checkpoint support |
| DFlare | separate experiment | requires non-upstream implementation |

## Promotion Gates

1. `show` resolves a valid model/method/checkpoint and preserves recipe controls.
2. `test-only` passes for the exact topology with no dependency or `--gres`.
3. `smoke2` reaches the second completed training step and emits required metrics.
4. `smoke5` confirms stable runner, CUDA Graph mode, and acceptance telemetry.
5. `final20` completes; steps 2-20 are compared only to the exact runner-matched baseline.

## Result Table

| Model | Variant | Steps | E2E time | E2E TPS/GPU | Gen time | Gen TPS/GPU | Gen ratio | Acceptance | Mean accepted | Speedups | W&B |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |

## Validation

Run from a supported Linux checkout (the lock intentionally excludes macOS),
before pushing and again from the clean cluster checkout or nightly container:

```bash
uv run --locked pytest -q \
  tests/experiments/test_vllm_0251_drafter_matrix.py \
  tests/experiments/test_vllm_0251_dynamic_schedule.py \
  tests/experiments/test_vllm_0251_drafter_results.py \
  tests/experiments/test_vllm_0251_drafter_staging.py \
  tests/experiments/test_vllm_0251_suffix_dependency.py
bash -n experiments/vllm_0251_drafter_matrix/submit_matrix.sh
bash -n experiments/vllm_0251_drafter_matrix/submit_stage_drafters.sh
uv run --locked ruff check \
  experiments/vllm_0251_drafter_matrix \
  tests/experiments/test_vllm_0251_drafter_matrix.py \
  tests/experiments/test_vllm_0251_drafter_results.py \
  tests/experiments/test_vllm_0251_drafter_staging.py
uv run --locked pyright \
  experiments/vllm_0251_drafter_matrix/matrix.py \
  experiments/vllm_0251_drafter_matrix/stage_drafters.py \
  experiments/vllm_0251_drafter_matrix/collect_results.py
uv lock --check
git diff --check
```
