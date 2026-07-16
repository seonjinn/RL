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

The Qwen3-32B Thinking head is the stronger K1 checkpoint in this matched
window. Its reward remains aligned with the baseline while generation and E2E
throughput improve by 16.5% and 7.0%, respectively. Qwen3-30B-A3B K3 delivers
the largest final20 gain in this wave: 59.3% generation-throughput and 16.9%
E2E-throughput improvement.

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

## Qwen3-235B Port Failure

Baseline job `2402495` and base EAGLE3 K1 job `2402497` failed at the same
vLLM TP=8 engine initialization boundary. In each cross-node RayExecutorV2
engine, vLLM probed the fixed NeMo-RL `VLLM_PORT` for both its TCPStore and
remote MessageQueue; the MessageQueue bound it before TCPStore initialization,
producing `torch.distributed.DistNetworkError: EADDRINUSE` on ports such as
`7000` and `7100`. This reproduces without a drafter, so it is not evidence
against either Qwen3-235B checkpoint.

The isolated retry delegates rendezvous allocation to vLLM 0.25.1 by setting
`NRL_DISABLE_VLLM_PORT_OVERRIDE=1` only for Qwen3-235B. Qwen3-30B and Qwen3-32B
retain their already validated runtime environment. The remaining old 235B
smokes were cancelled before allocation and will be resubmitted only after the
fixed baseline passes. Fixed baseline smoke job `2405130` passed scheduler
preflight and is pending Lyris capacity.

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
| Qwen3-32B | EAGLE3 Thinking | K1, K2, K3, K5 | MRv2 | K1 final20 complete; K2/K3 smoke complete | `2405078`, `2406250`, `2406253` | K2 fills the fixed-policy gap; K3 has a matched smoke rerun |
| Qwen3-32B | EAGLE3 Thinking DynamicSD | K0-K3 | MRv2 | seed smoke complete | `2406255` | final20 requires matched vLLM 0.25.1 calibration |
| Qwen3-32B | DFlash | K3, K5 | MRv2 | planned | pending | exact head; draft FlashAttention |
| Qwen3-32B | draft/PARD | draft K1/K5; PARD K5/K16 | MRv1 | planned | pending | shared AMD 0.6B drafter; sequential/parallel split |
| Qwen3-32B | suffix/ngram | suffix K32; ngram K5; ngram-gpu K5 | MRv1 | planned | pending | checkpoint-free proposers |
| Qwen3-235B-A22B | baseline | MRv2, MRv1 | mixed | fixed smoke pending | `2405130` | MRv2 retry delegates rendezvous ports to vLLM |
| Qwen3-235B-A22B | EAGLE3 | K1, K3, K5 | MRv2 | blocked by baseline gate | `2402497/500/502` | K1 port failure reproduced; K3/K5 cancelled |
| Qwen3-235B-A22B | EAGLE3 Thinking | K1, K3, K5 | MRv2 | blocked by baseline gate | `2402626/30/32` | cancelled before allocation pending fixed baseline |
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
