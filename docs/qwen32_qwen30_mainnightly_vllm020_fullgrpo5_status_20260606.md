# Qwen3-32B / Qwen3-30B-A3B NeMo-RL vLLM0.20 Full-GRPO Control

Date: 2026-06-06 PDT

## What Is Proven

NeMo-RL latest main/nightly is running vLLM `0.20.0` inside the Ray
`VllmGenerationWorker`, not only in standalone vLLM. Logs show:

```text
Initializing a V1 LLM engine (v0.20.0)
SpeculativeConfig(method='draft_model', model='amd/PARD-Qwen3-0.6B', num_spec_tokens=5)
```

The strongest positive control is Qwen3-32B PARD K5 job `3197802`:

| Model | Job | Config | Status | Evidence |
|---|---:|---|---|---|
| Qwen3-32B | `3197802` | public PARD K5 | `COMPLETED 0:0` | Full-GRPO reached Step 5/5 and printed Step 1-5 timing metrics. |
| Qwen3-30B-A3B | `3197890` | public PARD K3 retry | `FAILED 1:0` | Step 1-2 metrics printed; Step 3 hit vLLM CuMem allocator CUDA OOM. |
| Qwen3-32B | `3197980` | baseline mem80/bt16k retry | `COMPLETED 0:0` | Step 1-5 metrics printed. |
| Qwen3-32B | `3197981` | public PARD K3 mem80/bt16k retry | `COMPLETED 0:0` | Step 1-5 metrics printed; acceptance mostly `~55-74%` after the first tiny warmup bucket. |

The `3197802` cleanup log has TCPStore/NCCL warnings after the max-step exit,
but Slurm reports `COMPLETED 0:0`; treat those as shutdown noise, not a failed
run.

## Measured Timing Snapshot

Machine-readable step metrics:

```text
docs/qwen32_qwen30_mainnightly_vllm020_fullgrpo5_metrics_20260606.csv
```

Qwen3-32B K5 completed 5 steps:

| Metric | Step 1-5 average |
|---|---:|
| Total step time | `211.91s` |
| Generation time | `61.75s` |
| Policy training time | `102.40s` |
| Policy/ref logprobs time | `29.11s` |

Matched against the original Qwen3-32B baseline Step 1-2 before it OOMed:

| Comparison | Generation speedup | E2E step-time speedup | Caveat |
|---|---:|---:|---|
| Qwen3-32B public PARD K3 vs baseline, Step 1-2 | `1.62x` | `1.19x` | Both original runs later failed at Step 3 OOM. |
| Qwen3-32B public PARD K5 vs baseline, Step 1-2 | `1.86x` | `1.22x` | K5 completed 5 steps; baseline did not. |
| Qwen3-30B-A3B public PARD K5 vs baseline, Step 1-2 | `1.88x` | `1.12x` | Both original runs later failed at Step 3 OOM. |
| Qwen3-30B-A3B public PARD K3 retry vs baseline, Step 1-2 | `1.59x` | `1.10x` | Retry later failed at Step 3 OOM. |

The Qwen3-30B-A3B E2E result is consistent with the measured generation
fraction. Baseline Step 1-2 average total time is `300.54s`, while generation
is only `56.63s` (`18.8%`). With generation sped up by `1.88x`, Amdahl's law
predicts only about `1.10x` E2E step-time speedup; the measured result is
`1.12x`. The small E2E gain is therefore expected unless the workload becomes
more generation-bound or the training/logprob phases are also reduced.

The K3 retry shows the same pattern: generation `56.63s -> 35.66s` (`1.59x`)
but total step time `300.54s -> 272.61s` (`1.10x`). Acceptance stayed roughly
`59-72%` on Step 1-2, so the limited E2E gain is not primarily an acceptance
problem; it is dominated by policy training and policy/reference logprob time.

For the Qwen3-32B mem80/bt16k matched retry, Step 1-5 baseline vs K3 shows
generation time `118.01s -> 71.49s` (`1.65x`) and total step time
`261.31s -> 217.89s` (`1.20x`). Baseline generation fraction is `45.2%`, so
the observed E2E gain is again close to the expected generation-fraction bound.

The completed K3 Step 1-5 average is total `217.89s`, generation `71.49s`,
policy training `100.74s`, and policy/ref logprobs `28.71s`. The completed
baseline Step 1-5 average is total `261.31s`, generation `118.01s`, policy
training `99.53s`, and policy/ref logprobs `28.39s`.

## Failure Classification

The current blocker is not "vLLM 0.20 cannot run SpecDec in NeMo-RL." It can.
The blocker is stability at the original memory envelope:

| Job | Failure |
|---:|---|
| `3197798` Qwen3-32B baseline | Step 3 CUDA OOM in vLLM CuMem allocator. |
| `3197799` Qwen3-32B public PARD K3 | Step 3 CUDA OOM in vLLM CuMem allocator. |
| `3197800` Qwen3-30B-A3B baseline | Step 3 CUDA OOM in vLLM CuMem allocator. |
| `3197803` Qwen3-30B-A3B public PARD K5 | Step 3 CUDA OOM in vLLM CuMem allocator. |
| `3197890` Qwen3-30B-A3B public PARD K3 retry | Step 3 CUDA OOM in vLLM CuMem allocator. |

Retries `3197980`/`3197981` lower `gpu_memory_utilization` to `0.80` and
`max_num_batched_tokens` to `16384` while keeping `max_num_seqs=32`. These are
the current stability controls for a matched Qwen3-32B baseline/K3 comparison.

For Qwen3-30B-A3B, the 20-step stability pair completed with the same
conservative envelope and a worker-batch-32 shape:

| Model | Job | Config | Status | Purpose |
|---|---:|---|---|---|
| Qwen3-30B-A3B | `3198446` | baseline, GBS512, mem80/bt16k | `COMPLETED 0:0`, 20/20 | Passed previous Step-3 CuMem OOM point; Step 2-20 avg total `84.34s`, generation `15.77s`, E2E `141.17 tok/s/GPU`. |
| Qwen3-30B-A3B | `3198447` | public PARD K3, GBS512, mem80/bt16k | `COMPLETED 0:0`, 20/20 | Passed previous Step-3 CuMem OOM point; Step 2-20 avg total `81.58s`, generation `10.82s`, E2E `146.21 tok/s/GPU`; generation speedup `1.46x`, E2E throughput `1.04x`, total step-time `1.03x`; avg acceptance `69.45%`. |

The final Qwen3-30B-A3B result shows the same pattern as Qwen3-32B but with a
smaller E2E gain. PARD K3 reduces generation time by `31.4%`, but generation is
only `18.7%` of the matched baseline step, and non-generation work is roughly
flat to slightly slower. This makes the observed end-to-end speedup `1.03-1.04x`
even though generation alone is `1.46x` faster.

Details are tracked in:

```text
docs/qwen3_30ba3b_fullgrpo20_status_20260606.md
docs/qwen3_30ba3b_fullgrpo20_status_20260606.csv
```

## Implication For Qwen3-235B

This validates the latest-main vLLM0.20 NeMo-RL integration path before scaling
back to Qwen3-235B. It does not prove Qwen3-235B Full-GRPO E2E speedup yet.
For Qwen3-235B, use the same gates:

- first require Step 1+ full-GRPO metrics, not stop-after-generation only;
- keep `max_num_batched_tokens` conservative if CuMem OOM repeats;
- compare generation speedup and total step-time speedup on matched steps;
- do not promote a Qwen3-235B E2E claim until Slurm completion and parsed E2E
  metrics both exist.
