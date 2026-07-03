# Qwen3-32B 32K SpecDec Long-Context Diagnosis

Date: 2026-07-02
Updated: 2026-07-02 20:31 PDT

## Setup

- Recipe: `examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml`
- Mode: sync
- Cluster: Pretyche GB200, 4 nodes x 4 GPUs, `segment=4`
- Sampling: temperature `1.0`, top-p `1.0`
- Maximum output length: `32,768`
- Batch shape: 16 prompts x 16 generations, global batch size 256
- Context parallel size: 2
- CUDA Graphs: enabled (`enforce_eager=false`)
- Attention/MoE: `TRITON_ATTN` / Triton MoE
- Final comparison window: baseline Steps 2-20; each timeout-partial SpecDec
  row uses its own Steps 2-N window and the same baseline step span

## Final Five-Hour Outcomes

Both baselines completed 20/20. Every SpecDec row reached valid steady-state
steps and then exhausted the five-hour walltime; none of these rows is a crash
or a completed 20-step result. Eagle uses the standard baseline `2319103`.
PARD uses the `fuse_allreduce_rms=false` baseline `2319107` because the generic
draft-model path cannot share the target's fused-all-reduce workspace safely.

| Method | Job | State / steps | Gen time | Gen tok/s/GPU | Gen speedup | E2E time | E2E tok/s/GPU | E2E speedup | Acceptance | Mean accepted length |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline, standard | `2319103` | completed, 2-20 | 750.3s | 187.4 | 1.000x | 828.2s | 168.0 | 1.000x | n/a | n/a |
| Eagle K5 | `2319104` | timeout partial, 2-14 | 1,121.1s | 133.6 | 0.704x | 1,201.6s | 122.4 | 0.722x | 27.8% | 2.39 |
| Eagle K7 | `2319105` | timeout partial, 2-13 | 1,082.0s | 137.1 | 0.698x | 1,162.5s | 125.6 | 0.718x | 20.5% | 2.44 |
| Eagle K9 | `2319106` | timeout partial, 2-13 | 1,168.0s | 120.3 | 0.613x | 1,247.8s | 111.7 | 0.639x | 16.1% | 2.45 |
| Baseline, no fused all-reduce RMS | `2319107` | completed, 2-20 | 775.5s | 179.2 | 1.000x | 855.7s | 161.1 | 1.000x | n/a | n/a |
| PARD K1 | `2319108` | timeout partial, 2-6 | 2,446.7s | 62.8 | 0.365x | 2,531.6s | 60.1 | 0.388x | 60.8% | 1.61 |
| PARD K7 | `2319109` | timeout partial, 2-8 | 1,924.9s | 81.2 | 0.439x | 2,009.4s | 76.5 | 0.463x | 19.3% | 2.35 |
| PARD K9 | `2319110` | timeout partial, 2-7 | 2,168.9s | 69.5 | 0.403x | 2,252.3s | 66.3 | 0.427x | 15.1% | 2.36 |
| PARD K16 | `2319111` | timeout partial, 2-6 | 2,662.7s | 62.6 | 0.364x | 2,750.3s | 59.7 | 0.385x | 8.5% | 2.36 |

The speedup columns use each SpecDec row's exact observed step span against the
same baseline steps. Absolute baseline values above show final Steps 2-20 means
and are not used as a mismatched denominator for shorter partial rows.

## Early Steps 2-4 Diagnosis

| Method | Mean generated tokens | Max generated tokens | Acceptance | Mean accepted length | Verifier work amplification | Generation throughput speedup | E2E throughput speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 8,053 | 32,682 | n/a | 1.00 | 1.00x | 1.000x | 1.000x |
| Eagle K5 | 7,975 | 32,676 | 28.4% | 2.42 | 2.48x | 0.742x | 0.750x |
| Eagle K7 | 8,083 | 32,674 | 20.9% | 2.47 | 3.24x | 0.740x | 0.758x |
| Eagle K9 | 8,087 | 32,672 | 16.4% | 2.48 | 4.03x | 0.755x | 0.769x |

Verifier work amplification is `(K + 1) / mean accepted length`. It estimates
the target query-token work required per emitted token before accounting for
parallel-kernel efficiency.

The generated-token means differ from baseline by at most 1.0%, and all long
steps reach nearly the same 32K maximum. The throughput regression is therefore
not explained by shorter or longer sampled outputs.

## Step-Level Evidence

| Method | Step | Mean generated tokens | Generation time | Generation tok/s/GPU |
|---|---:|---:|---:|---:|
| Baseline | 2 | 3,498 | 322.4s | 178.4 |
| Baseline | 3 | 10,465 | 983.6s | 172.0 |
| Baseline | 4 | 10,197 | 1,234.8s | 133.4 |
| Eagle K5 | 2 | 3,476 | 358.2s | 159.6 |
| Eagle K5 | 3 | 10,332 | 1,333.2s | 125.3 |
| Eagle K5 | 4 | 10,118 | 2,211.0s | 74.0 |
| Eagle K7 | 2 | 3,572 | 421.6s | 139.2 |
| Eagle K7 | 3 | 10,385 | 1,347.2s | 124.6 |
| Eagle K7 | 4 | 10,292 | 1,763.7s | 94.3 |
| Eagle K9 | 2 | 3,614 | 372.0s | 159.6 |
| Eagle K9 | 3 | 10,507 | 1,458.1s | 116.5 |
| Eagle K9 | 4 | 10,141 | 1,840.7s | 89.0 |

The degradation grows on the 10K-token steps. This is consistent with target
verification becoming KV-attention-bandwidth dominated as the active context
grows. Each speculative iteration verifies `K + 1` query tokens, while the
observed mean accepted length remains near 2.4 tokens.

## Default OSL Versus 32K

The same CUDA-Graph-enabled sync recipe improves at the default OSL for small K
but regresses at 32K. This comparison uses the original graph-cap rows for PARD
so that the runtime path remains aligned with the 32K runs.

| Method | Default OSL gen speedup | 32K gen speedup | Default OSL E2E speedup | 32K E2E speedup | Acceptance, default to 32K | Verifier work amplification, default to 32K |
|---|---:|---:|---:|---:|---:|---:|
| Eagle K5 | 1.331x | 0.704x | 1.212x | 0.722x | 31.3% to 27.8% | 2.34x to 2.51x |
| Eagle K7 | 1.163x | 0.698x | 1.140x | 0.718x | 23.1% to 20.5% | 3.05x to 3.28x |
| Eagle K9 | 0.851x | 0.613x | 0.925x | 0.639x | 18.1% to 16.1% | 3.80x to 4.08x |
| PARD K1, target/draft TP2 | 1.085x | 0.365x | 1.051x | 0.388x | 76.2% to 60.8% | 1.14x to 1.24x |
| PARD K7, target/draft TP2 | 1.131x | 0.439x | 1.079x | 0.463x | 30.8% to 19.3% | 2.54x to 3.41x |
| PARD K9, target/draft TP2 | 0.756x | 0.403x | 0.823x | 0.427x | 24.5% to 15.1% | 3.12x to 4.24x |
| PARD K16, target/draft TP2 | 0.648x | 0.364x | 0.739x | 0.385x | 13.3% to 8.5% | 5.44x to 7.20x |

Eagle K5 is the cleanest isolation: acceptance falls only 3.5 percentage points
and verifier work amplification rises about 7%, yet generation changes from a
33% gain to a 30% loss. The long-context reversal therefore cannot be explained
by acceptance alone. The cost per target verification query increases with the
active KV-cache length, so fixed K becomes progressively less economical.

PARD K1 strengthens the same conclusion. Its 32K acceptance remains high at
60.8%, but target/draft TP2 generation throughput falls to 0.365x. The dense
28-layer draft model and target verification cost dominate at this context.
Moving the async-1off target and draft to TP1 can remove TP2 proposer overhead,
but it cannot remove the active-context-dependent target verification cost.

## E2E Breakdown

Across Steps 2-4, non-generation stages remain close:

| Method | Prepare generation | Policy training | Policy/reference logprobs | Generation |
|---|---:|---:|---:|---:|
| Baseline | 10.7s | 36.0s | 33.2s | 846.9s |
| Eagle K5 | 10.8s | 37.6s | 34.4s | 1,300.8s |
| Eagle K7 | 11.6s | 37.7s | 32.5s | 1,177.5s |
| Eagle K9 | 11.0s | 36.9s | 33.4s | 1,223.6s |

The E2E regression is concentrated in rollout generation rather than policy
training, weight preparation, or logprob evaluation.

## Current Code Limitation

The existing runtime dynamic-draft patch in
`nemo_rl/models/generation/vllm/specdec_runtime_gate_patch.py` only selects a
smaller K when `drafter.parallel_drafting` is true. That covers PARD-style
parallel drafting but not the sequential Eagle proposer. Its tiers are also
selected from scheduled request and scheduled token counts; they do not use
the active sequence/context length. The current mechanism therefore cannot
directly correct the observed Eagle long-context regression.

The evidence supports a context-aware gate as the next implementation design:
retain the default-OSL winning K, reduce K as active context grows, and disable
SpecDec once the predicted verifier work exceeds the baseline decode cost. The
gate must cover both sequential Eagle and parallel PARD paths and log its chosen
K per step. This design is not implemented or submitted yet.

## Async-1off Coverage Gap

No Qwen3-32B async-1off 32K NeMo-RL manifest or result exists yet. The official
async performance recipe inherits the same Megatron TP2 policy but moves vLLM
to four dedicated generation nodes with target TP1 and
`gpu_memory_utilization=0.8`. Dedicated generation GPUs remove the colocated
vLLM sleep/wake memory boundary, but they do not reduce policy activation memory.

The sync smoke sequence already established the policy requirements:

1. CP1 failed policy training with 178.8-178.9 GiB in use.
2. CP2 passed the memory boundary but required packed lengths divisible by 8.
3. CP2 plus `policy.make_sequence_length_divisible_by=8` completed the smoke and
   the later 20-step baseline.

A valid async-1off 32K baseline smoke must therefore inherit
`grpo-qwen3-32b-8n4g-async-1off.yaml` and apply the same CP2/pad8, 16x16 batch,
max OSL 32,768, `max_num_seqs=16`, `max_num_batched_tokens=32768`, CUDA Graph,
TRITON_ATTN, and Triton MoE settings. It should run for three steps with a new
`sna-nemorl-specdec-lyris` W&B name before any async SpecDec rows are submitted.
This control has not been submitted.

## Current Conclusion

The five-hour outcomes confirm that the early regression persists. Eagle
K5/K7/K9 finish at 0.704x, 0.698x, and 0.613x generation throughput and 0.722x,
0.718x, and 0.639x E2E throughput. PARD K1/K7/K9/K16 finish at 0.365x, 0.439x,
0.403x, and 0.364x generation throughput. None completes 20 steps before the
walltime.

Low acceptance and long-context target verification remain the leading Eagle
explanation. PARD adds a much larger 28-layer dense drafter cost; even K1's high
60.8% acceptance cannot offset that proposer at this context and runtime shape.
The evidence does not support increasing K for either method. The next valid
long-context experiment is the missing async baseline smoke, not another sync K
sweep.

## Sources

- `docs/pretyche_qwen32_sync_osl32k_matched_live_metrics_20260702.csv`
- `docs/latest_pretyche_qwen32_sync_osl32k_matched_step20_20260702_jobs.csv`
- `docs/latest_pretyche_qwen32_sync_osl32k_smoke_20260702_jobs.csv`
