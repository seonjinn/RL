# DynamicSD Under Synchronous RL Rollout: Where a Batch-Size-Aware K Schedule Pays Off

vLLM 0.24 ships DynamicSD: a user-supplied batch-size -> K lookup
(`num_speculative_tokens_per_batch_size`) that the scheduler applies per step.
A synchronous GRPO rollout traverses the whole batch-size axis every step
(launch at N x G concurrency, drain to a long tail), so it is the natural
stress test. We asked: profiled per-model K schedules vs the best fixed K vs
no speculation, on the exact shapes of the NeMo-RL GB200 SyncRL recipes
(temperature 1.0, top_p 1.0, 32 generations per prompt, barrier per step),
with RedHatAI EAGLE3 Thinking drafters.

---

## Fixed-K3 is a 2x lever on 30B-A3B; DynamicSD matches but does not beat it

| Setting (TP) | baseline | fixed K3 | dynamic (capture-aware) |
|---|---|---|---|
| 30B-A3B math (openmath/math500/dapo, TP1) | 1.00x | **1.86-2.00x** | 1.71-1.90x |
| 30B-A3B swe_verified (TP1) | 1.00x | **1.85x** | 1.68x |
| 32B openmath (TP2) | 1.00x | 1.12x | 1.10x |
| 32B swe_verified (TP2) | 1.00x | 0.96x | **1.08x** |
| 32B math500 (TP2) | 1.00x | pending | 1.16x |
| 235B math500 (TP4) | 1.00x | 0.44x | 0.51x |
| 30B-A3B 40K long-tail (TP2, 32K gen) | 1.00x | **1.19x** | 0.63x |

Speedups are mean rollout-step wall-time ratios over 4 steps (2 for 40K).

At 4K generations the step spends most wall time at high concurrency, and the
per-BS optimum there is exactly K=3, so a fixed K=3 already sits on the
optimum; the dynamic schedule can only match it (1.87-1.90x vs 2.00x, the gap
is schedule-switching overhead plus the BS 86-127 capture-capped K4/K3 band).

**The one clear DynamicSD win is Qwen3-32B on SWE prompts: fixed-K3 is a net
loss (0.96x) while the derived schedule, which turns speculation off at
BS 256, converts it into a 1.08x gain.** This is DynamicSD working as
designed: not "more speculation", but insurance against speculating in
compute-bound regimes.

---

## The cudagraph capture cliff dominates naive schedules

K=5 profiling collapses at BS=128 (22.2k -> 8.0k tok/s on 30B-A3B): once
bs x (K+1) exceeds `max_cudagraph_capture_size` (512), decode falls back to
eager mode. Our first derived table carried K=5 into BS 86-127 and the
"dynamic" rollout ran *slower than fixed* (37.4s vs 25.4s per step). Profiled
grid points alone cannot see between-point cliffs; the table derivation now
caps K analytically (`bs x (K+1) <= capture budget`), which recovered dynamic
from 1.36x to 1.90x on openmath. **Any DynamicSD deployment must encode
hardware execution-mode boundaries, not just measured throughput points.**

## Deeper K is not the memory-bound answer

K=7 raises mean acceptance length to 4.11 (from 3.73 at K=5) but never beats
K=5 tokens/s, and at BS=1 plain K=3 is fastest (607 vs 590/556 tok/s):
per-position acceptance decays (0.82 / 0.66 / 0.53 / ...) faster than the
extra draft positions add value. The profiled schedules never select K>5.

## Probabilistic drafting adds nothing here

vLLM 0.24's `draft_sample_method="probabilistic"` left acceptance length
unchanged (2.99 vs 3.01 at K=3, temperature 1.0) and cost 3-10% tokens/s from
draft-logit caching. Greedy drafting remains the right default for these
EAGLE3 heads.

## MoE at scale inverts the sign

Qwen3-235B-A22B (TP4): despite healthy acceptance (AL 3.05 at temp 1.0),
fixed-K3 runs at **0.44x** - verifying K+1 tokens multiplies MoE expert
dispatch, which is already compute-bound at BS 64. The profiled schedule
correctly zeroes K at BS >= 64, yet the dynamic rollout still landed at 0.51x;
the residual gap is under diagnosis.

## Long-tail exposes the wrong index variable

The 40K preset was expected to be DynamicSD's showcase; it is its clearest
failure, and the logs say why. **EAGLE3 acceptance collapses with generation
depth**: shallow-phase AL is 2.3-2.5, but by the time median depth reaches
~10K tokens the cumulative draft acceptance rate is 2.9-3.7% (per-position
0.08 / 0.003 / 0.000). Fixed-K3 wins the shallow phase (step 0: 77s vs
baseline 106s) and evaporates in the deep phase (step 1: 186s vs 206s),
netting 1.19x. The dynamic schedule does worse (0.63x) because during the
drain the batch falls into its K=5 bands, paying 5-token drafting overhead at
~0% acceptance exactly where it was tuned to be aggressive - the schedule was
profiled at shallow depth (OSL 2048) and is indexed by batch size only.
**At long generation lengths the binding variable is sequence depth, not
batch size, and `num_speculative_tokens_per_batch_size` cannot express a
depth-aware schedule.** Raising max OSL further (64K) would widen, not close,
this gap; the fix is a depth-conditioned K (or drafters trained for deep
thinking contexts).

---

## Key takeaway

**On these RL-rollout shapes, EAGLE3 with a well-chosen fixed K is the
workhorse (up to 2.0x per-step wall time on Qwen3-30B-A3B at temperature 1.0),
and DynamicSD's value is asymmetric: it cannot beat a fixed K that already
sits on the optimum, but it converts speculation from a liability into a gain
where the optimum crosses zero (Qwen3-32B SWE at BS 256).** The derived
schedules reproduce the Cohere-reported structure (dense = monotonically
decreasing K, MoE = non-monotonic), and the practical lesson is that schedule
quality is bounded by profiling fidelity: capture-mode cliffs, deep-context
drift, and MoE dispatch costs all have to be encoded, or the schedule
confidently picks poisoned points.

Data: `data/` (profile grid, rollout summaries, drain curves). Plots:
`plots/`. Live page: `docs/dynamic_sd_sync_rollout_results_latest.html`.
Open items: 235B/40K dynamic regression diagnosis, 3 transient-node retries,
suffix-decoding composition.
