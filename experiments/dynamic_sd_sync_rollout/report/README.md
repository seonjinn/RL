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
| 32B openmath / dapo (TP2) | 1.00x | **1.12-1.16x** | 1.06-1.10x |
| 32B math500 (TP2) | 1.00x | 1.13x | **1.16x** |
| 32B swe_verified (TP2) | 1.00x | 0.96x | **1.08x** |
| 235B all benches (TP4) | 1.00x | 0.31-0.44x | 0.47-0.53x (swe 0.30x) |
| 30B-A3B 40K long-tail (TP2, 32K gen) | 1.00x | **1.19x** | 0.63x |

Speedups are mean rollout-step wall-time ratios over 4 steps (2 for 40K).
On 235B, where speculation is a net loss everywhere, the dynamic schedule
consistently *mitigates* the damage vs fixed-K3 (0.47-0.53x vs 0.31-0.44x on
math benches) but cannot cross 1.0x - the right call there is no speculation
at all, which only an operator (or a K=0-everywhere schedule) can make.

**vLLM 0.24 caveat, resolved by a 0.25 pilot rerun:** 0.24 has a single
`uniform_decode_query_len = 1 + max K`, so any step where DynamicSD selects
K &lt; max falls off the FULL cudagraph path onto piecewise graphs, while a
fixed-K engine always runs FULL. PR #45953 (v0.25 + V2 model runner,
`VLLM_USE_V2_MODEL_RUNNER=1` required for Qwen3-MoE) captures a graph per K.
Rerunning the pilot points on 0.25 with per-K FULL capture verified:
30B openmath fixed 2.19x / dynamic 2.01x, 32B swe fixed 0.92x / dynamic
0.96x, 40K dynamic 0.67x. Dynamic gained ~13% wall time everywhere, but
fixed gained the same, so **the rankings are version-independent**: fixed-K3
still leads on 30B math, dynamic still leads fixed on 32B SWE (though both
dip below baseline on 0.25), and the 40K depth-collapse persists. Bonus
finding: vLLM 0.25.0 crashes at engine init (ZeroDivisionError in the
drafter's per-K cudagraph manager, `gpu/cudagraph_utils.py`) for any
DynamicSD schedule - we run with a local one-line guard patch; upstream
report pending.

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

We prototyped that fix as a ~15-line scheduler patch on vLLM 0.25 (cap K to 0
once mean generated depth exceeds a threshold; `patches/`). The concept is
not enough on its own: the first run fell into a *third* capture-coverage
trap (a runtime K=0 has query_len 1, which the per-K capture list does not
include), and after adding the K=0 shape the V2 dispatcher mis-matched
speculative batches and slowed even the shallow phase (191s vs 116s step 0).
Depth-aware K therefore needs a dispatch-aware upstream implementation, not a
scheduler-only monkey-patch. On the same 0.25 stack the 40K standings are:
fixed-K3 **1.33x**, dynamic 0.68x, depth-capped dynamic 0.35-0.42x.

---

## Nemotron3 MTP: built-in drafting flips the big-MoE verdict

Nemotron3 Super 120B-A12B (FP8, TP4) and Ultra 550B-A55B (NVFP4, TP4) ship a
single in-checkpoint MTP module; vLLM reuses it chained for K>1
(`{"method": "mtp", "num_speculative_tokens": K}`). Two surprises:

1. **Chained reuse does not decay the way an external head does**: K=3 mean
   acceptance length is 2.96 (Super) / 3.00 (Ultra) at temperature 1.0 -
   equal to the separately-trained EAGLE3 drafters on Qwen3.
2. **The big-MoE sign flips.** Same-shape sync rollouts (temp 1.0):

| Setting | baseline | fixed K3 (MTP) | dynamic |
|---|---|---|---|
| Super 120B openmath | 37.2s | 1.50x | **1.53x** |
| Super 120B swe_verified | 48.3s | **1.47x** | 1.43x |
| Ultra 550B openmath | 67.8s | **1.75x** | 1.75x (schedule is K3-everywhere) |
| Ultra 550B swe_verified | 70.7s | **1.56x** | 1.55x |

Qwen3-235B-A22B with an external EAGLE3 drafter was a net loss (0.31-0.44x);
Nemotron3-Ultra at 2.4x the parameter count gains 1.56-1.75x. **"SpecDec
does not pay at MoE scale" was a statement about external drafters, not about
speculation** - the in-checkpoint MTP head shares the target's backbone and
quantization, eliminating the dispatch-heavy separate-drafter overhead.
DynamicSD adds nothing for Ultra because its profiled optimum is K=3 at every
batch size (the schedule degenerates to fixed-K); Super's schedule has a
K1/K2 range at BS 128, whose dynamic run needed one more vLLM patch (Mamba
per-K capture assert, ledger #7). With that fix, Super openmath is the first
math setting where dynamic edges out fixed-K (1.53x vs 1.50x) - the K1 range
at BS 128 pays for itself.

## MTP survives depth; long-context gains die elsewhere

Super 120B long-tail runs (32K / 64K max_tokens, openmath): **MTP acceptance
holds at depth** - AL stays 2.0-2.6 at median generated depth 4-7K, where
EAGLE3 had already collapsed to per-position 0.08. The in-checkpoint head
tracks the target distribution independent of depth. Yet the 32K rollout
shows **no net speedup (fixed 1.01x, dynamic 1.00x vs 1.50x at 4K)**: with
acceptance intact, the remaining suspect is verification cost - the Mamba
hybrid must roll back / recompute state for speculative verification, and
that overhead appears to grow with context until it cancels the acceptance
gains. This is a different failure mode from EAGLE3's acceptance collapse and
needs a per-phase breakdown to confirm. The 64K numbers (fixed 0.56x, dynamic
0.29x) are not interpretable: with only 2 steps x 64 seqs at temperature 1.0,
sampled length distributions diverge across variants (p50 3.9K vs 5.9K) and
dominate wall time; a matched-length control or more steps is required.

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
Every vLLM patch/change and its measured perf impact is tracked in
[`../PATCH_LEDGER.md`](../PATCH_LEDGER.md) for upstream PR conversion.
Open items: 235B/40K dynamic regression diagnosis, 3 transient-node retries,
suffix-decoding composition.
