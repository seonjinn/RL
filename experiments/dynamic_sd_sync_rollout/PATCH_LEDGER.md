# DynamicSD Patch & Change Ledger

Running log of every change we made to make vLLM DynamicSD work well, with the
measured performance impact of each. Purpose: each entry should be directly
convertible into an upstream vLLM issue/PR with its evidence attached. Update
this file whenever a new change lands or new measurements arrive.

Baseline workload for all numbers unless stated: Qwen3-30B-A3B + EAGLE3
Thinking drafter, openmath sync-rollout (4x32 gens, temp 1.0, GB200 TP1),
mean step-wall speedup vs no-SD baseline.

| # | Change | Layer | Perf impact (measured) | Upstream status |
|---|--------|-------|------------------------|-----------------|
| 1 | Capture-aware K-table derivation: cap K so bs x (K+1) <= max_cudagraph_capture_size when building `num_speculative_tokens_per_batch_size` | our tooling (`derive_dynamic_k_table.py --max-capture-tokens`) | dynamic **1.36x -> 1.90x** (+40%) on vLLM 0.24 | PR-able as: validation warning / auto-cap in `SpeculativeConfig`, + docs. Not sent yet |
| 2 | vLLM 0.25.0 crash fix: drafter cudagraph manager ZeroDivisionError under DynamicSD | vLLM patch (`vllm/v1/worker/gpu/cudagraph_utils.py`, `patches/vllm0250_dynamic_sd_drafter_cudagraph_zerodiv.patch`) | blocker -> runs at all; unlocked #3 | **Bug present in upstream main (checked 2026-07-13). Strong PR candidate, 1-line + test** |
| 3 | vLLM 0.25 + V2 model runner (per-K FULL cudagraph, PR #45953) with `VLLM_USE_V2_MODEL_RUNNER=1` | config discovery (no code) | dynamic **1.90x -> 2.01x** (step wall 26.8 -> 23.4s, -13%); fixed-K3 also 2.00x -> 2.19x | PR-able as: docs fix + auto-enable Qwen3MoeForCausalLM in `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES` (currently silent PIECEWISE downgrade) |
| 4 | K=0 capture extension: add plain decode shape to per-K capture set | vLLM patch, **REVERTED** | negative: V2 dispatcher mis-matched spec batches; 40K step-0 wall 116s -> 191s; overall 0.35x -> 0.42x only | Do not send as-is. Evidence for #6 design discussion |
| 5 | Depth-aware K cap: scheduler caps K when mean generated depth > threshold (`VLLM_DYNAMIC_SD_DEPTH_THRESHOLD_TOKENS`) | vLLM patch, prototype (`patches/vllm0250_depth_aware_dynamic_sd.patch`), disabled | negative in current form: 40K dynamic 0.68x -> 0.35-0.42x (runtime K=0 shape uncaptured / dispatch mismatch) | Feature request material: depth-conditioned K needs dispatch-aware upstream design. Attach 40K depth-collapse data (acceptance 2.9-3.7% at depth 10K+) |
| 6 | (open) Dispatch-correct runtime K=0 / off-schedule K support | upstream design gap | blocks #5; also affects any schedule with K=0 ranges under V2 | Issue candidate with #4/#5 measurements |

---

## Detail per change

### 1. Capture-aware table (tooling)

The profiled BS grid {1,2,...,64,128} cannot see that bs x (K+1) crosses the
512-token cudagraph capture budget between grid points (K=5 fine at BS 64,
eager fallback from BS 86). The naive argmax table carried K=5 into BS 86-127
and dynamic ran slower than fixed (37.4s vs 25.4s per step). Enforcing
`K <= 512/bs - 1` analytically in the derivation recovered it. Upstream angle:
vLLM accepts such schedules silently; a validation warning or auto-cap at
`SpeculativeConfig` level would prevent the whole failure class.

### 2. Drafter ZeroDivisionError (vLLM 0.25.0 bug)

`_init_candidates()` recovers `num_new_sampled = decode_query_len - max_K`,
correct for the target manager (query_len = 1+max_K) but negative for the
drafter manager (query_len = 1), yielding a per-K query length of 0 used as a
`round_up` divisor. Any DynamicSD schedule + EAGLE3/MTP crashes at engine
init. Fix: only expand per-K lens when `num_new > 0`. Verified still broken
in upstream main as of 2026-07-13.

### 3. V2 runner requirement (docs/config gap)

0.24 has a single `uniform_decode_query_len = 1 + max_K`; any dynamic K < max
silently runs piecewise while fixed-K runs FULL - a systematic handicap on
every 0.24 dynamic number. 0.25 fixes this only under Model Runner V2, and
Qwen3MoE is not in the auto-enable list, so without
`VLLM_USE_V2_MODEL_RUNNER=1` vLLM silently downgrades DynamicSD to piecewise
(warning only in logs). Measured effect of getting FULL per-K graphs:
-13% dynamic step wall.

### 4-5. Negative results (kept for upstream design discussion)

Depth-aware K is the right diagnosis for 32K long tails (EAGLE3 acceptance
collapses to ~3% beyond ~10K generated tokens; batch-size-indexed schedules
cannot express depth), but a scheduler-only patch is insufficient: a runtime
K=0 needs its decode shape captured, and naively adding that shape makes the
V2 dispatcher mis-match speculative batches. The correct implementation must
coordinate schedule, capture set, and dispatch keys - an upstream-level
change.

---

## Cumulative effect (Qwen3-30B-A3B openmath rollout, dynamic variant)

| Stage | dynamic speedup | step wall |
|---|---|---|
| naive table, vLLM 0.24 | 1.36x | 37.4s |
| + capture-aware table (#1) | 1.90x | 26.8s |
| + 0.25 crash fix (#2) + V2 per-K graphs (#3) | **2.01x** | 23.4s |
| reference: best fixed-K3 on same stack | 2.19x | 21.5s |

Net: **+48% dynamic throughput from our changes**; remaining 8% gap to
fixed-K3 is workload-structural (K=3 already optimal at the dominant batch
sizes), not an implementation deficit we have identified.
