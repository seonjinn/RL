# NeMo-RL PARD CUDA Graph Regression Analysis

Updated: 2026-07-02 08:30 PDT

## Scope

This note analyzes the Qwen3-30B-A3B sync performance-recipe cohort with CUDA
Graphs enabled. All comparisons use steady-state Steps 2-20 for completed runs
or Steps 2-N for explicitly marked live runs. The matched baseline uses the same
model, recipe, temperature 1.0, top-p 1.0, max OSL 4096, TRITON_ATTN, Triton MoE
backend, max_num_seqs 128, and max_num_batched_tokens 32768.

## Current Evidence

| Method | Step span | Generation tok/s/GPU | Generation throughput speedup | Acceptance | Mean accepted length |
|---|---:|---:|---:|---:|---:|
| Baseline | 2-20 | 5,173.6 | 1.000x | n/a | n/a |
| Eagle-3 K5 | 2-20 | 4,695.6 | 0.908x | 50.1% | 3.51 |
| PARD K1 | 2-7 live | 3,482.7 | 0.673x | 79.6% | 1.80 |
| PARD K7 | 2-7 live | 2,734.5 | 0.529x | 35.6% | 3.49 |
| PARD K9 | 2-6 live | 2,355.5 | 0.455x | 28.0% | 3.52 |
| PARD K16 | 2-20 | 1,950.8 | 0.377x | 15.5% | 3.48 |

The live rows are preliminary, but the monotonic cost trend and the completed
K16 result are already strong enough to reject a CUDA-Graph-disabled hypothesis.

## Confirmed Runtime Facts

1. The workers run vLLM 0.20.0 with `enforce_eager=false`.
2. Runtime logs show both mixed prefill/decode and full decode CUDA Graph capture
   for PARD and Eagle-3 workers.
3. `parallel_drafting=true` is consumed by vLLM's `SpecDecodeBaseProposer`.
   The parallel path expands each request by K masked slots and executes one
   drafter forward pass, rather than K autoregressive drafter passes.
4. The PARD checkpoint contains `pard_token=151670`, so the parallel-drafting
   mask-token initialization path is active.
5. Qwen3-30B-A3B target and PARD drafter both use TP1 in this cohort. The
   regression is therefore not caused by draft TP communication.
6. Completed PARD K16 reward and generation-KL means match the baseline closely.
   The observed problem is performance, not an obvious sampling-correctness
   divergence.

## Architecture Cost Difference

| Component | Layers | Hidden size | Intermediate size | Notes |
|---|---:|---:|---:|---|
| Qwen3-30B-A3B target | 48 | 2,048 | 6,144 | MoE, 128 experts, 8 active |
| Eagle-3 drafter | 1 | 2,048 | 6,144 | Dedicated Eagle-3 proposer |
| PARD drafter | 28 | 1,024 | 3,072 | Dense Qwen3-0.6B with full vocabulary head |

PARD avoids K sequential drafter passes, but it does not make its single pass
free. At K16 and max_num_seqs 128, the drafter can process up to 2,048 masked
positions through 28 dense transformer layers plus a 151,936-token vocabulary
head. The target must then verify K+1 positions per request.

The accepted length saturates near 3.5 from K7 through K16. Increasing K beyond
7 therefore expands drafter and verifier work without increasing useful accepted
tokens. K1 is also slower because its 28-layer dense proposer cost is too high
relative to the already efficient high-concurrency MoE target decode.

## Root Cause

The current regression is caused by proposer and verification cost exceeding the
saved target decode work. CUDA Graphs and PARD's one-pass parallel-drafting path
are functioning. The large 28-layer dense proposer, full-vocabulary sampling,
high request concurrency, and K+1 target verification width dominate the saved
target iterations. Qwen3-30B-A3B makes this especially visible because only 3B
target parameters are active per token.

For Qwen3-32B, the generic draft-model path currently also requires draft TP2 to
match target TP2. That adds tensor-parallel communication to the small drafter and
is expected to be less efficient than a supported draft-TP1 path. This remains a
hypothesis until the queued K1/K7/K9/K16 TP2 cohort completes.

## Next Evidence Gates

1. Complete the Qwen3-32B target-TP2/draft-TP2 K1/K7/K9/K16 cohort and compare
   proposer sensitivity against Qwen3-30B-A3B.
2. Profile one baseline, PARD K1, PARD K7, and Eagle-3 K5 generation window to
   separate target forward, drafter forward, verification, sampler, and collective
   time.
3. Test a draft-TP1 proposer for Qwen3-32B only after the isolated implementation
   design is approved. The test must use separate compile caches/CUDA Graph state
   and broadcast proposals to target ranks.
4. Preserve greedy token equality and temperature-1 reward/KL checks before any
   optimized path is used for performance claims.

## Sources

- Lyris jobs: baseline `2250928`, Eagle-3 K5 `2250930`, PARD K1 `2260579`,
  PARD K7 `2260580`, PARD K9 `2260581`, and PARD K16 `2250929`.
- W&B project: `nvidia/sna-nemorl-specdec-lyris`.
- Runtime vLLM files: `vllm/v1/spec_decode/llm_base_proposer.py`,
  `vllm/v1/spec_decode/draft_model.py`, `vllm/v1/spec_decode/eagle.py`, and
  `vllm/config/speculative.py` from the job actor environment.
- Checkpoint configs: `amd/PARD-Qwen3-0.6B` and
  `RedHatAI/Qwen3-30B-A3B-Thinking-2507-speculator.eagle3`.
