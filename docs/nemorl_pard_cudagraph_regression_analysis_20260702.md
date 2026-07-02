# NeMo-RL PARD CUDA Graph Regression Analysis

Updated: 2026-07-02 09:10 PDT

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
| PARD K1 | 2-20 | 3,484.4 | 0.673x | 79.7% | 1.80 |
| PARD K7 | 2-19 live | 2,646.6 | 0.512x | 35.7% | 3.50 |
| PARD K9 | 2-17 live | 2,422.7 | 0.468x | 28.3% | 3.55 |
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
free. The parallel proposer retains the target query position and adds K masked
slots per request. At K16 and max_num_seqs 128, one drafter invocation can
therefore cover up to 2,176 positions through 28 dense transformer layers plus a
151,936-token vocabulary head. The target must then verify K+1 positions per
request.

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
match target TP2. The first strict TP2 K1/K7/K9/K16 cohort did not reach a timed
step: all four jobs failed deterministically during vLLM compilation because the
FlashInfer TRT-LLM fused all-reduce workspace was initialized for target hidden
size 5,120 and then reused for the PARD draft hidden size 1,024. At token count
32,768, vLLM rejected the workspace rather than risking an illegal memory access.
This is a separate operability bug from the Qwen3-30B-A3B performance regression.

## Next Evidence Gates

1. Re-run a matched Qwen3-32B baseline and PARD K1 smoke with only
   `compilation_config.pass_config.fuse_allreduce_rms=false`. CUDA Graphs remain
   enabled; this isolates the broken fusion without switching to eager mode.
   PARD K1 smoke `2261065` passed scheduler preflight `2261059`, then passed the
   prior 6m28s-6m44s failure window and completed draft TP2 CUDA Graph capture
   with the fusion disabled. Expand to a matched baseline and K7/K9/K16 only
   after the smoke reaches a complete timed step.
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
- Qwen3-32B failed TP2 jobs: K1 `2260695`, K7 `2260697`, K9 `2260699`, and
  K16 `2260701`.
- W&B project: `nvidia/sna-nemorl-specdec-lyris`.
- Runtime vLLM files: `vllm/v1/spec_decode/llm_base_proposer.py`,
  `vllm/v1/spec_decode/draft_model.py`, `vllm/v1/spec_decode/eagle.py`, and
  `vllm/config/speculative.py` from the job actor environment.
- Checkpoint configs: `amd/PARD-Qwen3-0.6B` and
  `RedHatAI/Qwen3-30B-A3B-Thinking-2507-speculator.eagle3`.
