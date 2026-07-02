# NeMo-RL PARD CUDA Graph Regression Analysis

Updated: 2026-07-02 15:48 PDT

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
| PARD K7 | 2-20 | 2,639.4 | 0.510x | 35.7% | 3.50 |
| PARD K9 | 2-20 | 2,439.2 | 0.472x | 28.1% | 3.54 |
| PARD K16 | 2-20 | 1,950.8 | 0.377x | 15.5% | 3.48 |

These completed rows establish the slowdown, but they do not by themselves
prove that every runtime shape uses a captured graph.

## Confirmed Runtime Facts

1. The workers run vLLM 0.20.0 with `enforce_eager=false`.
2. Runtime logs show mixed prefill/decode and full decode CUDA Graph capture for
   PARD and Eagle-3 workers, but the default largest captured shape is 512
   tokens. PARD expands one iteration to `max_num_seqs * (K + 1)` proposer
   tokens. With Qwen3-30B-A3B `max_num_seqs=128`, K7, K9, and K16 require
   1,024, 1,280, and 2,176 tokens respectively and therefore fall outside the
   default graph coverage.
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
tokens. K1 remains slower in the sync cohort even though its 256-token proposer
shape is covered, demonstrating a second intrinsic cost from the 28-layer dense
proposer and full-vocabulary head.

## Root Cause

The regression has two confirmed layers:

1. **Graph coverage:** the default 512-token capture limit leaves large PARD
   proposer shapes on an eager fallback. On Qwen3-32B with `max_num_seqs=64`,
   raising the cap from 512 to 640 recovered 1.317x generation throughput for K9
   and reached 0.989x of baseline over matched Steps 2-3. Raising it to 1,088 for
   K16 recovered only 1.074x and reached 0.701x of baseline.
2. **Intrinsic proposer and verifier cost:** the 28-layer dense drafter,
   full-vocabulary sampling, high concurrency, and K+1 target verification still
   dominate when acceptance saturates. The covered K1 sync result and the
   graph-covered K16 smoke prove that capture coverage alone is insufficient.

Qwen3-30B-A3B makes the intrinsic cost especially visible because only 3B target
parameters are active per token.

The Qwen3-30B-A3B async replacement cohort also exposed substantial run-to-run
variance. Its normalized r1 and r2 commands are identical after replacing only
run paths and W&B names, but r2 measured E2E throughput speedups of 1.055x,
1.026x, 0.968x, and 0.713x for K1, K7, K9, and K16. The original r1 values were
0.871x, 0.914x, 0.882x, and 0.684x. Because both waves use a single baseline from
a different node allocation and time window, final Qwen3-30B claims require a
same-wave baseline replicate rather than selecting the faster replicate.

For Qwen3-32B, the generic draft-model path currently also requires draft TP2 to
match target TP2. The first strict TP2 K1/K7/K9/K16 cohort did not reach a timed
step: all four jobs failed deterministically during vLLM compilation because the
FlashInfer TRT-LLM fused all-reduce workspace was initialized for target hidden
size 5,120 and then reused for the PARD draft hidden size 1,024. At token count
32,768, vLLM rejected the workspace rather than risking an illegal memory access.
This is a separate operability bug from the Qwen3-30B-A3B performance regression.

## Next Evidence Gates

1. Complete the Qwen3-32B graph-covered 20-step confirmations: K9 job `2262387`
   with cap 640 and K16 job `2262687` with cap 1,088. Both use the completed
   no-fused-all-reduce-RMS baseline `2261208`.
2. Run Qwen3-30B-A3B baseline and PARD K1/K7/K9/K16 as one time-matched cohort.
   K7/K9/K16 require capture caps 1,024/1,280/2,176 respectively. Do not use the
   existing cross-wave replicate spread as a final speedup claim.
3. Profile one baseline, PARD K1, PARD K7, and Eagle-3 K5 generation window to
   separate target forward, drafter forward, verification, sampler, and collective
   time.
4. Test a draft-TP1 proposer for Qwen3-32B only after the isolated implementation
   design is approved. The test must use separate compile caches/CUDA Graph state
   and broadcast proposals to target ranks.
5. Preserve greedy token equality and temperature-1 reward/KL checks before any
   optimized path is used for performance claims.

## Sources

- Lyris jobs: baseline `2250928`, Eagle-3 K5 `2250930`, PARD K1 `2260579`,
  PARD K7 `2260580`, PARD K9 `2260581`, and PARD K16 `2250929`.
- Qwen3-32B failed TP2 jobs: K1 `2260695`, K7 `2260697`, K9 `2260699`, and
  K16 `2260701`.
- Qwen3-32B graph-cap smoke jobs: K9 `2262236` and K16 `2262238`.
- Qwen3-30B-A3B async replacement jobs: K1 `2260883`, K7 `2260885`, K9
  `2260886`, and K16 `2260887`.
- W&B project: `nvidia/sna-nemorl-specdec-lyris`.
- Runtime vLLM files: `vllm/v1/spec_decode/llm_base_proposer.py`,
  `vllm/v1/spec_decode/draft_model.py`, `vllm/v1/spec_decode/eagle.py`, and
  `vllm/config/speculative.py` from the job actor environment.
- Checkpoint configs: `amd/PARD-Qwen3-0.6B` and
  `RedHatAI/Qwen3-30B-A3B-Thinking-2507-speculator.eagle3`.
