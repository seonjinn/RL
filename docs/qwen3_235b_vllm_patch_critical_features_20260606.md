# Qwen3-235B vLLM Patch-Version Critical Feature Notes

Date: 2026-06-06 PDT

This note summarizes vLLM runtime changes or feature areas that can plausibly
affect Qwen3-235B speculative decoding performance when comparing our current
standalone path and NeMo-RL path.

## Runtime Split Observed In Our Runs

| Path | Runtime observed | Why it matters |
|---|---:|---|
| vLLM standalone PARD/OpenMath/high-batch Qwen3-235B | vLLM v0.20.2 | This is the path where we observed the stronger standalone PARD results. |
| NeMo-RL Qwen3-235B PARD/PARD-style sync/full-GRPO path | mostly vLLM 0.17.0 extracted site | This path is not version-identical to standalone, so lower speedup cannot be attributed only to the RL loop until a matched control is run. |
| Current provider-MoE freshvenv job 3195285 | not locally proven | The local manifest has an empty source_vllm_site field, so the imported vLLM version must be checked from remote logs. |

## Critical Feature Areas

| Priority | Feature area | Performance relevance | What to check |
|---|---|---|---|
| P0 | PARD / parallel draft model support | Directly affects draft/verify scheduling, accepted-token accounting, and draft model overhead. vLLM v0.20 docs expose PARD via `method=draft_model` and `parallel_drafting=true`. | Run the same Qwen3-235B PARD K3/K5 prompt set on standalone v0.17 and v0.20.2, or make a NeMo-RL v0.20.2 import probe. |
| P0 | Scheduler and chunked prefill behavior | SpecDec creates draft plus verification work. Decode-priority scheduling, chunked prefill, `max_num_batched_tokens`, and `max_num_seqs` can change whether accepted tokens turn into throughput. | Match `max_num_batched_tokens`, `max_model_len`, batch size, and prompt/output lengths across standalone and NeMo-RL. |
| P0 | MoE kernels, expert parallelism, and EPLB | Qwen3-235B-A22B is MoE. Target verification cost can dominate, and expert imbalance can hide draft benefits. Newer EP/EPLB and MoE backend choices can shift speedup. | Record `enable_expert_parallel`, `enable_eplb`, all2all backend, MoE backend, TP/DP/EP shape, and expert load metrics if available. |
| P1 | CUDA graph / compile / eager mode | Decode loops are overhead-sensitive. If NeMo-RL forces eager mode while standalone uses graph/compile optimizations, similar acceptance can still produce lower throughput. | Compare `enforce_eager`, compilation config, cudagraph capture sizes, and vLLM cudagraph metrics. |
| P1 | KV cache, context parallel, and DCP | Long-context Qwen3-235B runs are sensitive to KV duplication and memory pressure. Context/DCP settings can affect feasible batch size and decode throughput. | Record TP, DCP/CP settings, KV cache dtype, GPU memory utilization, and maximum concurrency. |
| P1 | Blackwell/CUDA/PyTorch/FlashInfer stack | GB200 performance depends on the container's CUDA, PyTorch, FlashInfer, and custom kernel compatibility. vLLM v0.20-era images may use a materially different stack than extracted v0.17. | Capture `vllm.__version__`, CUDA, PyTorch, FlashInfer, GPU arch, and container image for every run. |
| P2 | SpecDec metrics and observability | Metrics do not speed up runs, but they determine whether we can distinguish low acceptance from runtime overhead. | Always collect acceptance rate, mean accepted length, draft tokens, accepted tokens, emitted tokens, generation time, and E2E step time. |
| P2 | Additional speculator families | DFlash and other new speculator paths may require newer source/runtime support. Our stock 0.17 probe only exposed EAGLE3-style support for DFlash. | Do not mix DFlash/PARD conclusions unless runtime and checkpoint compatibility are proven separately. |

## Current Interpretation

Version skew is a credible systems variable, but it is not yet the sole root
cause. The clean decision test is:

1. If acceptance is similar on v0.17 and v0.20.2, but throughput is worse on
   v0.17/NeMo-RL, focus on runtime/scheduler/MoE overhead.
2. If acceptance differs, first fix prompt/sampling/drafter/config alignment.
3. If generation improves but E2E does not, split the NeMo-RL step into
   generation, reference/logprob, reward, training, and orchestration time.

## Related Files

- `docs/qwen3_235b_runtime_version_matrix_20260606.md`
- `docs/qwen3_235b_next_experiment_runbook_20260606.md`
- `docs/qwen3_235b_team_report_20260606.md`

## External Reference Points

- vLLM v0.20 PARD documentation describes `parallel_drafting=true` with
  `method=draft_model`.
- vLLM optimization documentation describes V1 chunked prefill and decode
  prioritization, controlled by `max_num_batched_tokens`.
- vLLM EP/EPLB documentation describes expert parallelism and load balancing
  for MoE models.
