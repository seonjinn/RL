# [Enhancement][PR draft] Auto-enable V2 model runner for Qwen3MoeForCausalLM; surface the DynamicSD piecewise downgrade

**Target**: vllm-project/vllm (v0.25.0)
**Components**: `vllm/config/vllm.py` (`DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES`,
`_maybe_override_dynamic_sd_cudagraph_mode`), docs

## Problem

DynamicSD's per-K FULL cudagraph capture (#45953) requires the V2 model
runner. `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES` currently contains
{DeepseekV2ForCausalLM, Qwen2MoeForCausalLM, GraniteMoeForCausalLM} - but not
`Qwen3MoeForCausalLM`, one of the most common speculative-decoding targets.
Without `VLLM_USE_V2_MODEL_RUNNER=1`, `_maybe_override_dynamic_sd_cudagraph_mode`
silently downgrades DynamicSD to PIECEWISE (a log-level warning only), while
fixed-K engines keep FULL graphs - so any fixed-vs-dynamic benchmark on
Qwen3-MoE is systematically biased against DynamicSD unless the user knows
the env var.

Measured effect on Qwen3-30B-A3B (GB200, GRPO-style sync rollout): FULL per-K
graphs are worth ~13% dynamic step-wall time (26.8s -> 23.4s per rollout
step) vs the piecewise fallback.

## Proposed changes

1. Add `"Qwen3MoeForCausalLM"` to `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES`
   (validated: Qwen3-30B-A3B runs V2 with per-K capture; no regressions
   observed on baseline/fixed-K runs on the same stack).
2. Make the DynamicSD piecewise downgrade prominent: docs note in the
   dynamic-spec-decoding page + consider a startup warning that names the
   env var when a schedule is present and V2 is off.
3. Docs: state explicitly that `num_speculative_tokens_per_batch_size`
   schedule quality is capture-budget-bound - schedules must respect
   `bs * (K+1) <= max_cudagraph_capture_size` per range or those batch sizes
   silently fall off the captured path (we measured a 40% throughput loss
   from one such uncapped range: 1.36x -> 1.90x after capping).
