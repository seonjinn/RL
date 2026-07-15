# [Bug][PR draft] DynamicSD + EAGLE3/MTP crashes at engine init: ZeroDivisionError in per-K cudagraph capture (drafter manager)

**Target**: vllm-project/vllm (bug present in v0.25.0 and `main` as of 2026-07-13)
**Component**: `vllm/v1/worker/gpu/cudagraph_utils.py` (`_init_candidates`)
**Introduced by**: #45953 (per-K full cudagraph capture for Dynamic SD)

## Symptom

Any engine with `speculative_config.num_speculative_tokens_per_batch_size`
set and an EAGLE3 (or MTP) drafter dies at initialization:

```
File ".../vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py", line 79, in init_cudagraph_manager
File ".../vllm/v1/worker/gpu/cudagraph_utils.py", line 234, in _init_candidates
File ".../vllm/utils/math_utils.py", line 22, in round_up
ZeroDivisionError: integer division or modulo by zero
```

Repro: Qwen3-30B-A3B + RedHatAI/Qwen3-30B-A3B-Thinking-2507-speculator.eagle3,
`{"method": "eagle3", "num_speculative_tokens": 5,
"num_speculative_tokens_per_batch_size": [[1,1,3],[2,85,5],[86,102,4],[103,128,3]]}`,
`VLLM_USE_V2_MODEL_RUNNER=1`. Any schedule containing a K different from max K
triggers it.

## Root cause

`_init_candidates` recovers the per-step sampled-token count as

```python
num_new_sampled_tokens_per_step = (
    self.decode_query_len - self.vllm_config.num_speculative_tokens
)
decode_query_lens = [
    x[2] + num_new_sampled_tokens_per_step for x in num_spec_per_batch_size
]
```

This is correct for the **target** manager (`decode_query_len == 1 + max_K`,
so `num_new == 1`) but the same class is instantiated for the **drafter**
manager, whose `decode_query_len == 1` (autoregressive drafting processes one
token per sequence per step regardless of K). There `num_new == 1 - max_K < 0`
and any schedule entry with `K == max_K - 1` produces a per-K query length of
0, used as a `round_up` divisor.

## Fix (6 lines)

The drafter's per-step query length does not vary with K, so per-K expansion
only applies to the target manager:

```python
if num_new_sampled_tokens_per_step > 0:
    decode_query_lens = [
        max(1, x[2] + num_new_sampled_tokens_per_step)
        for x in num_spec_per_batch_size
    ]
else:
    # drafter manager: query len does not vary with K
    decode_query_lens = [self.decode_query_len]
```

## Validation

With the fix, the same configuration initializes and runs; measured on
GB200: Qwen3-30B-A3B sync-rollout dynamic schedule reaches 2.01x over no-SD
(fixed-K3 2.19x on the same stack), and per-K FULL graphs verified captured.

Patch file: `patches/vllm0250_dynamic_sd_drafter_cudagraph_zerodiv.patch`.
Suggested test: engine-init smoke with a 2-range schedule where the second
range's K != max K, for both eagle3 and mtp methods.
