# [Bug][PR draft] Mamba/hybrid models reject DynamicSD per-K cudagraph capture (over-strict assert)

**Target**: vllm-project/vllm (v0.25.0)
**Component**: `vllm/v1/attention/backends/mamba_attn.py` (`build_for_cudagraph_capture`, ~line 183)
**Related**: #45953 follow-up - the Mamba backend was not updated for per-K capture

## Symptom

NemotronH (Mamba hybrid, e.g. nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8)
with in-checkpoint MTP + a DynamicSD schedule containing any K below max K
fails during cudagraph capture:

```
File ".../vllm/v1/attention/backends/mamba_attn.py", line 183, in build_for_cudagraph_capture
    assert m.max_query_len == 1 + self.num_spec_tokens  # decode-only
AssertionError
```

Schedules where every range uses K == max K pass by luck (that is why the
issue can hide in simple configs).

## Root cause

Per-K capture (V2 model runner) builds uniform-decode graphs for every K in
the schedule, i.e. `max_query_len = K + 1 <= 1 + max_K`. The Mamba builder
still asserts strict equality against the max-K shape only, even though the
preceding assert already allows `<=` and `self.build(0, m, ...)` sizes all
metadata from `m` itself (`num_accepted_tokens = torch.diff(m.query_start_loc)`
handles any uniform query length).

## Fix (1 line)

```python
-    assert m.max_query_len == 1 + self.num_spec_tokens  # decode-only
+    assert m.max_query_len <= 1 + self.num_spec_tokens  # decode-only
```

## Validation

With the fix (plus the drafter ZeroDivision fix in the companion PR),
Nemotron3-Super-120B (FP8, TP4, GB200) runs a two-range schedule
`[[1,127,3],[128,128,1]]` end-to-end; the dynamic schedule measured 1.53x
step-wall speedup vs no-SD, slightly ahead of fixed-K3 (1.50x) on openmath
sync rollouts, confirming the K=1 range at BS 128 is exercised correctly.
