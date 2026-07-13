# vLLM 0.25.0 local patches

## vllm0250_dynamic_sd_drafter_cudagraph_zerodiv.patch

vLLM 0.25.0 crashes at engine init (`ZeroDivisionError` in
`vllm/v1/worker/gpu/cudagraph_utils.py::_init_candidates`) whenever DynamicSD
(`num_speculative_tokens_per_batch_size`) is combined with an EAGLE3 drafter
and the V2 model runner. The per-K graph-capture code (PR #45953) recovers
`num_new_sampled_tokens_per_step = decode_query_len - max_K`, which is 1 for
the target manager but negative for the drafter manager
(`decode_query_len == 1`), producing a per-K query length of 0 that is used
as a `round_up` divisor. The drafter's per-step query length does not vary
with K (K only changes the loop count), so the fix keeps the single capture
shape for that manager.

Apply after installing vllm==0.25.0:

```bash
cd <venv>/lib/python3.12/site-packages
patch -p0 < vllm0250_dynamic_sd_drafter_cudagraph_zerodiv.patch
rm -rf vllm/v1/worker/gpu/__pycache__
```

Already applied to `/lustre/fsw/coreai_dlalgo_llm/users/sna/venvs/vllm025`
on Lyris (original kept as `cudagraph_utils.py.orig`). Upstream report
pending.
