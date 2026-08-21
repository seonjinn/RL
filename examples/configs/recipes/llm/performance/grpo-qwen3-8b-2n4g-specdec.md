# Qwen3-8B Math speculative decoding performance recipe

`grpo-qwen3-8b-2n4g-specdec.yaml` is a matched Qwen3-8B Sync GRPO recipe for
`OpenMathInstruct-2` with the `hf_math_verify` reward. It uses two GB200 nodes
with four GPUs per node and keeps vLLM CUDA Graph enabled. The registered
performance run defaults to public DFlash K7 so the suite exercises speculative
decoding rather than silently measuring a no-spec control.

Run the default DFlash K7 recipe:

```bash
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/performance/grpo-qwen3-8b-2n4g-specdec.yaml
```

Select a different compatible DFlash or DSpark checkpoint and K using the same
environment contract as the SWE SpecDec recipe. The pinned vLLM 0.25.1 schema
supports both method names and `num_speculative_tokens`:

```bash
NRL_SPEC_METHOD=dspark \
NRL_DRAFT_MODEL=/path/to/compatible/qwen3-8b-dspark-drafter \
NRL_NUM_SPECULATIVE_TOKENS=5 \
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/performance/grpo-qwen3-8b-2n4g-specdec.yaml
```

Run the matched no-spec control by clearing only `speculative_config`:

```bash
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/performance/grpo-qwen3-8b-2n4g-specdec.yaml \
  policy.generation.vllm_kwargs.speculative_config=null
```

Keep the target model, sampling settings, dataset, reward, topology, and CUDA
Graph setting unchanged when comparing the three modes. The drafter must be
compatible with the Qwen3-8B target and selected method.
