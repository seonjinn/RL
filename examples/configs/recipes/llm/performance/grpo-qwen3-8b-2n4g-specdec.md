# Qwen3-8B Math speculative decoding performance recipe

`grpo-qwen3-8b-2n4g-specdec.yaml` is a matched Qwen3-8B Sync GRPO recipe for
`OpenMathInstruct-2` with the `hf_math_verify` reward. It uses two GB200 nodes
with four GPUs per node and keeps vLLM CUDA Graph enabled. The default run is
the no-speculative-decoding control.

Run the no-spec control:

```bash
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/performance/grpo-qwen3-8b-2n4g-specdec.yaml
```

Add a fixed DFlash or DSpark drafter with Hydra overrides. The pinned vLLM
0.25.1 schema supports both method names and `num_speculative_tokens`:

```bash
SPEC_METHOD=dflash  # or dspark
DRAFT_MODEL=/path/to/compatible/qwen3-8b-drafter
SPEC_K=7

uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/performance/grpo-qwen3-8b-2n4g-specdec.yaml \
  ++policy.generation.vllm_kwargs.speculative_config.method="$SPEC_METHOD" \
  ++policy.generation.vllm_kwargs.speculative_config.model="$DRAFT_MODEL" \
  ++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens="$SPEC_K" \
  ++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1
```

Keep the target model, sampling settings, dataset, reward, topology, and CUDA
Graph setting unchanged when comparing the three modes. The drafter must be
compatible with the Qwen3-8B target and selected method.
