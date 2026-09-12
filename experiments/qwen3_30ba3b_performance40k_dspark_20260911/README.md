# Qwen3-30B-A3B native performance-40K DSpark study

This cohort uses
`examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n8g-40K.yaml`
as the authoritative workload source. It preserves the recipe's OpenMath data,
40,960-token context, GBS/rollout settings, and TP4/EP8/CP8 parallelism.

OCI-HSG has four GPUs per node, so the physical allocation maps 4 nodes x 8
GPUs to 8 nodes x 4 GPUs while preserving the 32-GPU world size. Both baseline
and DSpark use BF16, `flashinfer_trtllm`, vLLM TP2, and FULL_AND_PIECEWISE CUDA
Graphs. The only method difference is the frozen step-44000 DSpark K3/K5 arm.

This cohort must not be mixed with the sibling DAPOMath17K 47K/49K cohort:

- performance-40K: OpenMathInstruct-2, `hf_math_verify`, 40,960 tokens
- DAPO47K: DAPOMath17K, `dapo_math_verify`, 47,104 generation / 49,152 total

## Launch protocol

```bash
bash submit_matrix.sh --test-only
bash submit_matrix.sh --submit-gates
# Only after all gates complete successfully:
bash submit_matrix.sh --submit-20
```
