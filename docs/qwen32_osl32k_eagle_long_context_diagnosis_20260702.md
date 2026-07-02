# Qwen3-32B 32K Eagle Long-Context Diagnosis

Date: 2026-07-02

## Setup

- Recipe: `examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml`
- Mode: sync
- Cluster: Pretyche GB200, 4 nodes x 4 GPUs, `segment=4`
- Sampling: temperature `1.0`, top-p `1.0`
- Maximum output length: `32,768`
- Batch shape: 16 prompts x 16 generations, global batch size 256
- Context parallel size: 2
- CUDA Graphs: enabled (`enforce_eager=false`)
- Attention/MoE: `TRITON_ATTN` / Triton MoE
- Comparison window: matched W&B Steps 2-4

## Matched Results

| Method | Mean generated tokens | Max generated tokens | Acceptance | Mean accepted length | Verifier work amplification | Generation throughput speedup | E2E throughput speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 8,053 | 32,682 | n/a | 1.00 | 1.00x | 1.000x | 1.000x |
| Eagle K5 | 7,975 | 32,676 | 28.4% | 2.42 | 2.48x | 0.742x | 0.750x |
| Eagle K7 | 8,083 | 32,674 | 20.9% | 2.47 | 3.24x | 0.740x | 0.758x |
| Eagle K9 | 8,087 | 32,672 | 16.4% | 2.48 | 4.03x | 0.755x | 0.769x |

Verifier work amplification is `(K + 1) / mean accepted length`. It estimates
the target query-token work required per emitted token before accounting for
parallel-kernel efficiency.

The generated-token means differ from baseline by at most 1.0%, and all long
steps reach nearly the same 32K maximum. The throughput regression is therefore
not explained by shorter or longer sampled outputs.

## Step-Level Evidence

| Method | Step | Mean generated tokens | Generation time | Generation tok/s/GPU |
|---|---:|---:|---:|---:|
| Baseline | 2 | 3,498 | 322.4s | 178.4 |
| Baseline | 3 | 10,465 | 983.6s | 172.0 |
| Baseline | 4 | 10,197 | 1,234.8s | 133.4 |
| Eagle K5 | 2 | 3,476 | 358.2s | 159.6 |
| Eagle K5 | 3 | 10,332 | 1,333.2s | 125.3 |
| Eagle K5 | 4 | 10,118 | 2,211.0s | 74.0 |
| Eagle K7 | 2 | 3,572 | 421.6s | 139.2 |
| Eagle K7 | 3 | 10,385 | 1,347.2s | 124.6 |
| Eagle K7 | 4 | 10,292 | 1,763.7s | 94.3 |
| Eagle K9 | 2 | 3,614 | 372.0s | 159.6 |
| Eagle K9 | 3 | 10,507 | 1,458.1s | 116.5 |
| Eagle K9 | 4 | 10,141 | 1,840.7s | 89.0 |

The degradation grows on the 10K-token steps. This is consistent with target
verification becoming KV-attention-bandwidth dominated as the active context
grows. Each speculative iteration verifies `K + 1` query tokens, while the
observed mean accepted length remains near 2.4 tokens.

## E2E Breakdown

Across Steps 2-4, non-generation stages remain close:

| Method | Prepare generation | Policy training | Policy/reference logprobs | Generation |
|---|---:|---:|---:|---:|
| Baseline | 10.7s | 36.0s | 33.2s | 846.9s |
| Eagle K5 | 10.8s | 37.6s | 34.4s | 1,300.8s |
| Eagle K7 | 11.6s | 37.7s | 32.5s | 1,177.5s |
| Eagle K9 | 11.0s | 36.9s | 33.4s | 1,223.6s |

The E2E regression is concentrated in rollout generation rather than policy
training, weight preparation, or logprob evaluation.

## Current Code Limitation

The existing runtime dynamic-draft patch in
`nemo_rl/models/generation/vllm/specdec_runtime_gate_patch.py` only selects a
smaller K when `drafter.parallel_drafting` is true. That covers PARD-style
parallel drafting but not the sequential Eagle proposer. Its tiers are also
selected from scheduled request and scheduled token counts; they do not use
the active sequence/context length. The current mechanism therefore cannot
directly correct the observed Eagle long-context regression.

## Current Conclusion

The first three matched steady-state steps show a real 23-26% generation
throughput regression for Eagle K5/K7/K9 at 32K OSL. Low acceptance combined
with long-context verification work is the leading explanation. The full
Steps 2-20 result is still required to quantify the final effect and determine
whether the additional step-to-step degradation is entirely context-length
driven or includes a persistent runtime-state cost.
