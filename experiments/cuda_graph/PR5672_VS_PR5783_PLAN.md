# Llama-3.1-8B packed CUDA graph comparison

This experiment compares the existing packed CUDA graph implementation with the
Megatron-LM PR #5672 adapter while holding the NeMo-RL baseline and container
constant.

| Condition | NeMo-RL worktree | Megatron-LM | Scope |
| --- | --- | --- | --- |
| `nocg` | PR #5783 baseline | `725933134` | none |
| `current-attn` | PR #5783 baseline | `725933134` | ATTN |
| `current-attn-mlp` | PR #5783 baseline | `725933134` | ATTN + MLP |
| `pr5672-attn` | PR #5672 adapter | `28cfc6ae7` | ATTN |
| `pr5672-attn-mlp` | PR #5672 adapter | `28cfc6ae7` | ATTN + MLP |

Both graph conditions use sequence packing, a 4096-token static shape, a
4-GPU Lyris GB200 node, Transformer Engine 2.15, and
`cuda_graph_max_packed_seqs: 16`. The adapter pads graph-facing `cu_seqlens`
to 17 entries and passes a THD sample to `TECudaGraphHelper` at capture.

Run the five conditions for 20 steps first. Compare post-warmup policy train
time, policy train tokens/s, end-to-end step time, and graph capture/replay
messages. If the adapter has no correctness failures, run the same matrix for
100 steps and compare reward, loss, KL, entropy, and response length at the
same optimizer step.

The initial Lyris scheduling preflight has no eligible GB200 capacity; it
reported an estimated start of 2027-07-15. Jobs are intentionally not queued
until capacity becomes available.
