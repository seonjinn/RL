# BF16 Triton versus FlashInfer TRTLLM refit

This experiment validates PR 3659 with a matched 20-step Qwen3-30B-A3B Async
GRPO pair on four four-GPU GB200 nodes. Both arms use BF16 training and rollout,
NCCL Reshard, CUDA Graphs, two generation nodes, seed 42, and both previous-policy
and reference-policy logprob computation. The only intended difference is the
MoE backend and the native layerwise reload path that it selects.
