# PR 3652 Qwen3-30B Async BF16/MXFP8 A/B

This matched 4x4 GB200 experiment compares BF16 and routed-expert-only MXFP8
rollout in Async GRPO. Both arms use NCCL Reshard, CUDA Graphs, 20 steps, seed
42, and both previous-policy and reference-policy logprob forwards.

The MXFP8 arm disables trainer prequantization. It uses PR 3477's receiver-side
BF16-to-MXFP8 conversion after NCCL resharding. PR 3652 keeps `lm_head`, QKVO,
routers, and other non-expert families outside the MXFP8 scope.

Report steps 3 through 20 with source SHA, container, backend, scope, W&B URL,
and the included metric counts.
