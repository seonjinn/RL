# Qwen3-235B Async BF16 versus MXFP8

This experiment compares BF16 and routed-expert-only MXFP8 rollout with the
same Qwen3-235B Async GRPO workload on 32 four-GPU GB200 nodes. Both arms use
16 generation nodes, NCCL Reshard refit, CUDA Graphs, 20 steps, seed 42, and
both previous-policy and reference-policy logprob computation. Report steps
3-20 after both arms complete.
