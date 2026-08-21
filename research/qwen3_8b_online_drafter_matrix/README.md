# Qwen3-8B Matched Online-Drafter Matrix

This experiment compares target-only GRPO, fixed DFlash K5/K7, and online
DFlash K5/K7 while holding the training and rollout configuration fixed.

The shared contract is DAPOMath17K, seed 42, CP1, TP2/DP2, GBS32,
PPS8/GPS4, sequence packing disabled, sequence parallel disabled, 4K total
context, BF16, and PIECEWISE CUDA Graph capture. Each arm uses one persistent
W&B run in `nvidia/sna-nemo-rl-online-drafter` and a fail-closed
gate→350→700→1000 checkpoint chain.

DSpark K5/K7 use the same matrix contract in their provider branch after the
corrected sampled-label and weighted-normalization objective passes OCI.
