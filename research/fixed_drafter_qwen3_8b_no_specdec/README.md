# No-SpecDec Qwen3-8B control arm

This 100-step control arm is the K=0 ablation for the fixed public DFlash experiment. It preserves the completed DFlash K=15 resolved training, sampling, target, tokenizer, precision, and topology contract while removing speculative decoding. `policy.draft.enabled` remains false, and no drafter checkpoint is loaded or refit.

The parity validator compares resolved configs and fails on any difference outside speculative decoding, arm/run/output/W&B identity, and the 100-step run length. The run uses DAPOMath17K with seed 42, 8 prompts by 4 generations, global batch 32, microbatch 1, BF16 policy TP2/PP1/CP1 with sequence parallelism, and vLLM TP1. TensorBoard and W&B are enabled under the shared fixed-drafter sweep group with the `no-specdec` and `k0` tags.
