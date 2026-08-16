# Current MXFP8 MoE Rollout A/B

This experiment compares BF16 training with BF16 rollout against BF16 training
with expert-only MXFP8 rollout. Both arms use FlashInfer TRTLLM MoE, NCCL
Reshard Refit, CUDA Graphs, the same model topology, and 20 GRPO steps.

Run the scheduler check before submission:

```bash
MODEL=qwen30 ARM=bf16 ACTION=test-only ./experiments/current_mxfp8_moe_rollout_ab/submit_oci_hsg.sh
MODEL=qwen30 ARM=mxfp8 ACTION=test-only ./experiments/current_mxfp8_moe_rollout_ab/submit_oci_hsg.sh
MODEL=nano ARM=bf16 ACTION=test-only ./experiments/current_mxfp8_moe_rollout_ab/submit_oci_hsg.sh
MODEL=nano ARM=mxfp8 ACTION=test-only ./experiments/current_mxfp8_moe_rollout_ab/submit_oci_hsg.sh
```

Set `ACTION=submit` only after the exact branch is pushed and checked out on the
cluster. The launcher writes the complete run provenance to `metadata.env`.
