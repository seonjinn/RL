# PR 3294 latest-head Qwen3-30B Sync A/B

This experiment compares BF16 rollout with the full PR 3294 MXFP8 rollout
path on the same 4x4 GB200 synchronous colocated workload.

Both arms run 20 steps with CUDA Graphs, seed 42, global batch size 2048, and
both previous-policy and reference-policy logprob forwards. The BF16 control
uses the recipe's refit-safe Triton MoE backend. The MXFP8 arm quantizes routed
expert weights only and enables trainer prequantization, persistent 4 GiB CUDA
IPC buffers, batched expert shuffle, loader-route caching, and slim offload.

Report arithmetic means over steps 3 through 20 and retain source SHA,
container, backend, quantization scope, and W&B provenance with every result.

```bash
ACTION=test-only ARM=bf16 ./experiments/pr3294_latest_qwen30_sync_ab/submit_oci_hsg.sh
ACTION=test-only ARM=mxfp8 ./experiments/pr3294_latest_qwen30_sync_ab/submit_oci_hsg.sh
```
