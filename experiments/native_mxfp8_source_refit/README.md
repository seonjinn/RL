# Qwen3-235B MXFP8 Parameter Storage Validation

This experiment compares two MXFP8-training sources with the same MXFP8
FlashInfer TRTLLM rollout and NCCL Reshard refit path:

| Arm | Training compute | Training parameter storage | Rollout |
| --- | --- | --- | --- |
| `fp8_param=false` | routed-expert MXFP8 | BF16 | routed-expert MXFP8 |
| `fp8_param=true` | routed-expert MXFP8 | MXFP8 values and E8M0 scales | routed-expert MXFP8 |

Attention, routers, and `lm_head` remain BF16 in both arms. The 20-node smoke
keeps the 16-node policy topology and reduces rollout to four nodes. It also
uses two prompts and four generations per prompt so that a 20-step functional
run finishes sooner. The 32-node configs retain the performance recipe's full
batch and rollout topology.

Both arms use the same NeMo RL commit, Bridge revision, Megatron-LM router
padding fix, model cache, container, and W&B project. Build and compile caches
are stored on node-local `/raid/scratch`; only the immutable container, model,
logs, and final results are stored on `/lustre`.

## Lyris

Run the scheduler preflight before each submission:

```bash
MODEL=qwen235smoke FP8_PARAM=false MAX_STEPS=20 ACTION=test-only \
  ./experiments/native_mxfp8_source_refit/submit_lyris_qwen235.sh
MODEL=qwen235smoke FP8_PARAM=true MAX_STEPS=20 ACTION=test-only \
  ./experiments/native_mxfp8_source_refit/submit_lyris_qwen235.sh
```

Submit the two functional arms:

```bash
MODEL=qwen235smoke FP8_PARAM=false MAX_STEPS=20 ACTION=submit \
  ./experiments/native_mxfp8_source_refit/submit_lyris_qwen235.sh
MODEL=qwen235smoke FP8_PARAM=true MAX_STEPS=20 ACTION=submit \
  ./experiments/native_mxfp8_source_refit/submit_lyris_qwen235.sh
```

Use `MODEL=qwen235` for the full 32-node topology. Accept an arm only after all
20 steps finish with finite loss, reward, entropy, and generation KL metrics.
For `fp8_param=true`, the logs must also show native MXFP8 component planning
and receiver reload without BF16 receiver quantization of native components.
