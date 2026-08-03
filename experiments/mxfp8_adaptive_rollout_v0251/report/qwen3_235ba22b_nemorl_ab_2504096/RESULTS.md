# Qwen3-235B MXFP8 Adaptive Canary

Ptyche job `2504096` used vLLM 0.25.1, FlashInfer 0.6.13, two TP4/EP4
rollout replicas, and CUDA Graph execution. Every arm generated
262,144 output tokens on 8 GPUs.

| Backend policy | tokens/sec/GPU | vs CuTeDSL | Generation time (s) |
|---|---:|---:|---:|
| CuTeDSL | 100.31 | 1.000x | 326.665 |
| TRTLLM default | 89.82 | 0.895x | 364.813 |
| Complete-table adaptive | 92.92 | 0.926x | 352.662 |

Complete-table adaptive was `0.926x` versus CuTeDSL and
`1.034x` versus TRTLLM default. TRTLLM default was
`0.895x` versus CuTeDSL.

**Decision:** Do not enable the complete-table adaptive policy for this workload.

The offline table covered all five observed signatures and passed numerical and
CUDA Graph microbenchmark gates. This single matched run does not establish a
production gain, and the GSM8K promotion gate was therefore not executed.

## Provenance

```text
nemo_rl_commit=c8a512a52afc4cc15cc30c0b7b9d5bc012d7f5f6
custom_vllm_commit=658d7b1571a914bee7df48f717c2a428ee7c45ad
tactic_sha256=2b8121d1b56ccb44a4ee9bdb10adc5e355f58bf21e79079eadeb2ac7494bf417
```
