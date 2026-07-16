# vLLM 0.25.1 Eagle-3 Performance Recipe Experiment

This experiment compares the unmodified Qwen3-30B-A3B `4n4g` NeMo-RL
performance recipe against Eagle-3 K1/K3/K5/K7/K9. Both baseline and
speculative runs use Model Runner V2 and `FULL_AND_PIECEWISE` CUDA Graph mode.
The default `CAPTURE_PROFILE=native` leaves graph sizing to vLLM 0.25.1's
MRv2 implementation. `CAPTURE_PROFILE=compact` reproduces the earlier
`(K+1) * request_count` capture list for a controlled A/B comparison.

The launcher defaults to a two-step smoke test. Promote a passing matched pair
to `MAX_STEPS=20`; report step 2 through 20 to exclude initialization.

```bash
MODE=submit VARIANT=baseline MAX_STEPS=20 \
  bash experiments/vllm_0251_eagle3_perfcfg/submit_qwen30_eagle3_ptyche.sh

MODE=submit VARIANT=eagle3_k3 MAX_STEPS=20 CUDAGRAPH_METRICS=true \
  bash experiments/vllm_0251_eagle3_perfcfg/submit_qwen30_eagle3_ptyche.sh

MODE=submit VARIANT=eagle3_k3 MAX_STEPS=20 CAPTURE_PROFILE=compact \
  bash experiments/vllm_0251_eagle3_perfcfg/submit_qwen30_eagle3_ptyche.sh
```

DynamicSD is opt-in through `DYNAMIC_SD_SCHEDULE`. Its vLLM 0.25.1 source
patch only prevents target DynamicSD shapes from being incorrectly applied to
the autoregressive draft model's one-token CUDA Graph manager.
