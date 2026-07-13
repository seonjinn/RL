# NeMo-RL Full-GRPO Online Drafter Comparison

| Variant | Steps | Draft refits | Acceptance | Delta pp | Step time | Step speedup | Gen tok/s/GPU speedup | Draft loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen8_online_pard2_canary | 2 | 2 | 0.000 | -29.179 | 25.900 | 0.874 | 0.905 | 2.060 |

Positive acceptance delta means the variant accepted more draft tokens than the baseline. Step speedup above 1.0 means the variant had lower mean step time.
