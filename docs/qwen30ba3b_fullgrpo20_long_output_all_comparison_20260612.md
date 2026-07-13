# NeMo-RL Full-GRPO Online Drafter Comparison

| Variant | Steps | Draft refits | Acceptance | Delta pp | Step time | Step speedup | Gen tok/s/GPU speedup | Draft loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_nospec_longosl_batchlong_retry1 | 19 | 0 |  |  | 880.775 | 1.011 | 1.003 |  |
| static_pard2_win2048 | 19 | 0 | 49.165 |  | 489.966 | 1.818 | 1.964 |  |
| online_start1_pard2_win2048 | 19 | 1 | 46.540 |  | 489.275 | 1.821 | 1.967 | 1.998 |
| online_start10_pard2_win2048 | 19 | 2 | 47.785 |  | 495.575 | 1.797 | 1.950 | 2.062 |
| online_start5_int5_pard2_win2048 | 19 | 4 | 48.080 |  | 495.627 | 1.797 | 1.948 | 1.979 |
| suffix_k32_longosl | 19 | 0 | 35.056 |  | 511.700 | 1.741 | 1.866 |  |

Positive acceptance delta means the variant accepted more draft tokens than the baseline. Step speedup above 1.0 means the variant had lower mean step time.
