# NeMo-RL Full-GRPO Online Drafter Comparison

| Variant | Steps | Draft refits | Acceptance | Delta pp | Step time | Step speedup | Gen tok/s/GPU speedup | Draft loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| online_start1_pard2_win2048 | 19 | 1 | 46.540 | -2.625 | 489.275 | 1.001 | 1.002 | 1.998 |
| online_start10_pard2_win2048 | 19 | 2 | 47.785 | -1.381 | 495.575 | 0.989 | 0.993 | 2.062 |
| online_start5_int5_pard2_win2048 | 19 | 4 | 48.080 | -1.085 | 495.627 | 0.989 | 0.992 | 1.979 |

Positive acceptance delta means the variant accepted more draft tokens than the baseline. Step speedup above 1.0 means the variant had lower mean step time.
