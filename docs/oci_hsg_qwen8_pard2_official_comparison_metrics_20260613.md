# NeMo-RL Integrated SpecDec Metrics - 2026-06-13

Step filter: `step>=2`.
Speedups use the parsed no-spec `baseline` row with the same run id, model, max output tokens, and minimum output tokens.

Metric status counts:

- `parsed`: 3

Speedup basis counts:

- `matched_baseline`: 3

| Run | Model | Method | Job | K | Metrics | Speedup basis | Gen worker TPS | Gen worker speedup | E2E speedup | Gen time speedup | Acceptance | Steps | Log |
| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 20260613_oci_hsg_qwen8_pard2_official_comparison_masterconfigfix_shortqos | `qwen8` | `baseline` | 3288181 | 0 | `parsed` | `matched_baseline` | 225.469 | 1.0000 | 1.0000 | 1.0000 |  | 9/9 | `ray-driver.log` |
| 20260613_oci_hsg_qwen8_pard2_official_comparison_masterconfigfix_shortqos | `qwen8` | `online_pard2` | 3288183 | 1 | `parsed` | `matched_baseline` | 132.736 | 0.5887 | 0.6705 | 0.5880 | 2.553 | 9/9 | `ray-driver.log` |
| 20260613_oci_hsg_qwen8_pard2_official_comparison_masterconfigfix_shortqos | `qwen8` | `static_pard2` | 3288182 | 1 | `parsed` | `matched_baseline` | 136.891 | 0.6071 | 0.8291 | 0.6058 | 1.836 | 9/9 | `ray-driver.log` |

Caveats:

- Some existing 20260613 integrated trackers did not include baseline jobs; those rows correctly show `missing_matched_baseline` when logs are parsed.
- Rows with `missing_log` need remote log collection before performance can be evaluated.
