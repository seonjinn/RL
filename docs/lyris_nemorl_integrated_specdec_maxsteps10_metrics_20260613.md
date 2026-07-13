# Lyris NeMo-RL Integrated SpecDec Metrics - 2026-06-13

Step filter: `step>=2`.
Speedups use the parsed no-spec `baseline` row with the same run id, model, max output tokens, and minimum output tokens.

Metric status counts:

- `missing_log`: 18

| Run | Model | Method | Job | K | Metrics | Speedup basis | Gen worker TPS | Gen worker speedup | E2E speedup | Gen time speedup | Acceptance | Steps | Log |
| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 20260613_nemorl_integrated_specdec_step10 | `qwen235b` | `eagle3` | 2109942 | 3 | `missing_log` | `` |  |  |  |  |  |  | `` |
| 20260613_nemorl_integrated_specdec_step10 | `qwen235b` | `pard` | 2109940 | 5 | `missing_log` | `` |  |  |  |  |  |  | `` |
| 20260613_nemorl_integrated_specdec_step10 | `qwen235b` | `suffix` | 2109939 | 32 | `missing_log` | `` |  |  |  |  |  |  | `` |
| 20260613_nemorl_integrated_specdec_step10 | `qwen30ba3b` | `eagle3` | 2109935 | 3 | `missing_log` | `` |  |  |  |  |  |  | `` |
| 20260613_nemorl_integrated_specdec_step10 | `qwen30ba3b` | `pard` | 2109934 | 5 | `missing_log` | `` |  |  |  |  |  |  | `` |
| 20260613_nemorl_integrated_specdec_step10 | `qwen30ba3b` | `suffix` | 2109933 | 32 | `missing_log` | `` |  |  |  |  |  |  | `` |
| 20260613_nemorl_integrated_specdec_step10 | `qwen32` | `eagle3` | 2109938 | 3 | `missing_log` | `` |  |  |  |  |  |  | `` |
| 20260613_nemorl_integrated_specdec_step10 | `qwen32` | `pard` | 2109937 | 5 | `missing_log` | `` |  |  |  |  |  |  | `` |
| 20260613_nemorl_integrated_specdec_step10 | `qwen32` | `suffix` | 2109936 | 32 | `missing_log` | `` |  |  |  |  |  |  | `` |
| 20260613_nemorl_integrated_specdec_step10_raymatch | `qwen235b` | `eagle3` | 2110003 | 3 | `missing_log` | `` |  |  |  |  |  |  | `` |
| 20260613_nemorl_integrated_specdec_step10_raymatch | `qwen235b` | `pard` | 2110002 | 5 | `missing_log` | `` |  |  |  |  |  |  | `` |
| 20260613_nemorl_integrated_specdec_step10_raymatch | `qwen235b` | `suffix` | 2110001 | 32 | `missing_log` | `` |  |  |  |  |  |  | `` |
| 20260613_nemorl_integrated_specdec_step10_raymatch | `qwen30ba3b` | `eagle3` | 2109992 | 3 | `missing_log` | `` |  |  |  |  |  |  | `` |
| 20260613_nemorl_integrated_specdec_step10_raymatch | `qwen30ba3b` | `pard` | 2109991 | 5 | `missing_log` | `` |  |  |  |  |  |  | `` |
| 20260613_nemorl_integrated_specdec_step10_raymatch | `qwen30ba3b` | `suffix` | 2109990 | 32 | `missing_log` | `` |  |  |  |  |  |  | `` |
| 20260613_nemorl_integrated_specdec_step10_raymatch | `qwen32` | `eagle3` | 2109995 | 3 | `missing_log` | `` |  |  |  |  |  |  | `` |
| 20260613_nemorl_integrated_specdec_step10_raymatch | `qwen32` | `pard` | 2109994 | 5 | `missing_log` | `` |  |  |  |  |  |  | `` |
| 20260613_nemorl_integrated_specdec_step10_raymatch | `qwen32` | `suffix` | 2109993 | 32 | `missing_log` | `` |  |  |  |  |  |  | `` |

Caveats:

- Existing 20260613 integrated trackers did not include baseline jobs; those rows correctly show `missing_matched_baseline` when logs are parsed.
- Rows with `missing_log` need Lyris log collection before performance can be evaluated.
