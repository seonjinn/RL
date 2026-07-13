# Qwen8 Official PARD-2 Online Impact - 2026-06-13

Trackers:

- `/Users/sna/Nemo-RL_Qwen3_Roadmap/latest_oci_hsg_qwen8_pard2_official_comparison_20260613_jobs.csv`
- `/Users/sna/Nemo-RL_Qwen3_Roadmap/latest_lyris_qwen8_pard2_official_comparison_20260613_jobs.csv`

Step filter: `step>=2`.

| Variant | Job | Metrics | Gen worker TPS | Speedup vs baseline | Acceptance | Draft refits |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `baseline` | 3288181 | `parsed` | 225.469 | 1.0000 |  | 0 |
| `static_pard2` | 3288182 | `parsed` | 136.891 | 0.6071 | 1.836 | 0 |
| `online_pard2` | 3288183 | `parsed` | 132.736 | 0.5887 | 2.553 | 9 |

## Online vs Static

| Run | Model | Status | Gen worker online/static | E2E online/static | Gen-time speedup | Acceptance delta | Refit steps |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `20260613_oci_hsg_qwen8_pard2_official_comparison_masterconfigfix_shortqos` | `qwen8` | `parsed` | 0.9696 | 0.8087 | 0.9707 | 0.717 | 9 |

## Current Read

- `qwen8` online PARD-2 refit ran for `9` post-step rows. It changed acceptance from `1.836` to `2.553` (`0.717` points), while generation-worker TPS was `0.9696`x of static and E2E TPS was `0.8087`x of static.
- Versus the matched baseline, static PARD-2 was `0.6071`x generation-worker TPS and online PARD-2 was `0.5887`x; this 10-step Qwen8 proof validates online refit mechanics but does not show a throughput win.

Interpretation gate:

- `online_pard2` is useful only if it preserves static PARD-2 speedup and improves acceptance or accepted length after refit.
- Empty metric cells mean logs have not been fetched or no completed step metrics are present yet.
