# Qwen8 PARD-2 Online Drafter Diagnosis - 2026-06-12

## Current Finding

The Qwen8 PARD-2 online path is functional when the online worker performs an initial drafter refit before generation. The earlier 2-step canary that showed `0%` acceptance is now best understood as a bad ordering/configuration point, not evidence that PARD-2 online training is inherently broken.

The required controls for the working path are:

- `policy.draft.initial_refit=true`
- explicit PARD token metadata, observed as `pard_token=151670`
- delayed or interval-based online train/refit, rather than dummy-loaded draft generation
- comparisons should skip step 1 when step 1 includes initialization or first-refit behavior

## Evidence

Refreshed source CSVs:

- `docs/public_pard2_job_status_20260611.csv`
- `docs/public_pard2_nemorl_steps_20260611.csv`
- `docs/public_pard2_nemorl_summary_20260611.csv`
- `docs/public_pard2_vllm_standalone_20260611.csv`

Key 2-step gates:

| Job | Variant | Steps | Refits | Token acceptance | Mean step time | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `3269510` | no initial refit | 2 | 1 | `0.00%` | `29.28s` | bad path: dummy-loaded online draft |
| `3269985` | initial-refit static-equivalent | 2 | 0 | `30.82%` | `23.54s` | initial refit restores acceptance |
| `3270216` | online after initial refit | 2 | 2 | `31.92%` | `25.80s` | online train/refit works functionally |

20-step OSL1024 / GBS8 comparison:

| Job | Variant | Steps | Refits | Token acceptance | Mean step time | Draft loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `3270833` | initial-refit static-equivalent | 20 | 0 | `48.94%` | `33.05s` |  |
| `3270834` | online every step | 20 | 20 | `48.32%` | `35.90s` | `1.9768` |
| `3271494` | online interval-5 | 20 | 4 | `49.65%` | `33.03s` | `1.9715` |
| `3271495` | online interval-10 | 20 | 2 | `47.56%` | `34.12s` | `1.8208` |

50-step OSL1024 / GBS8 final comparison, matched steps `2-50`:

| Job | Variant | Matched steps | Refits | Token acceptance | Mean step time | Mean generation time | Gen worker tok/s/GPU |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `3271789` | initial-refit static-equivalent | 49 | 0 | `47.54%` | `32.68s` | `17.20s` | `140.70` |
| `3271807` | online interval-5 | 49 | 9 | `47.16%` | `32.93s` | `16.91s` | `142.77` |

## Interpretation

The initial-refit fix changes the status from "online PARD-2 collapses" to "online PARD-2 is wired and can preserve acceptance." On Qwen8, interval-5 is the best observed online-update setting:

- It avoids the `0%` acceptance failure seen without initial refit.
- It is better than every-step online updates in the 20-step run.
- In the full 50-step run, it stays close to static-equivalent acceptance and has slightly better generation time/worker throughput.
- It is not yet an end-to-end performance win because update overhead still makes mean step time slightly worse than static-equivalent.

This means the next performance claim should not use the obsolete 2-step `0%` canary as the conclusion. The current conclusion is: PARD-2 online training works functionally with initial refit, but needs interval/update-cost tuning before claiming speedup.

## SWE-Bench Standalone Context

The refreshed standalone vLLM public PARD-2 SWE-Bench rows are still negative for throughput:

- Qwen8 lite, BS8: `0.960x`, acceptance `28.44%`
- Qwen8 lite, BS16: `0.798x`, acceptance `28.73%`
- Qwen8 verified long OSL16K, BS1/2/4/8: `0.576x-0.771x`, acceptance `7.50%-11.79%`
- Qwen14 lite: about `0.788x-0.790x`, acceptance `18.02%-18.47%`

So PARD-2 online training and SWE-Bench standalone inference should be tracked separately: the online-training path is now functional, while public PARD-2 standalone SWE-Bench remains a throughput loss at the measured K=5 shapes.
