# Qwen3-235B SWE Standalone Temp1/Top-p1 Performance Summary

Source metrics: `docs/lyris_qwen235b_standalone_temp1rl_20260614_metrics.csv`

Refresh state: SWE standalone is complete: `90/90` jobs completed. Overall standalone ended at `93/95` completed; Math500 baseline `2124147` and official PARD2 `2124150` hit the 5-hour walltime and timed out.

Sampling setting: this is the RL rollout setting, `temperature=1` and `top_p=1`, not the deterministic `temperature=0` acceptance-ceiling run.

## High-Level Readout

- Suffix decoding is the strongest standalone method under RL sampling.
- EAGLE3 is usually faster than baseline, but weaker than suffix decoding.
- Current PARD2 standalone is consistently slower than baseline because acceptance stays around `1-2%`.
- Current PARD is mixed: it helps at small batches on `full`, but is weak or slower on `verified`.
- On Math500, baseline and official PARD2 timed out before producing `breakdown.json`; suffix K32, PARD K5, and EAGLE3 K3 completed.

## Method Summary

| Dataset | Method | Rows | Mean speedup | Speedup range | Mean tok/s/GPU | Mean acceptance | Best batch | Best speedup |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| full | eagle3_k11 | 5 | 1.163 | 0.993-1.324 | 27.13 | 7.13% | 4 | 1.324 |
| full | eagle3_k9 | 5 | 1.239 | 1.019-1.395 | 29.59 | 9.57% | 4 | 1.395 |
| full | pard2_k11 | 5 | 0.811 | 0.730-0.849 | 19.56 | 1.26% | 8 | 0.849 |
| full | pard2_k9 | 5 | 0.833 | 0.730-0.877 | 19.68 | 1.76% | 8 | 0.877 |
| full | pard_k11 | 5 | 1.083 | 0.762-1.424 | 22.59 | 7.53% | 2 | 1.424 |
| full | pard_k9 | 5 | 0.998 | 0.767-1.392 | 21.53 | 8.12% | 2 | 1.392 |
| full | suffix_k16 | 5 | 1.887 | 1.389-3.402 | 37.85 | 56.02% | 2 | 3.402 |
| full | suffix_k8 | 5 | 1.929 | 1.418-3.790 | 38.60 | 52.62% | 2 | 3.790 |
| verified | eagle3_k11 | 5 | 1.297 | 1.132-1.449 | 30.25 | 7.96% | 2 | 1.449 |
| verified | eagle3_k9 | 5 | 1.178 | 1.049-1.453 | 27.72 | 8.65% | 2 | 1.453 |
| verified | pard2_k11 | 5 | 0.816 | 0.760-0.859 | 19.89 | 1.33% | 2 | 0.859 |
| verified | pard2_k9 | 5 | 0.819 | 0.750-0.864 | 19.75 | 1.80% | 2 | 0.864 |
| verified | pard_k11 | 5 | 0.921 | 0.829-1.049 | 21.69 | 5.33% | 2 | 1.049 |
| verified | pard_k9 | 5 | 0.959 | 0.793-1.095 | 22.11 | 6.19% | 2 | 1.095 |
| verified | suffix_k16 | 5 | 1.590 | 1.282-1.922 | 35.49 | 45.79% | 4 | 1.922 |
| verified | suffix_k8 | 5 | 1.636 | 1.267-2.341 | 34.69 | 50.84% | 2 | 2.341 |

## Best Method Per Batch

| Dataset | Batch | Best method | Speedup | tok/s/GPU | Acceptance |
| --- | ---: | --- | ---: | ---: | ---: |
| verified | 2 | suffix_k8 | 2.341 | 9.65 | 61.05% |
| verified | 4 | suffix_k16 | 1.922 | 16.15 | 46.82% |
| verified | 8 | suffix_k16 | 1.631 | 26.15 | 48.78% |
| verified | 16 | suffix_k16 | 1.282 | 41.98 | 49.12% |
| verified | 32 | suffix_k16 | 1.325 | 85.77 | 45.34% |
| full | 2 | suffix_k8 | 3.790 | 15.64 | 72.63% |
| full | 4 | suffix_k16 | 1.682 | 14.04 | 50.72% |
| full | 8 | suffix_k16 | 1.475 | 23.70 | 54.53% |
| full | 16 | suffix_k8 | 1.510 | 47.98 | 42.72% |
| full | 32 | suffix_k8 | 1.453 | 94.29 | 47.24% |

## Implication For Online Drafter Work

The standalone baseline for online drafter training should not use the old `temperature=0` results. Under the RL sampling regime, acceptance drops sharply, especially for learned-drafter methods. Current PARD2 needs online drafter training or another acceptance improvement before it can be expected to outperform baseline in SWE-style rollout.

For near-term comparisons:

- Use suffix decoding as the strongest non-online standalone reference.
- Use EAGLE3 as a learned-drafter reference that remains faster than baseline under sampling.
- Treat current PARD2 standalone as a negative control: useful for validating integration, but not yet a throughput win.
