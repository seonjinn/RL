# OCI-HSG SWE-RL Full-GRPO Completed 235B Summary - 2026-06-15

Source logs:

- `tmp/oci_hsg_swerl_fullgrpo_logs_live_extract/20260614_oci_hsg_swerl_qwen235b_fullgrpo_specdec_after_prewarm_n3post_wandb_r1`

Parsed with:

- `scripts/extract_nemorl_fullgrpo_step_metrics.py`

Step filter: `step>=2` to reduce cold-start bias.

## Completed 10-Step Cells

| Job | Method | Parsed completed steps | Mean step time | E2E tok/s/GPU | Gen-worker tok/s/GPU | E2E vs baseline | Gen-worker vs baseline |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `3299487` | baseline | 8 | `1243.29s` | `103.75` | `213.06` | `1.0000x` | `1.0000x` |
| `3299489` | PARD K5 | 5 | `1952.39s` | `27.38` | `57.07` | `0.2639x` | `0.2679x` |
| `3299491` | Eagle-3 K3 | 7 | `1389.83s` | `50.08` | `103.47` | `0.4827x` | `0.4857x` |

## Failed Cells In The Same Matrix

| Job | Method | Primary failure |
| --- | --- | --- |
| `3299488` | suffix K32 | vLLM worker import failed because `arctic_inference.suffix_decoding._C` was missing in the actor env. |
| `3299490` | PARD-2 K1 | vLLM import failed because the staged PARD-2 `_C.abi3.so` had a Torch/C10 ABI symbol mismatch. |

## Current Read

This is the best completed 235B SWE-RL 10-step evidence so far, but it does not show a speculative decoding performance win. PARD K5 and Eagle-3 K3 both completed, while both were slower than the baseline on step>=2 throughput. This does not settle PARD-2 because the PARD-2 cell failed before usable metrics; the active PARD-2 proof is still `3308774` on OCI-HSG and `2126915` on Lyris after the `2126895` baseline gate.

Artifacts:

- `docs/oci_hsg_swerl_fullgrpo_n3post_wandb_r1_completed_step_metrics_20260615.csv`
- `docs/oci_hsg_swerl_fullgrpo_n3post_wandb_r1_completed_summary_stepge2_20260615.csv`
