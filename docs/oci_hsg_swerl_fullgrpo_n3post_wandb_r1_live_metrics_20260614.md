# OCI-HSG SWE-RL Full-GRPO N3Post W&B Retry Live Metrics - 2026-06-14

Checked at `2026-06-14T04:20-07:00` (`2026-06-14T11:20Z`).

Source logs were fetched into `tmp/oci_hsg_swerl_fullgrpo_logs_live_extract/` and parsed with `scripts/extract_nemorl_fullgrpo_step_metrics.py`. Summary excludes step 1 to reduce cold-start bias.

| job_id | method | state at check | parsed steps | mean step time | E2E tok/s/GPU | gen-worker tok/s/GPU | versus baseline E2E |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `3299487` | baseline | `COMPLETED` | 2-9 (`8` steps) | `1243.29s` | `103.75` | `213.06` | `1.000x` |
| `3299489` | PARD K5 | `COMPLETED` | 2-6 (`5` steps) | `1952.39s` | `27.38` | `57.07` | `0.264x` |
| `3299491` | Eagle-3 K3 | `COMPLETED` | 2-8 (`7` steps) | `1389.83s` | `50.08` | `103.47` | `0.483x` |

Notes:

- Baseline, PARD, and Eagle-3 all logged `Async GRPO training complete!` and have top-level SLURM exit code `0`.
- The shutdown-time `EngineDeadError`, `ClientConnectorError`, and event-loop traces appear after completion and do not change the top-level `COMPLETED` state.
- Suffix `3299488` failed before usable metrics because `arctic_inference.suffix_decoding._C` was missing in the Python 3.13 actor env.
- PARD-2 `3299490` failed before usable metrics because the staged PARD-2 vLLM `_C.abi3.so` had a Torch/C10 ABI symbol mismatch.

Artifacts:

- Step CSV: `docs/oci_hsg_swerl_fullgrpo_n3post_wandb_r1_live_steps_20260614.csv`
- Summary CSV: `docs/oci_hsg_swerl_fullgrpo_n3post_wandb_r1_live_summary_20260614.csv`
