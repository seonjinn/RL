# NeMo-RL Integrated SpecDec Results - 2026-06-17

Updated from local artifacts plus bounded OCI-HSG `sacct`/`squeue` and metric greps. No jobs were submitted or cancelled. Slurm-reported submitted jobs use account `nemotron_n3_post`.

## Math-RL

| Model | Method | K | Job | State | Steps | Max new tokens | E2E step s | Gen s | E2E tok/s/GPU | Gen tok/s/GPU | E2E speedup | Gen speedup | Accept | Mean accepted | Notes |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3-30B-A3B | baseline | 0 | 3334218 | COMPLETED | 20/20 | 1024 | 290.248 | 211.283 | 125.7435 | 172.567 | 1.000x | 1.000x |  |  | matched baseline |
| Qwen3-30B-A3B | pard | 5 | 3333526 | COMPLETED | 20/20 | 1024 | 178.1765 | 100.4075 | 205.147 | 363.707 | 1.631x | 2.108x | 50.8293% | 3.5414 |  |
| Qwen3-30B-A3B | eagle3 | 3 | 3333528 | COMPLETED | 20/20 | 1024 | 161.0800 | 89.8225 | 227.419 | 406.4535 | 1.809x | 2.355x | 64.7547% | 2.9422 |  |
| Qwen3-30B-A3B | suffix | 32 | 3333715 | COMPLETED | 20/20 | 1024 | 179.2300 | 108.9350 | 204.0035 | 335.9465 | 1.622x | 1.947x | 35.6164% | 2.6431 |  |
| Qwen3-30B-A3B | pard2_8b |  | 3333527 | FAILED | 0/20 | 1024 |  |  |  |  |  |  |  |  | no completed step |
| Qwen3-32B | baseline | 0 | 3334219 | COMPLETED | 20/20 | 1024 | 528.1090 | 480.9410 | 69.0345 | 75.8090 | 1.000x | 1.000x |  |  | final metrics from remote grep |
| Qwen3-32B | pard | 5 | 3333531 | COMPLETED | 20/20 | 1024 | 287.8445 | 237.5955 | 126.7935 | 153.6420 | 1.837x | 2.027x | 46.7958% | 3.3395 |  |
| Qwen3-32B | eagle3 | 3 | 3333533 | COMPLETED | 20/20 | 1024 | 297.5705 | 248.1960 | 122.5530 | 146.9590 | 1.775x | 1.939x | 46.6907% | 2.4006 |  |
| Qwen3-32B | suffix | 32 | 3333716 | COMPLETED | 20/20 | 1024 | 322.2520 | 271.5890 | 113.2020 | 134.3570 | 1.640x | 1.772x | 29.8925% | 2.2252 | acceptance from local partial parse |
| Qwen3-32B | pard2_14b | 3 | 3334113 | TIMEOUT | 10/20 | 1024 | 741.4930 | 690.7190 | 49.3570 | 52.9880 | 0.715x | 0.699x | 1.6951% | 1.0507 | timed out |
| Qwen3-32B | pard2_8b |  | 3333532 | FAILED | 0/20 | 1024 |  |  |  |  |  |  |  |  | no completed step |
| Qwen3-235B-A22B | baseline | 0 | 3334220 | FAILED | 8/20 | 1024 | 196.2788 | 127.5325 | 12.4488 | 17.6675 | 1.000x | 1.000x |  |  | partial before NCCL watchdog |
| Qwen3-235B-A22B | pard |  | 3333535 | FAILED | 0/20 | 1024 |  |  |  |  |  |  |  |  | no clean parsed metrics |
| Qwen3-235B-A22B | pard2 |  | 3333536 | FAILED | 2/20 | 1024 | 336.7850 | 155.5650 | 7.6450 | 14.4700 | 0.614x | 0.819x | 5.4240% | 1.1631 | partial before failure |
| Qwen3-235B-A22B | eagle3 | 3 | 3333537 | FAILED | 14/20 | 1024 | 119.5586 | 66.2486 | 20.8286 | 34.1621 | 1.673x | 1.934x | 47.4243% | 2.4225 | partial before NCCL watchdog |
| Qwen3-235B-A22B | suffix | 32 | 3333717 | FAILED | 14/20 | 1024 | 145.8186 | 93.5486 | 16.6893 | 24.1929 | 1.341x | 1.369x | 26.3522% | 1.7436 | partial before NCCL watchdog |

## SWE-RL

| Model | Method | K | Job | State | Steps | Seq len | E2E step s | Gen s | E2E tok/s/GPU | Gen tok/s/GPU | E2E speedup | Gen speedup | Accept | Mean accepted | Notes |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3-235B-A22B | baseline | 0 | 3299487 | COMPLETED | 8/10 |  | 1243.2913 |  | 103.7525 | 213.0563 | 1.000x | 1.000x |  |  | step>=2 parsed summary |
| Qwen3-235B-A22B | suffix | 32 | 3299488 | FAILED | 0/10 |  |  |  |  |  |  |  |  |  | Arctic extension import failure |
| Qwen3-235B-A22B | pard | 5 | 3299489 | COMPLETED | 5/10 |  | 1952.3940 |  | 27.3820 | 57.0700 | 0.264x | 0.268x |  |  | completed but slower than baseline |
| Qwen3-235B-A22B | pard2 | 1 | 3299490 | FAILED | 0/10 |  |  |  |  |  |  |  |  |  | PARD-2 vLLM ABI mismatch |
| Qwen3-235B-A22B | eagle3 | 3 | 3299491 | COMPLETED | 7/10 |  | 1389.8343 |  | 50.0786 | 103.4714 | 0.483x | 0.486x |  |  | completed but slower than baseline |
| Qwen3-30B-A3B | baseline | 0 | 3344823 | COMPLETED | 1/1 | 40960 | 141.0400 |  | 190.8000 | 559.5000 | 1.000x | 1.000x |  |  | ctx40k baseline |
| Qwen3-30B-A3B | suffix | 32 | 3351394 | COMPLETED | 1/1 | 40960 | 142.2200 |  | 183.3100 | 533.7300 | 0.961x | 0.954x |  |  | generation time and acceptance not parsed |
| Qwen3-30B-A3B | eagle3 | 3 | 3365630 | PENDING | 0/1 | 40960 |  |  |  |  |  |  |  |  | squeue reason Priority |
| Qwen3-30B-A3B | pard | 5 | 3365631 | PENDING | 0/1 | 40960 |  |  |  |  |  |  |  |  | squeue reason Priority |
| Qwen3-30B-A3B | pard2 | 3 | 3365632 | PENDING | 0/1 | 40960 |  |  |  |  |  |  |  |  | squeue reason Priority |
| Qwen3-30B-A3B | online_pard | 5 | 3365633 | PENDING | 0/1 | 40960 |  |  |  |  |  |  |  |  | squeue reason Priority |
| Qwen3-30B-A3B | online_pard2 | 3 | 3365634 | PENDING | 0/1 | 40960 |  |  |  |  |  |  |  |  | squeue reason Priority |

## Key Missing Or Failed Rows

- Qwen30 SWE-RL non-suffix jobs `3365630`-`3365634` are still `PENDING` under `nemotron_n3_post`; no metrics exist yet.
- Qwen30 SWE-RL suffix `3351394` completed, but the bounded metric grep did not expose generation time or SpecDec acceptance metrics.
- Qwen235B Math-RL baseline/eagle3/suffix failed after usable partial steps; PARD `3333535` has no clean parsed metrics, and PARD-2 `3333536` only has 2 completed parsed steps.
- Qwen32 Math-RL PARD-2 14B `3334113` timed out after 10 completed parsed steps and is below baseline throughput; PARD-2 8B target-dimension rows `3333527` and `3333532` have no completed step.
- Qwen235B SWE-RL PARD and Eagle-3 completed but are slower than baseline; suffix and PARD-2 failed before usable metrics.

## Sources

- `docs/oci_hsg_mathrl_multimodel_specdec_step20_live_summary_20260616.csv`
- `docs/oci_hsg_mathrl_multimodel_specdec_step20_20260616_status_live.csv`
- `docs/oci_hsg_mathrl_qwen235b_baseline_step20_3334220_partial_summary_20260616.csv`
- `docs/oci_hsg_mathrl_qwen235b_eagle3_step20_3333537_partial_summary_20260616.csv`
- `docs/oci_hsg_mathrl_qwen235b_suffix_step20_3333717_partial_summary_20260616.csv`
- `docs/oci_hsg_swerl_fullgrpo_n3post_wandb_r1_completed_summary_stepge2_20260615.csv`
- `docs/oci_hsg_swerl_fullgrpo_specdec_after_prewarm_n3post_wandb_r1_status_20260614.md`
- `docs/oci_hsg_swerl_qwen30ba3b_baseline_ctx40k_3344823_summary_20260616.csv`
- `docs/oci_hsg_swerl_qwen30ba3b_specdec_manifest_submit_status_20260616.csv`
- Bounded OCI-HSG reads on `2026-06-17`: `squeue`, `sacct`, and metric-only `grep` snippets.
