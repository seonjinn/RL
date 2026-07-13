# OCI-HSG SWE-RL Baseline vs PARD Summary - N3Post W&B r1

Source run: `20260614_oci_hsg_swerl_qwen235b_fullgrpo_specdec_after_prewarm_n3post_wandb_r1`.

## Job Outcomes

| Job | Method | SLURM state | Parsed completed steps | Last parsed step | Notes |
| --- | --- | --- | ---: | ---: | --- |
| `3299487` | baseline | COMPLETED | 9 | 9 | exited 0 after checkpoint cutoff before Step 10 marker |
| `3299489` | PARD K=5 | COMPLETED | 6 | 6 | exited 0 after checkpoint cutoff before Step 7 marker |
| `3299490` | PARD2 K=1 | FAILED | 0 |  | failed during vLLM worker init on compiled `_C.abi3.so` ABI mismatch |

## Common Window Comparison

| Window | Method | Steps | E2E tok/s/GPU | Gen-worker tok/s/GPU | Mean step time s | Avg reward |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Step 1-6 | `baseline_steps10` | 6 | 149.02 | 304.80 | 944.89 | 0.0872 |
| Step 1-6 | `pard_k5_steps10` | 6 | 59.00 | 121.33 | 1671.36 | 0.0664 |
| Step 1-6 | `PARD / baseline` |  | 0.396x | 0.398x | 1.769x |  |
| Step 2-6 | `baseline_steps10` | 5 | 132.31 | 271.18 | 1080.53 | 0.0688 |
| Step 2-6 | `pard_k5_steps10` | 5 | 27.38 | 57.07 | 1952.39 | 0.0594 |
| Step 2-6 | `PARD / baseline` |  | 0.207x | 0.210x | 1.807x |  |

## Full Parsed Span

| Method | Parsed steps | E2E tok/s/GPU | Gen-worker tok/s/GPU | Mean step time s |
| --- | ---: | ---: | ---: | ---: |
| `baseline_steps10` | 9 | 118.06 | 241.93 | 1134.78 |
| `pard_k5_steps10` | 6 | 59.00 | 121.33 | 1671.36 |

## PARD2 Failure

The failed PARD2 run used the older compiled official target feature site:
`vllm_pard2_official_target_feat/vllm/_C.abi3.so`.

The decisive error was:

```text
ImportError: .../vllm/_C.abi3.so: undefined symbol: _ZN3c1013MessageLoggerC1ENS_14SourceLocationEib
ImportError: vLLM is not installed. Please check that the py_executable in the runtime_env of VllmGenerationWorker covers the vllm dependency.
```

The current r16 PARD2 retry avoids that compiled site by using the pyoverlay/eager no-level path and is still pending.

## Artifacts

- Per-step metrics: `docs/oci_hsg_swerl_fullgrpo_n3post_wandb_r1_baseline_pard_step_metrics_20260614.csv`
- Full parsed-span summary: `docs/oci_hsg_swerl_fullgrpo_n3post_wandb_r1_baseline_pard_summary_20260614.csv`
- Common Step 1-6 summary: `docs/oci_hsg_swerl_fullgrpo_n3post_wandb_r1_baseline_pard_summary_steps1to6_20260614.csv`
- Common Step 2-6 summary: `docs/oci_hsg_swerl_fullgrpo_n3post_wandb_r1_baseline_pard_summary_steps2to6_20260614.csv`
