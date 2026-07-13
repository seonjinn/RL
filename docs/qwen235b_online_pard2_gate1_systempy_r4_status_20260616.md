# Qwen235B MathRL Online PARD-2 Gate r4 - 2026-06-16

Snapshot time: `2026-06-16 01:02 PDT`.

## Launch

| Job | Method | State | Account | Shape | Max OSL | Notes |
|---:|---|---|---|---|---:|---|
| `3337599` | online PARD-2 K3 | `PENDING (Priority)` | `nemotron_n3_post` | 32 nodes x 4 GPUs, GBS256 | 1024 | Minimal `max_steps=1` startup/step gate using `/opt/nemo_rl_venv/bin/python`. |

Tracker:

- `latest_oci_hsg_qwen235b_mathrl_online_pard2_gate1_systempy_r4_20260616_jobs.csv`

Remote log root:

- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/mathrl_latest_main_logs/20260616_qwen235b_mathrl_online_pard2_gate1_systempy_r4/online_pard2_k3`

## Contract Checked Before Submission

The launcher contract validation passed before submission:

- `policy.draft.enabled=true`
- `PARD_ONLINE_TRAINING=true`
- `policy.draft.type=pard2`
- `policy.draft.loss=pard2`
- `policy.draft.training_mode=k_slot`
- `policy.draft.train_interval=1`
- `policy.draft.refit_interval=1`
- `policy.sequence_packing.enabled=false`
- `policy.megatron_cfg.context_parallel_size=1`
- `PYTHON_RUNNER_OVERRIDE=/opt/nemo_rl_venv/bin/python`

This is intentionally a small gate because the previous local-model r3 online PARD-2 job `3332283` hung before Step 1 in Ray driver registration and had to be cancelled after `02:58:58`.

## Related MathRL Refresh

The MathRL live summary was refreshed after fetching the latest OCI-HSG logs:

| Run | Completed | Gen tok/s/GPU | E2E step time | E2E tok/s/GPU | Acceptance |
|---|---:|---:|---:|---:|---:|
| `qwen30ba3b_baseline` | 20/20 | `172.57` | `290.25s` | `125.74` | n/a |
| `qwen30ba3b_pard` | 20/20 | `363.71` | `178.18s` | `205.15` | `50.83%` |
| `qwen30ba3b_eagle3` | 20/20 | `406.45` | `161.08s` | `227.42` | `64.75%` |
| `qwen30ba3b_suffix_py313_retry` | 20/20 | `335.95` | `179.23s` | `204.00` | `35.62%` |
| `qwen32_baseline` | 11/12 live | `75.93` | `529.02s` | `68.96` | n/a |
| `qwen32_pard2_14b_retry` | 7/8 live | `52.39` | `741.33s` | `48.79` | `1.73%` |

Baseline-relative readout at this snapshot:

- Qwen30 PARD/Eagle/Suffix are positive versus the completed Qwen30 baseline: E2E throughput speedups are about `1.63x`, `1.81x`, and `1.62x`.
- Qwen32 PARD-2 14B retry is negative versus the live Qwen32 baseline: E2E throughput speedup is about `0.71x`, with very low acceptance.

Updated artifacts:

- `docs/oci_hsg_mathrl_multimodel_specdec_step20_live_steps_20260616.csv`
- `docs/oci_hsg_mathrl_multimodel_specdec_step20_live_summary_20260616.csv`
- `docs/oci_hsg_mathrl_multimodel_specdec_step20_20260616_status_live.csv`
- `docs/specdec_benchmark_metrics_dashboard_20260616.html`
