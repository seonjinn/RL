# vLLM Standalone Temp0 vs Temp1 Trends - 2026-06-16

This page separates measured speedup from acceptance-only rows. Temp 1 uses the RL rollout sampling setting (`temperature=1.0`, `top_p=1.0`); temp 0/default rows are deterministic acceptance-ceiling style benchmarks.

## Current Math RL Baseline Status
| Model | Current spec rows | Mean generation time so far | Baseline status | Speedup status |
| --- | --- | ---: | --- | --- |
| qwen30ba3b | pard: 8 steps, 98.9s; eagle3: 9 steps, 89.6s; suffix_py313_retry: 5 steps, 107.1s |  | 3334218 running | waiting for baseline generation-time rows |
| qwen32 | pard: 5 steps, 237.5s; eagle3: 5 steps, 250.4s; suffix_py313_retry: 3 steps, 269.3s |  | 3334219 running | waiting for baseline generation-time rows |
| qwen235b | no completed spec rows yet |  | 3334220 pending | waiting for baseline generation-time rows |

Matched Math RL no-spec baseline jobs were submitted with `MAX_STEPS=20`, `MAX_NEW_TOKENS=1024`, `MIN_TOKENS=1024`, `temperature=1.0`, `top_p=1.0`, `top_k=-1`, and `ENABLE_VLLM_SPECDEC=false`: `3334218` Qwen30, `3334219` Qwen32, `3334220` Qwen235B.

Earlier Qwen235B Math RL reduced64/output-256 data is only a sanity signal, not the final 1024-token comparison: baseline job `3321180` averaged `35.8s` generation time, PARD K3 job `3321423` averaged `23.3s` (`1.54x` generation-time speedup), and PARD K5 job `3321424` averaged `21.9s` (`1.64x` generation-time speedup).

## Standalone Trend Summary

| Domain | Dataset | Model | Temp | Method | Rows | Mean speedup | Range | Mean tok/s/GPU | Mean acceptance | Basis |
| --- | --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | --- |
| SWE | full+verified OSL32K | Qwen3-235B-A22B | 0.0 | `eagle3_k3` | 10 | 2.022x | 1.817-2.501 |  | 54.2% | temp0 20260612 final spec rows / live baseline telemetry |
| SWE | full+verified OSL32K | Qwen3-235B-A22B | 0.0 | `suffix_k32` | 10 | 4.367x | 3.418-6.180 |  | 81.9% | temp0 20260612 final spec rows / live baseline telemetry |
| SWE | full+verified OSL32K | Qwen3-235B-A22B | 1.0 | `eagle3_k11` | 10 | 1.230x | 0.993-1.449 | 28.686 | 7.5% | temp1/top_p1 20260614 final rows / final matched baseline breakdowns |
| SWE | full+verified OSL32K | Qwen3-235B-A22B | 1.0 | `eagle3_k9` | 10 | 1.209x | 1.019-1.453 | 28.656 | 9.1% | temp1/top_p1 20260614 final rows / final matched baseline breakdowns |
| SWE | full+verified OSL32K | Qwen3-235B-A22B | 1.0 | `pard2_k11` | 10 | 0.814x | 0.730-0.859 | 19.723 | 1.3% | temp1/top_p1 20260614 final rows / final matched baseline breakdowns |
| SWE | full+verified OSL32K | Qwen3-235B-A22B | 1.0 | `pard2_k9` | 10 | 0.826x | 0.730-0.877 | 19.713 | 1.8% | temp1/top_p1 20260614 final rows / final matched baseline breakdowns |
| SWE | full+verified OSL32K | Qwen3-235B-A22B | 1.0 | `pard_k11` | 10 | 1.002x | 0.762-1.424 | 22.139 | 6.4% | temp1/top_p1 20260614 final rows / final matched baseline breakdowns |
| SWE | full+verified OSL32K | Qwen3-235B-A22B | 1.0 | `pard_k9` | 10 | 0.979x | 0.767-1.392 | 21.819 | 7.2% | temp1/top_p1 20260614 final rows / final matched baseline breakdowns |
| SWE | full+verified OSL32K | Qwen3-235B-A22B | 1.0 | `suffix_k16` | 10 | 1.739x | 1.282-3.402 | 36.667 | 50.9% | temp1/top_p1 20260614 final rows / final matched baseline breakdowns |
| SWE | full+verified OSL32K | Qwen3-235B-A22B | 1.0 | `suffix_k8` | 10 | 1.782x | 1.267-3.790 | 36.643 | 51.7% | temp1/top_p1 20260614 final rows / final matched baseline breakdowns |
| Math | Math500 OSL32K | Qwen3-30B-A3B | 0.0/default | `draft_model_k3` | 2 | 1.783x | 1.716-1.849 | 53.033 | 71.3% | temp0/default 20260612 final rows / matched final baseline |
| Math | Math500 OSL32K | Qwen3-30B-A3B | 0.0/default | `draft_model_k5` | 2 | 1.820x | 1.535-2.105 | 51.951 | 47.8% | temp0/default 20260612 final rows / matched final baseline |
| Math | Math500 OSL32K | Qwen3-30B-A3B | 0.0/default | `suffix_k32` | 2 | 7.434x | 7.296-7.572 | 225.274 | 90.5% | temp0/default 20260612 final rows / matched final baseline |
| Math | Math500 OSL32K | Qwen3-8B | 0.0/default | `eagle3_k3` | 2 | 1.814x | 1.606-2.021 | 96.451 | 63.3% | temp0/default 20260612 final rows / matched final baseline |
| Math | Math500 OSL32K | Qwen3-8B | 0.0/default | `pard2_k3` | 1 | 0.435x | 0.435-0.435 | 15.994 | 0.1% | temp0/default 20260612 final rows / matched final baseline |
| Math | Math500 OSL32K | Qwen3-8B | 0.0/default | `pard2_k5` | 1 | 0.438x | 0.438-0.438 | 16.103 | 0.0% | temp0/default 20260612 final rows / matched final baseline |
| Math | Math500 OSL32K | Qwen3-8B | 0.0/default | `suffix_k32` | 2 | 6.119x | 5.871-6.367 | 342.923 | 87.9% | temp0/default 20260612 final rows / matched final baseline |
| Math | Math500 OSL32K | Qwen3-235B-A22B | 1.0 | `eagle3_k3` | 2 |  |  | 6.238 | 42.6% | temp1/top_p1 acceptance/tok-s only; baseline and official PARD2 timed out, so speedup blank |
| Math | Math500 OSL32K | Qwen3-235B-A22B | 1.0 | `pard_k5` | 2 |  |  | 5.870 | 31.8% | temp1/top_p1 acceptance/tok-s only; baseline and official PARD2 timed out, so speedup blank |
| Math | Math500 OSL32K | Qwen3-235B-A22B | 1.0 | `suffix_k32` | 2 |  |  | 15.367 | 66.3% | temp1/top_p1 acceptance/tok-s only; baseline and official PARD2 timed out, so speedup blank |

## Readout

- SWE temp 0/default is much more favorable to speculative decoding: EAGLE3 K3 averages about 2.0x and suffix K32 about 4.4x, with high acceptance.
- SWE temp 1/top-p 1 is closer to RL: suffix remains positive but drops to about 1.7-1.8x; EAGLE3 stays modestly positive around 1.2x; current PARD/PARD2 learned drafters are around baseline or slower because acceptance collapses.
- Math temp 1 Qwen235B needs a completed baseline row before exact speedup can be stated. In that sweep, suffix/PARD/EAGLE3 completed, but baseline and official PARD2 timed out.
- The Math RL speedup table will become meaningful once `3334218/3334219/3334220` produce baseline generation-time rows.

## Sources
- `docs/lyris_math500_osl32k_metrics_20260612.csv`
- `docs/lyris_qwen235b_standalone_temp1rl_20260614_metrics.csv`
- `docs/lyris_qwen235b_swebench_osl32k_batch_sweep_speedups_20260612.csv`
- `docs/oci_hsg_mathrl_qwen235b_reduced64_temp1_pard_k3k5_speedups_step2_10_20260615.csv`
- `docs/oci_hsg_mathrl_multimodel_specdec_step20_live_summary_20260616.csv`
- `latest_oci_hsg_mathrl_multimodel_baseline_step20_20260616_jobs.csv`
