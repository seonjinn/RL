# NeMo-RL Integrated Unincorporated Results Audit - 2026-06-17

Scope narrowed per request: representative local artifacts only, centered on `mathrl`, `swerl`, `fullgrpo`, and `nemorl` files with `metrics`, `summary`, or `status` in `docs/`, especially `20260615`-`20260617`. No jobs were submitted or cancelled.

## Method

- Compared against `docs/nemorl_integrated_specdec_results_clean_20260617.csv` and `.md` using source-file references and parsed job IDs from the already-inspected representative set.
- This is not an exhaustive historical row-level scan. Older retry/log/manifest files are sampled only where they looked likely to explain missing or superseded integrated evidence.
- Residual risk: older pre-20260615 artifacts may contain historical rows not listed here; the likely current gaps are covered below.

## Summary

- Representative artifacts reported: `22`.
- Recommended `add to clean integrated report`: `11`.
- Recommended `archive/superseded`: `8`.
- Recommended `ignore` because represented/control: `3`.

## Add To Clean Integrated Report
| Source file | Rows | Domain | Model | Missing/parsed job IDs | Represented | Rationale |
|---|---:|---|---|---|---|---|
| `docs/nemorl_clean_results_20260617.csv` | 29 csv_rows | Math-RL | Qwen3-235B-A22B;Qwen3-32B;Qwen3-30B-A3B;Qwen3-1.7B/17B;Qwen3-8B;Qwen3-17B | `3231517;3231518;3288181;3288182;3288183;3321180;3321423;3321424;3337769;3345352` | partial | intermediate clean aggregate has rows absent by job_id from integrated clean report |
| `docs/oci_hsg_mathrl_multimodel_specdec_step20_live_summary_20260616.csv` | 16 csv_rows | Math-RL | Qwen3-235B-A22B;Qwen3-32B;Qwen3-30B-A3B;Qwen3-8B | `3337769` | partial | result metric/summary row(s) are absent by job_id from integrated clean report |
| `docs/oci_hsg_mathrl_multimodel_specdec_step20_20260616_status_live.csv` | 28 csv_rows | Math-RL | Qwen3-235B-A22B;Qwen3-32B;Qwen3-30B-A3B | `3232413;3232414;3232415;3232416;3232417;3232418;3332282;3332283;3333529;3333534;3333538;3337599` | partial | partially represented; add or explicitly exclude current missing statuses, especially pending 3337599; older ablations are superseded |
| `docs/oci_hsg_mathrl_active_refresh_summary_20260616.csv` | 4 csv_rows | Math-RL | Qwen3-32B;Qwen3-30B-A3B;Qwen3-14B | `3337769` | partial | result metric/summary row(s) are absent by job_id from integrated clean report |
| `docs/oci_hsg_mathrl_qwen32_online_pard2_hardce_r11_summary_20260616.csv` | 1 csv_rows | Math-RL | Qwen3-32B | `3345352` | no | result metric/summary row(s) are absent by job_id from integrated clean report |
| `docs/oci_hsg_mathrl_qwen235b_reduced64_temp1_pard_k3k5_summary_step2_10_20260615.csv` | 3 csv_rows | Math-RL | Qwen3-235B-A22B | `3321180;3321423;3321424` | no | result metric/summary row(s) are absent by job_id from integrated clean report |
| `docs/oci_hsg_mathrl_qwen235b_main_baseline_fixed256_3342356_summary_20260616.csv` | 1 csv_rows | Math-RL | Qwen3-235B-A22B | `3342356` | no | result metric/summary row(s) are absent by job_id from integrated clean report |
| `docs/nemorl_performance_config_resubmit_status_20260617.csv` | 3 csv_rows | Math-RL | Qwen3-235B-A22B;Qwen3-8B | `3365679;3365680;3365681` | partial | 2026-06-17 pending follow-up status row(s) are absent from integrated clean report |
| `docs/swerl_suffix_resubmit_status_20260617.csv` | 1 csv_rows | SWE-RL | Qwen3-30B-A3B;/lustre/fsw/portfolios/llmservice/users/igitman/hf_models/Qwen3-30B-A3B-Thinking-25... | `3365678` | no | 2026-06-17 pending follow-up status row(s) are absent from integrated clean report |
| `docs/qwen235b_online_pard2_gate1_systempy_r4_status_20260616.md` | 9 md_table_rows | Math-RL | Qwen3-235B-A22B;Qwen3-32B;Qwen3-30B-A3B;Qwen3-14B | `3332283;3337599` | no | primary missing status row is pending job 3337599; job 3332283 is prior r3 context and should be archived |
| `docs/nemorl_long_osl_16k_32k_status_20260616.md` | 37 md_table_rows | Math-RL | Qwen3-235B-A22B;Qwen3-32B;Qwen3-30B-A3B;Qwen3-1.7B/17B | `12550182;12550221;12561705;12561707;12561710;12561712;12612995;2123407;2123638;2123875;2124030;21...` | no | authoritative long-OSL NeMo-RL status summary is not represented in clean report |

## Archive Or Superseded
| Source file | Rows | Domain | Model | Missing/parsed job IDs | Represented | Rationale |
|---|---:|---|---|---|---|---|
| `docs/lyris_nemorl_integrated_specdec_maxsteps10_metrics_20260613.csv` | 18 csv_rows | NeMo-RL | Qwen3-235B-A22B;Qwen3-32B;Qwen3-30B-A3B;qwen235b;Qwen/Qwen3-235B-A22B;nvidia/Qwen3-235B-A22B-Eagle3;amd/PARD-Qwen3-0.6B;qwen30ba3b | `2109933;2109934;2109935;2109936;2109937;2109938;2109939;2109940;2109942;2109990;2109991;2109992;2...` | no | representative integrated launch/status artifact, but metrics rows are missing_log and not suitable for clean result inclusion |
| `docs/lyris_nemorl_integrated_specdec_maxsteps10_status_20260613.csv` | 18 csv_rows | NeMo-RL | Qwen3-235B-A22B;Qwen3-32B;Qwen3-30B-A3B;qwen235b;Qwen/Qwen3-235B-A22B;nvidia/Qwen3-235B-A22B-Eagle3;amd/PARD-Qwen3-0.6B;qwen30ba3b | `2109933;2109934;2109935;2109936;2109937;2109938;2109939;2109940;2109942;2109990;2109991;2109992;2...` | no | representative integrated launch/status artifact, but metrics rows are missing_log and not suitable for clean result inclusion |
| `docs/nemorl_235b_active_gates_history_20260616.csv` | 7 csv_rows | Math-RL | Qwen3-235B-A22B | `2129203;2129271;2129272;3308774;3315380;3315381;3315382` | no | gate/failed-step evidence without completed clean timing; keep as diagnostics, not clean result rows |
| `docs/nemorl_235b_failed_step1_metrics_20260615.csv` | 3 csv_rows | NeMo-RL | Qwen3-235B-A22B | `3315380;3315381;3315382` | no | gate/failed-step evidence without completed clean timing; keep as diagnostics, not clean result rows |
| `docs/qwen235b_online_pard2_localmodel_r3_status_20260616.md` | 3 md_table_rows | Math-RL | Qwen3-235B-A22B | `3332282;3332283` | no | superseded by later r4/r11 or current clean-report rows; do not add except as diagnostic context |
| `docs/oci_hsg_mathrl_qwen32_online_pard2_step5_r10_status_live_20260616.csv` | 1 csv_rows | Math-RL | Qwen3-32B | `3344974` | no | superseded by later r4/r11 or current clean-report rows; do not add except as diagnostic context |
| `docs/oci_hsg_mathrl_qwen32_online_pard2_hardce_r9_summary_20260616.csv` | 1 csv_rows | Math-RL | Qwen3-32B | `` | no | superseded by later r4/r11 or current clean-report rows; do not add except as diagnostic context |
| `docs/oci_hsg_mathrl_qwen32_online_pard2_force_r4_summary_20260616.csv` | 1 csv_rows | Math-RL | Qwen3-32B | `3340709` | no | superseded by later r4/r11 or current clean-report rows; do not add except as diagnostic context |

## Represented Controls
| Source file | Rows | Domain | Model | Missing/parsed job IDs | Represented | Rationale |
|---|---:|---|---|---|---|---|
| `docs/oci_hsg_swerl_fullgrpo_n3post_wandb_r1_completed_summary_stepge2_20260615.csv` | 3 csv_rows | SWE-RL | Qwen3-235B-A22B | `3299487;3299489;3299491` | yes | represented in clean integrated report; retained here as a control check |
| `docs/oci_hsg_swerl_qwen30ba3b_specdec_manifest_submit_status_20260616.csv` | 6 csv_rows | SWE-RL | Qwen3-30B-A3B | `3351394;3365630;3365631;3365632;3365633;3365634` | yes | represented in clean integrated report; retained here as a control check |
| `docs/oci_hsg_swerl_qwen30ba3b_baseline_ctx40k_3344823_summary_20260616.csv` | 1 csv_rows | SWE-RL | Qwen3-30B-A3B | `3344823` | yes | represented in clean integrated report; retained here as a control check |

## Concise Findings

- Strongest clean-report omissions: `nemorl_clean_results_20260617.csv` has 10 rows absent by job ID, especially Qwen235B reduced64 jobs `3321180/3321423/3321424`, Qwen32 online PARD-2 `3345352`, and Qwen32 no-step online PARD-2 `3337769`.
- Current pending status omissions: SWE-RL Qwen30 suffix `3365678` and Math-RL Qwen235B reruns `3365679/3365680/3365681`.
- The clean report partially uses `oci_hsg_mathrl_multimodel_specdec_step20_live_summary_20260616.csv` but does not carry forward `3337769`.
- `nemorl_long_osl_16k_32k_status_20260616.md` is a meaningful unincorporated summary if the clean report is intended to include long-OSL NeMo-RL evidence.
- Lyris integrated maxsteps10 files are not add candidates because parsed rows are `missing_log`; active-gate/failed-step files should remain diagnostic archive material.

Detailed representative CSV: `docs/nemorl_integrated_unincorporated_results_audit_20260617.csv`.
