# OCI-HSG Qwen8 Official PARD-2 Comparison Status

Tracker: `/Users/sna/Nemo-RL_Qwen3_Roadmap/latest_oci_hsg_qwen8_pard2_official_comparison_20260613_jobs.csv`
Host: `oci-hsg-cs-001-vscode-02`
Summary: COMPLETED=3

| job_id | variant | K | steps | state | reason | elapsed | nodes | start_time |
| ---: | --- | ---: | ---: | --- | --- | --- | ---: | --- |
| 3288181 | `baseline` | 0 | 10 | COMPLETED | None | 00:11:35 | nvl72063-T17 | 2026-06-12T20:07:27 |
| 3288182 | `static_pard2` | 1 | 10 | COMPLETED | None | 00:12:00 | nvl72053-T15 | 2026-06-12T20:07:27 |
| 3288183 | `online_pard2` | 1 | 10 | COMPLETED | QOSMaxJobsPerUserLimit | 00:13:34 | nvl72087-T12 | 2026-06-12T20:19:15 |
