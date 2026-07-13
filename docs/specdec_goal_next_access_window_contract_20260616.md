# SpecDec Goal Next Access Window Contract

Overall: **PASS**

Driver: `/Users/sna/Nemo-RL_Qwen3_Roadmap/scripts/run_specdec_goal_next_access_window.py`
Status CSV: `/Users/sna/Nemo-RL_Qwen3_Roadmap/docs/specdec_goal_next_access_window_status_20260616.csv`

| check | status | detail |
| --- | --- | --- |
| default workflow step coverage | PASS | `math_qwen32_reference_poll, math_qwen32_online_pard_submit, swerl_qwen30_suffix_poll, swerl_qwen30_static_submit, swerl_qwen30_online_submit, dashboard_rebuild` |
| default workflow includes missing MathRL online PARD gate | PASS | `--gate-name online_pard_step5_k5_tp4_r1` |
| default workflow forwards per-command timeout | PASS | `--command-timeout 120` |
| default workflow forwards finer log chunk size | PASS | `--log-chunk-lines 20` |
| default workflow limits recent log files per poll | PASS | `--max-log-files 3` |
| default workflow excludes optional PARD-2 replay | PASS | `not present: online_pard2_step5_r12_replay` |
| default workflow includes SWE-RL online launch phase | PASS | `--action launch_online` |
| default workflow includes SWE-RL static launch phase | PASS | `--action launch_static` |
| PARD-2 replay is included only with explicit flag | PASS | `--gate-name online_pard2_step5_r12_replay` |
| math online-pard execute selects one step | PASS | `['math_qwen32_online_pard_submit']` |
| execute mode forwards --execute | PASS | `--execute` |
| execute math online-pard targets missing gate only | PASS | `--gate-name online_pard_step5_k5_tp4_r1` |
| common remote host override is forwarded to child helpers | PASS | `--remote-host-override new-oci-host` |
| Math-specific remote host override is forwarded | PASS | `--remote-host-override new-math-host` |
| SWE-RL-specific remote host override is forwarded | PASS | `--remote-host-override new-swerl-host` |
| math_qwen32_reference_poll command bash -n | PASS | `` |
| math_qwen32_reference_poll has child status path | PASS | `/Users/sna/Nemo-RL_Qwen3_Roadmap/docs/oci_hsg_mathrl_qwen32_online_pard_gate_submit_status_20260616.csv` |
| math_qwen32_online_pard_submit command bash -n | PASS | `` |
| math_qwen32_online_pard_submit has child status path | PASS | `/Users/sna/Nemo-RL_Qwen3_Roadmap/docs/oci_hsg_mathrl_qwen32_online_pard_gate_submit_status_20260616.csv` |
| swerl_qwen30_suffix_poll command bash -n | PASS | `` |
| swerl_qwen30_suffix_poll has child status path | PASS | `/Users/sna/Nemo-RL_Qwen3_Roadmap/docs/oci_hsg_swerl_qwen30ba3b_specdec_manifest_submit_status_20260616.csv` |
| swerl_qwen30_static_submit command bash -n | PASS | `` |
| swerl_qwen30_static_submit has child status path | PASS | `/Users/sna/Nemo-RL_Qwen3_Roadmap/docs/oci_hsg_swerl_qwen30ba3b_specdec_manifest_submit_status_20260616.csv` |
| swerl_qwen30_online_submit command bash -n | PASS | `` |
| swerl_qwen30_online_submit has child status path | PASS | `/Users/sna/Nemo-RL_Qwen3_Roadmap/docs/oci_hsg_swerl_qwen30ba3b_specdec_manifest_submit_status_20260616.csv` |
| dashboard_rebuild command bash -n | PASS | `` |
| math_qwen32_reference_poll command bash -n | PASS | `` |
| math_qwen32_reference_poll has child status path | PASS | `/Users/sna/Nemo-RL_Qwen3_Roadmap/docs/oci_hsg_mathrl_qwen32_online_pard_gate_submit_status_20260616.csv` |
| math_qwen32_online_pard_submit command bash -n | PASS | `` |
| math_qwen32_online_pard_submit has child status path | PASS | `/Users/sna/Nemo-RL_Qwen3_Roadmap/docs/oci_hsg_mathrl_qwen32_online_pard_gate_submit_status_20260616.csv` |
| math_qwen32_online_pard2_replay_submit command bash -n | PASS | `` |
| math_qwen32_online_pard2_replay_submit has child status path | PASS | `/Users/sna/Nemo-RL_Qwen3_Roadmap/docs/oci_hsg_mathrl_qwen32_online_pard_gate_submit_status_20260616.csv` |
| math_qwen32_online_pard_submit command bash -n | PASS | `` |
| math_qwen32_online_pard_submit has child status path | PASS | `/Users/sna/Nemo-RL_Qwen3_Roadmap/docs/oci_hsg_mathrl_qwen32_online_pard_gate_submit_status_20260616.csv` |
| math_qwen32_reference_poll command bash -n | PASS | `` |
| math_qwen32_reference_poll has child status path | PASS | `/Users/sna/Nemo-RL_Qwen3_Roadmap/docs/oci_hsg_mathrl_qwen32_online_pard_gate_submit_status_20260616.csv` |
| math_qwen32_online_pard_submit command bash -n | PASS | `` |
| math_qwen32_online_pard_submit has child status path | PASS | `/Users/sna/Nemo-RL_Qwen3_Roadmap/docs/oci_hsg_mathrl_qwen32_online_pard_gate_submit_status_20260616.csv` |
| swerl_qwen30_suffix_poll command bash -n | PASS | `` |
| swerl_qwen30_suffix_poll has child status path | PASS | `/Users/sna/Nemo-RL_Qwen3_Roadmap/docs/oci_hsg_swerl_qwen30ba3b_specdec_manifest_submit_status_20260616.csv` |
| swerl_qwen30_static_submit command bash -n | PASS | `` |
| swerl_qwen30_static_submit has child status path | PASS | `/Users/sna/Nemo-RL_Qwen3_Roadmap/docs/oci_hsg_swerl_qwen30ba3b_specdec_manifest_submit_status_20260616.csv` |
| swerl_qwen30_online_submit command bash -n | PASS | `` |
| swerl_qwen30_online_submit has child status path | PASS | `/Users/sna/Nemo-RL_Qwen3_Roadmap/docs/oci_hsg_swerl_qwen30ba3b_specdec_manifest_submit_status_20260616.csv` |
| math_qwen32_reference_poll command bash -n | PASS | `` |
| math_qwen32_reference_poll has child status path | PASS | `/Users/sna/Nemo-RL_Qwen3_Roadmap/docs/oci_hsg_mathrl_qwen32_online_pard_gate_submit_status_20260616.csv` |
| math_qwen32_online_pard_submit command bash -n | PASS | `` |
| math_qwen32_online_pard_submit has child status path | PASS | `/Users/sna/Nemo-RL_Qwen3_Roadmap/docs/oci_hsg_mathrl_qwen32_online_pard_gate_submit_status_20260616.csv` |
| swerl_qwen30_suffix_poll command bash -n | PASS | `` |
| swerl_qwen30_suffix_poll has child status path | PASS | `/Users/sna/Nemo-RL_Qwen3_Roadmap/docs/oci_hsg_swerl_qwen30ba3b_specdec_manifest_submit_status_20260616.csv` |
| swerl_qwen30_static_submit command bash -n | PASS | `` |
| swerl_qwen30_static_submit has child status path | PASS | `/Users/sna/Nemo-RL_Qwen3_Roadmap/docs/oci_hsg_swerl_qwen30ba3b_specdec_manifest_submit_status_20260616.csv` |
| swerl_qwen30_online_submit command bash -n | PASS | `` |
| swerl_qwen30_online_submit has child status path | PASS | `/Users/sna/Nemo-RL_Qwen3_Roadmap/docs/oci_hsg_swerl_qwen30ba3b_specdec_manifest_submit_status_20260616.csv` |
| timeout command-output refresh script exists | PASS | `/Users/sna/Nemo-RL_Qwen3_Roadmap/scripts/refresh_specdec_timeout_command_outputs.py` |
| driver exposes timeout refresh helper | PASS | `refresh_timeout_outputs` |
| driver timeout refresh target | PASS | `/Users/sna/Nemo-RL_Qwen3_Roadmap/scripts/refresh_specdec_timeout_command_outputs.py` |
| driver status includes timeout_refresh_state | PASS | `timeout_refresh_state` |
| driver status includes timeout_refresh_returncode | PASS | `timeout_refresh_returncode` |
| driver status includes timeout_refresh_stdout_tail | PASS | `timeout_refresh_stdout_tail` |
| driver status includes timeout_refresh_stderr_tail | PASS | `timeout_refresh_stderr_tail` |
| driver status includes child_selected_rows | PASS | `child_selected_rows` |
| driver status includes child_operation_summary | PASS | `child_operation_summary` |
| driver status includes child_preflight_summary | PASS | `child_preflight_summary` |
| driver status includes child_status_csv | PASS | `child_status_csv` |
| driver status has six rows | PASS | `6` |
| driver status follows workflow order | PASS | `math_qwen32_reference_poll, math_qwen32_online_pard_submit, swerl_qwen30_suffix_poll, swerl_qwen30_static_submit, swerl_qwen30_online_submit, dashboard_rebuild` |
| math_qwen32_reference_poll child row count | PASS | `3` |
| math_qwen32_reference_poll records DNS-blocked child preflight | PASS | `failed_dns=` |
| math_qwen32_reference_poll records child remote-unreachable operation | PASS | `dry_run_blocked_remote_unreachable=` |
| math_qwen32_reference_poll top-level operation reflects child state | PASS | `driver_child_remote_unreachable` |
| math_qwen32_online_pard_submit child row count | PASS | `1` |
| math_qwen32_online_pard_submit records DNS-blocked child preflight | PASS | `failed_dns=` |
| math_qwen32_online_pard_submit records child remote-unreachable operation | PASS | `dry_run_blocked_remote_unreachable=` |
| math_qwen32_online_pard_submit top-level operation reflects child state | PASS | `driver_child_remote_unreachable` |
| swerl_qwen30_suffix_poll child row count | PASS | `1` |
| swerl_qwen30_suffix_poll records DNS-blocked child preflight | PASS | `failed_dns=` |
| swerl_qwen30_suffix_poll records child remote-unreachable operation | PASS | `dry_run_blocked_remote_unreachable=` |
| swerl_qwen30_suffix_poll top-level operation reflects child state | PASS | `driver_child_remote_unreachable` |
| swerl_qwen30_static_submit child row count | PASS | `3` |
| swerl_qwen30_static_submit records DNS-blocked child preflight | PASS | `failed_dns=` |
| swerl_qwen30_static_submit records child remote-unreachable operation | PASS | `dry_run_blocked_remote_unreachable=` |
| swerl_qwen30_static_submit top-level operation reflects child state | PASS | `driver_child_remote_unreachable` |
| swerl_qwen30_online_submit child row count | PASS | `2` |
| swerl_qwen30_online_submit records DNS-blocked child preflight | PASS | `failed_dns=` |
| swerl_qwen30_online_submit records child remote-unreachable operation | PASS | `dry_run_blocked_remote_unreachable=` |
| swerl_qwen30_online_submit top-level operation reflects child state | PASS | `driver_child_remote_unreachable` |
