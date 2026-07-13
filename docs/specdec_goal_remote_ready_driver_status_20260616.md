# SpecDec Goal Remote-Ready Driver Status

Selection: `driver_dry_run_executed`
Selected host: `login-lyris`
Mode: `dry_run`
Audit CSV: `docs/specdec_remote_access_host_audit_20260616.csv`

| field | value |
| --- | --- |
| checked_at_local | `2026-06-17T11:46:25-07:00` |
| mode | `dry_run` |
| selection_state | `driver_dry_run_executed` |
| selected_host | `login-lyris` |
| selected_remote_hostname | `login-lyris02.lyris.clusters.nvidia.com` |
| selected_remote_user | `sna` |
| selected_squeue_path | `/usr/bin/squeue` |
| host_pattern | `lyris` |
| ready_host_count | `1` |
| audit_csv | `docs/specdec_remote_access_host_audit_20260616.csv` |
| driver_command | `/opt/homebrew/opt/python@3.14/bin/python3.14 /Users/sna/Nemo-RL_Qwen3_Roadmap/scripts/run_specdec_goal_next_access_window.py --scope all --math-phase all --swerl-phase all --ssh-timeout 8 --command-timeout 120 --step-timeout 180 --tail-lines 80 --log-chunk-lines 20 --max-log-files 3 --remote-log-timeout 20 --remote-host-override login-lyris --keep-going --no-dashboard` |
| driver_returncode | `0` |
| driver_timed_out | `false` |
| driver_stdout_path | `docs/specdec_goal_remote_ready_driver_outputs_20260616/2026-06-17T114625-0700_driver.stdout.log` |
| driver_stderr_path | `docs/specdec_goal_remote_ready_driver_outputs_20260616/2026-06-17T114625-0700_driver.stderr.log` |
| driver_stdout_tail | `-remote-log-timeout 20 --remote-host-override login-lyris --gate-name online_pard_step5_k5_tp4_r1
+ /opt/homebrew/opt/python@3.14/bin/python3.14 /Users/sna/Nemo-RL_Qwen3_Roadmap/scripts/submit_swerl_qwen30_specdec_from_manifest.py --ssh-timeout 8 --command-timeout 120 --tail-lines 80 --log-chunk-lines 20 --max-log-files 3 --remote-log-timeout 20 --remote-host-override login-lyris --action poll_existing
+ /opt/homebrew/opt/python@3.14/bin/python3.14 /Users/sna/Nemo-RL_Qwen3_Roadmap/scripts/submit_swerl_qwen30_specdec_from_manifest.py --ssh-timeout 8 --command-timeout 120 --tail-lines 80 --log-chunk-lines 20 --max-log-files 3 --remote-log-timeout 20 --remote-host-override login-lyris --action launch_static
+ /opt/homebrew/opt/python@3.14/bin/python3.14 /Users/sna/Nemo-RL_Qwen3_Roadmap/scripts/submit_swerl_qwen30_specdec_from_manifest.py --ssh-timeout 8 --command-timeout 120 --tail-lines 80 --log-chunk-lines 20 --max-log-files 3 --remote-log-timeout 20 --remote-host-override login-lyris --action launch_online
/Users/sna/Nemo-RL_Qwen3_Roadmap/docs/specdec_goal_next_access_window_status_20260616.csv
/Users/sna/Nemo-RL_Qwen3_Roadmap/docs/specdec_goal_next_access_window_status_20260616.md` |
| driver_stderr_tail | `` |
| notes | `Ready host found and driver command ran.` |
