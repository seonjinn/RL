# SpecDec Goal Next Access Window Status

Summary: driver_step_ok=5

| step | helper | scope | mode | rc | timeout | operation | child ops | child preflight | stdout | stderr |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| math_qwen32_reference_poll | `submit_mathrl_qwen32_online_pard_gate_from_manifest.py` | `--action completed_reference,static_reference` | `dry_run` | 0 | false | `driver_step_ok` | `dry_run_ready=3` | `ok_controlmaster=3` | `docs/specdec_goal_next_access_window_command_outputs_20260616/2026-06-17T114625-0700_math_qwen32_reference_poll.stdout.log` | `docs/specdec_goal_next_access_window_command_outputs_20260616/2026-06-17T114625-0700_math_qwen32_reference_poll.stderr.log` |
| math_qwen32_online_pard_submit | `submit_mathrl_qwen32_online_pard_gate_from_manifest.py` | `--gate-name online_pard_step5_k5_tp4_r1` | `dry_run` | 0 | false | `driver_step_ok` | `dry_run_ready=1` | `ok_controlmaster=1` | `docs/specdec_goal_next_access_window_command_outputs_20260616/2026-06-17T114625-0700_math_qwen32_online_pard_submit.stdout.log` | `docs/specdec_goal_next_access_window_command_outputs_20260616/2026-06-17T114625-0700_math_qwen32_online_pard_submit.stderr.log` |
| swerl_qwen30_suffix_poll | `submit_swerl_qwen30_specdec_from_manifest.py` | `--action poll_existing` | `dry_run` | 0 | false | `driver_step_ok` | `dry_run_ready=1` | `ok_controlmaster=1` | `docs/specdec_goal_next_access_window_command_outputs_20260616/2026-06-17T114625-0700_swerl_qwen30_suffix_poll.stdout.log` | `docs/specdec_goal_next_access_window_command_outputs_20260616/2026-06-17T114625-0700_swerl_qwen30_suffix_poll.stderr.log` |
| swerl_qwen30_static_submit | `submit_swerl_qwen30_specdec_from_manifest.py` | `--action launch_static` | `dry_run` | 0 | false | `driver_step_ok` | `dry_run_ready=3` | `ok_controlmaster=3` | `docs/specdec_goal_next_access_window_command_outputs_20260616/2026-06-17T114625-0700_swerl_qwen30_static_submit.stdout.log` | `docs/specdec_goal_next_access_window_command_outputs_20260616/2026-06-17T114625-0700_swerl_qwen30_static_submit.stderr.log` |
| swerl_qwen30_online_submit | `submit_swerl_qwen30_specdec_from_manifest.py` | `--action launch_online` | `dry_run` | 0 | false | `driver_step_ok` | `dry_run_ready=2` | `ok_controlmaster=2` | `docs/specdec_goal_next_access_window_command_outputs_20260616/2026-06-17T114625-0700_swerl_qwen30_online_submit.stdout.log` | `docs/specdec_goal_next_access_window_command_outputs_20260616/2026-06-17T114625-0700_swerl_qwen30_online_submit.stderr.log` |

## Commands

### math_qwen32_reference_poll

```bash
/opt/homebrew/opt/python@3.14/bin/python3.14 /Users/sna/Nemo-RL_Qwen3_Roadmap/scripts/submit_mathrl_qwen32_online_pard_gate_from_manifest.py --ssh-timeout 8 --command-timeout 120 --tail-lines 80 --log-chunk-lines 20 --max-log-files 3 --remote-log-timeout 20 --remote-host-override login-lyris --action completed_reference,static_reference
```

### math_qwen32_online_pard_submit

```bash
/opt/homebrew/opt/python@3.14/bin/python3.14 /Users/sna/Nemo-RL_Qwen3_Roadmap/scripts/submit_mathrl_qwen32_online_pard_gate_from_manifest.py --ssh-timeout 8 --command-timeout 120 --tail-lines 80 --log-chunk-lines 20 --max-log-files 3 --remote-log-timeout 20 --remote-host-override login-lyris --gate-name online_pard_step5_k5_tp4_r1
```

### swerl_qwen30_suffix_poll

```bash
/opt/homebrew/opt/python@3.14/bin/python3.14 /Users/sna/Nemo-RL_Qwen3_Roadmap/scripts/submit_swerl_qwen30_specdec_from_manifest.py --ssh-timeout 8 --command-timeout 120 --tail-lines 80 --log-chunk-lines 20 --max-log-files 3 --remote-log-timeout 20 --remote-host-override login-lyris --action poll_existing
```

### swerl_qwen30_static_submit

```bash
/opt/homebrew/opt/python@3.14/bin/python3.14 /Users/sna/Nemo-RL_Qwen3_Roadmap/scripts/submit_swerl_qwen30_specdec_from_manifest.py --ssh-timeout 8 --command-timeout 120 --tail-lines 80 --log-chunk-lines 20 --max-log-files 3 --remote-log-timeout 20 --remote-host-override login-lyris --action launch_static
```

### swerl_qwen30_online_submit

```bash
/opt/homebrew/opt/python@3.14/bin/python3.14 /Users/sna/Nemo-RL_Qwen3_Roadmap/scripts/submit_swerl_qwen30_specdec_from_manifest.py --ssh-timeout 8 --command-timeout 120 --tail-lines 80 --log-chunk-lines 20 --max-log-files 3 --remote-log-timeout 20 --remote-host-override login-lyris --action launch_online
```

