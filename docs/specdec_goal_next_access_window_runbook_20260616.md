# SpecDec Goal Next Access Window Runbook

This runbook drives the remaining active-goal proof points with restartable
small steps. It is dry-run by default.

Default dry-run for MathRL qwen32 online PARD gate plus SWE-RL qwen30 gates:

```bash
python3 scripts/run_specdec_goal_next_access_window.py --keep-going
```

When OCI-HSG SSH works, run only the missing MathRL qwen32 online PARD proof:

```bash
python3 scripts/run_specdec_goal_next_access_window.py --execute --scope math --math-phase online-pard --keep-going
```

When OCI-HSG SSH works, run MathRL references then submit the missing online PARD
row. The optional PARD-2 replay is intentionally excluded unless explicitly
requested.

```bash
python3 scripts/run_specdec_goal_next_access_window.py --execute --scope math --math-phase all --keep-going
```

Run the SWE-RL qwen30 integrated proof gates:

```bash
python3 scripts/run_specdec_goal_next_access_window.py --execute --scope swerl --swerl-phase all --keep-going
```

Run the full active-goal access-window workflow:

```bash
python3 scripts/run_specdec_goal_next_access_window.py --execute --scope all --math-phase all --swerl-phase all --keep-going
```

If the OCI-HSG SSH alias changes or the local DNS name does not resolve, override
the manifest host without editing the manifest:

```bash
python3 scripts/run_specdec_goal_next_access_window.py --execute --scope all --math-phase all --swerl-phase all --keep-going --remote-host-override <reachable-oci-host>
```

To avoid guessing manually, run the readiness wrapper. It audits candidate hosts,
requires both SSH and remote `squeue`, and only then invokes the driver with the
selected host as `--remote-host-override`:

```bash
python3 scripts/run_specdec_goal_when_remote_ready.py --execute --scope all --math-phase all --swerl-phase all --keep-going
```

MathRL and SWE-RL can also be split if they need different access aliases:

```bash
python3 scripts/run_specdec_goal_next_access_window.py --execute --scope all --math-phase all --swerl-phase all --keep-going --math-remote-host-override <math-host> --swerl-remote-host-override <swerl-host>
```

Add the online PARD-2 replay only when needed:

```bash
python3 scripts/run_specdec_goal_next_access_window.py --execute --scope math --math-phase all --include-pard2-replay --keep-going
```

Status and command output:

- Driver status: `docs/specdec_goal_next_access_window_status_20260616.csv`
- Remote-ready wrapper status:
  `docs/specdec_goal_remote_ready_driver_status_20260616.csv`
- Driver report: `docs/specdec_goal_next_access_window_status_20260616.md`
- Driver stdout/stderr logs:
  `docs/specdec_goal_next_access_window_command_outputs_20260616/`
- MathRL per-row status:
  `docs/oci_hsg_mathrl_qwen32_online_pard_gate_submit_status_20260616.csv`
- SWE-RL per-row status:
  `docs/oci_hsg_swerl_qwen30ba3b_specdec_manifest_submit_status_20260616.csv`

All helpers flush status after every processed row, so timeout-truncated runs
still leave already completed poll/submit rows and command-output paths usable.
