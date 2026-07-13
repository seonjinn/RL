# SWE-RL Qwen30 Next Access Window Runbook

This runbook drives the submit-ready manifest for the missing integrated
SWE-RL qwen30 SpecDec proof points.

Default dry-run, poll only:

```bash
python3 scripts/run_swerl_qwen30_next_access_window.py
```

When OCI-HSG SSH works, poll the existing suffix K32 r16 job first:

```bash
python3 scripts/run_swerl_qwen30_next_access_window.py --execute --phase poll
```

Submit static integrated SWE-RL SpecDec smokes:

```bash
python3 scripts/run_swerl_qwen30_next_access_window.py --execute --phase static --keep-going
```

Submit online drafter-training smokes:

```bash
python3 scripts/run_swerl_qwen30_next_access_window.py --execute --phase online --keep-going
```

Run the full sequence in one access window:

```bash
python3 scripts/run_swerl_qwen30_next_access_window.py --execute --phase all --keep-going
```

The workflow writes status rows to
`docs/oci_hsg_swerl_qwen30ba3b_specdec_manifest_submit_status_20260616.csv`.
If a remote command runs, full stdout/stderr are preserved under
`docs/oci_hsg_swerl_qwen30ba3b_specdec_manifest_command_outputs_20260616/`,
and the dashboard is rebuilt automatically unless `--no-dashboard` is passed.
