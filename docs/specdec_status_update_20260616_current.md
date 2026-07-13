# SpecDec Status Update - 2026-06-16

## MathRL Online PARD Gate Manifest - 22:40 PDT

- Added `scripts/build_mathrl_qwen32_online_pard_gate_manifest.py` and `docs/oci_hsg_mathrl_qwen32_online_pard_gate_manifest_20260616.{csv,md}`.
- The manifest keeps the completed qwen32 online PARD-2 r11 proof (`3345352`, 5/5 MathRL steps) beside the missing online PARD row:
  - online PARD launch row: `online_pard_step5_k5_tp4_r1`.
  - Shape: qwen32, `max_steps=5`, `Max OSL=1024`, `max_model_len=4096`, `4x4` GPUs, `GBS=512`, `temperature=1.0`, `top_p=1.0`, `top_k=-1`.
  - SpecDec/training contract: `DRAFT_FORMAT=pard`, `SPECDEC_METHOD=draft_model`, `NUM_SPECULATIVE_TOKENS=5`, `DRAFT_TP=4`, `PARD_ONLINE_TRAINING=true`, `POLICY_DRAFT_TYPE=pard`, `POLICY_DRAFT_LOSS=hard_ce`, `POLICY_DRAFT_MAX_TRAINING_SEQUENCE_LENGTH=256`.
  - The row mirrors the successful qwen32 online PARD-2 `pp1_tp4_drafttp4` geometry and uses `ACCOUNT=nemotron_n3_post`.
- Added `scripts/validate_mathrl_qwen32_online_pard_gate_manifest.py`; contract is `PASS` with the generated report at `docs/oci_hsg_mathrl_qwen32_online_pard_gate_contract_20260616.{json,md}`.
- Wired the new gate into `scripts/build_specdec_metrics_dashboard.py` and `scripts/build_specdec_goal_evidence_matrix.py`. The goal matrix now explicitly says online PARD is not yet functionally proven, but the qwen32 MathRL gate is submit-ready.

## MathRL Online PARD Submit/Timeout Handling - 22:46 PDT

- Added `scripts/submit_mathrl_qwen32_online_pard_gate_from_manifest.py`, a dry-run-by-default submit/poll helper for the qwen32 MathRL online PARD gate.
- The helper supports fine-grained execution filters so timeout-prone work can be split into small reusable units:
  - `--gate-name online_pard_step5_k5_tp4_r1` submits only the missing online PARD proof row.
  - `--action completed_reference` or `--action static_reference` polls only reference jobs.
  - `--tail-lines` bounds each remote log tail, so repeated refreshes stay small.
- It writes `docs/oci_hsg_mathrl_qwen32_online_pard_gate_submit_status_20260616.{csv,md}` on every run. Current dry-run status is `dry_run_blocked_remote_unreachable=5` because `oci-hsg-cs-001-vscode-02` still fails DNS from this shell.
- Added full stdout/stderr preservation for any command that actually runs. Timeout or failed command output is written under `docs/oci_hsg_mathrl_qwen32_online_pard_gate_command_outputs_20260616`, with `stdout_path` and `stderr_path` recorded in the status CSV.
- Added `scripts/validate_mathrl_qwen32_online_pard_gate_submit_contract.py`; contract is `PASS` and includes checks that fake timeout stdout/stderr are written to reusable files, execute mode requires an explicit row filter, and failed preflight does not submit.
- The exact execute command for the next access window is:
  `python3 scripts/submit_mathrl_qwen32_online_pard_gate_from_manifest.py --execute --gate-name online_pard_step5_k5_tp4_r1 --ssh-timeout 8 --command-timeout 120 --tail-lines 80`
- Wired the submit/poll status and contract into `docs/specdec_benchmark_metrics_dashboard_20260616.html` as `MathRL Qwen32 Online PARD Gate Submit/Poll Status`.
- Follow-up at `2026-06-16 22:49 PDT`: tightened both `scripts/submit_mathrl_qwen32_online_pard_gate_from_manifest.py` and `scripts/submit_swerl_qwen30_specdec_from_manifest.py` with `FLUSH_STATUS_PER_ROW=true`. Each processed row now rewrites its status CSV/Markdown immediately, so if a later SSH command times out, earlier poll/submit rows and any saved stdout/stderr paths are already usable. The MathRL qwen32 submit contract is now `PASS` with `67/67` checks; the SWE-RL qwen30 submit contract is also `PASS` and checks the same row-level flush behavior.
- Rechecked remote aliases and known FQDN fallbacks (`oci-hsg-cs-001-vscode-02`, `oci-hsg-cs-001-vscode-02.nvidia.com`, `oci-hsg-cs-001-login-01.nvidia.com`, `login-lyris`, `login-lyris.nvidia.com`, `login-lyris01`, `login-lyris02.lyris.clusters.nvidia.com`). All still fail at hostname resolution from this shell, so no remote submit/poll occurred in this pass.
- Follow-up at `2026-06-16 22:52 PDT`: added the integrated active-goal driver `scripts/run_specdec_goal_next_access_window.py` and runbook `docs/specdec_goal_next_access_window_runbook_20260616.md`. It orchestrates MathRL qwen32 reference polling, the missing `online_pard_step5_k5_tp4_r1` submit row, SWE-RL qwen30 suffix polling, SWE-RL static SpecDec smokes, SWE-RL online PARD/PARD-2 smokes, and dashboard rebuild. It is dry-run by default; optional qwen32 PARD-2 replay is excluded unless `--include-pard2-replay` is passed.
- Verified `python3 scripts/run_specdec_goal_next_access_window.py --keep-going --ssh-timeout 5 --tail-lines 40 --step-timeout 240`. It wrote `docs/specdec_goal_next_access_window_status_20260616.{csv,md}` and per-step stdout/stderr logs under `docs/specdec_goal_next_access_window_command_outputs_20260616/`. The dashboard now renders this as `SpecDec Goal Next Access Window` immediately after the Goal Evidence Matrix.
- Latest local preflight remains blocked: local `squeue` is absent, `oci-hsg-cs-001-vscode-02` has no ControlMaster and fails DNS, and `login-lyris` has no ControlMaster and fails DNS. No remote submit/poll occurred in this pass.
- Follow-up at `2026-06-16 22:55 PDT`: made the integrated driver read each child helper status CSV immediately after each step and summarize only the selected rows. The top-level driver status now distinguishes helper exit success from true remote availability: current summary is `driver_child_remote_unreachable=5, driver_step_ok=1`, with child summaries such as `dry_run_blocked_remote_unreachable=1` / `failed_dns=1` for the missing MathRL qwen32 online PARD gate and `dry_run_blocked_remote_unreachable=2` / `failed_dns=2` for SWE-RL qwen30 online PARD/PARD-2. The dashboard shows these child operation/preflight summaries directly.

## Next Access Window Driver - 22:30 PDT

- Added `scripts/run_swerl_qwen30_next_access_window.py`, a dry-run-by-default orchestrator around the qwen30 SWE-RL manifest helper:
  - `--phase poll` polls the already-submitted suffix K32 r16 job.
  - `--phase static` selects Eagle-3, PARD, and PARD-2 static step1 smokes.
  - `--phase online` selects online PARD and online PARD-2 step1 smokes.
  - `--phase all` selects all six manifest rows in a single helper invocation, preserving one aggregate status CSV.
- Added `docs/swerl_qwen30_next_access_window_runbook_20260616.md` with the exact dry-run and execute commands for the next SSH access window.
- Verified `python3 scripts/run_swerl_qwen30_next_access_window.py --phase all --ssh-timeout 5 --tail-lines 40`: it selected all six manifest rows, wrote `docs/oci_hsg_swerl_qwen30ba3b_specdec_manifest_submit_status_20260616.csv`, and rebuilt the dashboard. Current operation state remains `dry_run_blocked_remote_unreachable` for all rows because OCI-HSG DNS still fails.

## SWE-RL Submit Output Preservation - 22:28 PDT

- Rechecked local SSH routing. `~/.ssh/config` has direct aliases for `oci-hsg-cs-001-vscode-0[1-3]` and `login-lyris`, but there is no ProxyJump. All tried hostnames still return `Could not resolve hostname` / `no_dns`, so no remote poll/submit happened.
- Hardened `scripts/submit_swerl_qwen30_specdec_from_manifest.py` for the next access window:
  - Added `stdout_path` and `stderr_path` fields to the submit-status CSV.
  - Added `--command-output-dir`, defaulting to `docs/oci_hsg_swerl_qwen30ba3b_specdec_manifest_command_outputs_20260616`.
  - When `--execute` reaches a remote command, full stdout/stderr are written to per-row log files, so timeout/failed remote output is not limited to the CSV tail.
- Updated dashboard rendering so `SWE-RL Qwen30 Manifest Submit/Poll Status` shows the stdout/stderr log path columns.
- Updated `scripts/validate_swerl_qwen30_manifest_submit_contract.py`; contract is now `PASS` with `102/102` checks, including the output-preservation fields.
- Rebuilt and reopened `docs/specdec_benchmark_metrics_dashboard_20260616.html`.

## Objective Audit Refresh - 22:26 PDT

- Re-ran cluster preflight from this shell. Local `squeue` is unavailable, and both remote hosts still fail before authentication:
  - `oci-hsg-cs-001-vscode-02`: `Could not resolve hostname`.
  - `login-lyris`: `Could not resolve hostname`.
  - No ControlMaster socket exists for either host, so no remote poll/submit was possible in this pass.
- Strengthened `scripts/build_specdec_goal_evidence_matrix.py` so the dashboard now distinguishes implementation contracts from functional/performance proof:
  - `Online drafter training implementation`: local contract pass for PARD and PARD-2 online launcher/source paths.
  - `Online PARD training`: explicitly marked `not yet functionally proven` because no completed NeMo-RL online PARD training-step run exists in current artifacts.
  - `Online PARD-2 performance impact`: explicitly marked `measured negative` for Qwen3-8B online-vs-static PARD-2: `+0.717pp` acceptance, but `0.9696x` generation TPS and `0.8087x` E2E TPS versus static.
- Rebuilt and reopened `docs/specdec_benchmark_metrics_dashboard_20260616.html`; the `Goal Evidence Matrix` now shows these rows before the detailed Math/SWE sections.

## Timeout Partial-Log Handling - 22:24 PDT

- Updated `scripts/refresh_oci_nemorl_partial_logs.py` so timeout-prone refreshes can be split more finely:
  - `--job-id` filters to one or more selected job ids.
  - `--limit-jobs` processes only the first N enabled manifest rows after filtering.
  - `--build-dashboard-per-job` rebuilds `docs/specdec_benchmark_metrics_dashboard_20260616.html` after each job flush, so a later timeout does not hide already parsed results.
- Verified the new path with `python3 scripts/refresh_oci_nemorl_partial_logs.py --build-dashboard-per-job`. All three partial qwen235B MathRL logs flushed independently:
  - `3334220` baseline: `8/9` seen steps completed and immediately merged into the MathRL live summary.
  - `3333717` suffix K32: `14/15` seen steps completed and immediately merged.
  - `3333537` Eagle-3: `14/15` seen steps completed and immediately merged.
- Current reusable partial outputs are still `docs/oci_hsg_nemorl_partial_artifacts_20260616.csv` and `docs/oci_hsg_nemorl_partial_step_index_20260616.csv`: 3 artifact rows, 39 parsed step chunks, and 36 completed chunks marked `usable_for_metrics=true`.
- Updated `scripts/build_specdec_metrics_dashboard.py` to regenerate and render `Qwen8 PARD-2 Online Impact` from `docs/qwen8_pard2_official_online_impact_20260613.csv` and `docs/qwen8_pard2_official_comparison_metrics_20260613.csv`. Current read: online PARD-2 refit ran for 9 steps and improved acceptance by `+0.717` percentage points, but throughput stayed below static (`0.9696x` generation TPS, `0.8087x` E2E TPS).
- Latest dashboard was reopened locally: `docs/specdec_benchmark_metrics_dashboard_20260616.html`.

## OCI-HSG Incremental Update - 21:26 PDT

- Follow-up at `2026-06-16 22:19 PDT`: remote access is still blocked from this shell (`oci-hsg-cs-001-vscode-02` and `login-lyris` fail at hostname resolution; no ControlMaster socket is present). While waiting for DNS/SSH recovery, strengthened the local online PARD/PARD-2 source-bundle validation:
  - `scripts/validate_nemorl_pard_source_bundle.py` now checks the previous failure points explicitly: hard-label draft CE support (`teacher_token_ids`, `DistributedLogprob`, `draft_global_valid_toks`) and sequence-parallel PARD-2 target-feature alignment (`get_tensor_model_parallel_rank`, `gather_from_sequence_parallel_region`, local sequence slicing).
  - `docs/nemorl_pard_source_bundle_validation_20260616.json` / `.md` report `PASS`: 12/12 source checks and 11/11 compile checks pass.
  - `scripts/build_specdec_metrics_dashboard.py` now regenerates and renders this as `NeMo-RL PARD/PARD-2 Source Bundle Contract` in `docs/specdec_benchmark_metrics_dashboard_20260616.html`.
  - `scripts/test_pard2_target_feature_alignment.py` still cannot run in this local Python environment because `torch` is not installed; it remains the right tensor-level test to run on the cluster/container.
- Follow-up at `2026-06-16 22:16 PDT`: reran SSH preflight for `oci-hsg-cs-001-vscode-02`, `login-lyris`, and FQDN fallbacks from known_hosts (`oci-hsg-cs-001-vscode-02.nvidia.com`, `oci-hsg-cs-001-login-01.nvidia.com`, `login-lyris.nvidia.com`, `login-lyris01`, `login-lyris02.lyris.clusters.nvidia.com`). All failed at hostname resolution from this shell, so no remote poll/submit occurred in this pass.
- Local contract hardening at `2026-06-16 22:16 PDT`: extended `scripts/validate_nemorl_online_specdec_contract.py` with a missing `pard2-static` case and wired the validator into `scripts/build_specdec_metrics_dashboard.py`. The dashboard now renders `NeMo-RL Online SpecDec Launcher Contract`; `docs/nemorl_online_specdec_contract_20260616.json` / `.md` report `PASS` with `7/7` cases passing: baseline-no-spec, suffix-static, pard-static, pard-online, pard2-static, pard2-online, and the negative k-slot window guard.
- Queue update at `2026-06-16 22:17 PDT`: `scripts/build_specdec_next_experiment_queue.py` now emits explicit queue rows for qwen30 SWE-RL online PARD K5 and online PARD-2 K3, in addition to suffix poll, static Eagle-3, static PARD, static PARD-2, and the qwen235B MathRL short gate. The dashboard `Next Experiment Queue` now shows those online rows as `not launched in current tracker` and blocked only by the same OCI-HSG DNS/SSH recovery.
- Follow-up at `2026-06-16 22:12 PDT`: tightened timeout handling for local partial-log refresh. `scripts/refresh_oci_nemorl_partial_logs.py` now flushes `docs/oci_hsg_nemorl_partial_artifacts_20260616.csv`, `docs/oci_hsg_nemorl_partial_step_index_20260616.csv`, and the merged MathRL live summary after each processed job, replacing only that `job_id` in the shared indexes. If a refresh or SSH/log-copy loop times out later, completed step chunks already parsed from earlier jobs remain immediately usable. Re-ran `python3 scripts/refresh_oci_nemorl_partial_logs.py --build-dashboard`: the index still has 3 timeout/failed NeMo-RL runs, 39 parsed step chunks, and 36 completed chunks marked `usable_for_metrics=true`.
- Dashboard follow-up at `2026-06-16 22:12 PDT`: added and wired `scripts/validate_swerl_qwen30_manifest_submit_contract.py` into `scripts/build_specdec_metrics_dashboard.py`. The dashboard now renders `SWE-RL Qwen30 Manifest Command Contract`, and the validator report is `PASS` with `99/99` checks passing in `docs/oci_hsg_swerl_qwen30ba3b_manifest_submit_contract_20260616.json` / `.md`. This keeps the safe qwen30 SWE-RL poll/launch contract visible even when DNS/SSH or remote commands time out.
- Follow-up at `2026-06-16 22:06 PDT`: tightened the SWE-RL qwen30 submit helper after auditing the local SWE launcher contract. The qwen30 launcher path used for suffix retries has a pre-Hydra method gate that only allows `suffix`/`ngram`, so non-suffix rows now stage a patched copy of the launcher under the run log directory before launch. The patch only relaxes the gate to allow `draft_model`, `pard2`, and `eagle3`; the actual SpecDec settings still come from the manifest Hydra overrides. Current dry-run status shows `ok; stages patched non-suffix launcher` for static Eagle-3, static PARD, static PARD-2, online PARD, and online PARD-2. Direct SSH preflight still fails at DNS for both `oci-hsg-cs-001-vscode-02` and `login-lyris`, so no remote submit/poll occurred.
- Safety fix at `2026-06-16 22:07 PDT`: regenerated the submit-status CSV after changing generated dry-run commands to pass `DRY_RUN=true` literally. The SWE launcher checks the literal string `true`, so this prevents copied dry-run commands from behaving like submit commands.

- Follow-up at `2026-06-16 22:04 PDT`: added a manifest-driven SWE-RL qwen30 submit/poll helper:
  - `scripts/submit_swerl_qwen30_specdec_from_manifest.py`
  - `docs/oci_hsg_swerl_qwen30ba3b_specdec_manifest_submit_status_20260616.csv`
  - `docs/oci_hsg_swerl_qwen30ba3b_specdec_manifest_submit_status_20260616.md`
- The helper is dry-run by default and writes status rows even when SSH/DNS fails. Current dry-run result: all six manifest rows validate (`manifest_validation=ok`), suffix poll row now carries explicit `job_id=3351394`, and every row is `dry_run_blocked_remote_unreachable` because `oci-hsg-cs-001-vscode-02` still returns `Could not resolve hostname`. The dashboard now renders this as `SWE-RL Qwen30 Manifest Submit/Poll Status`.

- Follow-up at `2026-06-16 22:00 PDT`: reran `python3 scripts/refresh_oci_nemorl_partial_logs.py --build-dashboard`. The partial-log workflow is current again: `docs/oci_hsg_nemorl_partial_artifacts_20260616.csv` has 3 timeout/failed runs, and `docs/oci_hsg_nemorl_partial_step_index_20260616.csv` has 39 step chunks. Of those, 36 completed chunks are marked `usable_for_metrics=true` and can be consumed immediately; the three failed tail chunks are retained with the NCCL watchdog root-cause line and `usable_for_metrics=false`. The refreshed HTML dashboard is `docs/specdec_benchmark_metrics_dashboard_20260616.html`. SSH remains blocked from this shell because both `oci-hsg-cs-001-vscode-02` and `login-lyris` still fail hostname resolution.

- Follow-up at `2026-06-16 21:55 PDT`: local SSH/SLURM access is still unavailable from this shell (`login-lyris` and `oci-hsg-cs-001-vscode-02` do not resolve, and there is no `login-lyris` ControlMaster socket), so no remote poll/submit was possible. To keep progress moving, I added a submit-ready SWE-RL Qwen3-30B-A3B SpecDec launch manifest and wired it into the dashboard:
  - `scripts/build_swerl_qwen30_specdec_launch_manifest.py`
  - `docs/oci_hsg_swerl_qwen30ba3b_specdec_launch_manifest_20260616.csv`
  - `docs/oci_hsg_swerl_qwen30ba3b_specdec_launch_manifest_20260616.md`
- The manifest preserves the successful qwen30 SWE-RL ctx40k/vLLM40k geometry (`max_steps=1`, context cap `40960`, `TOTAL_NODES=8`, `GBS=64`, `VLLM_TP=1`, Ray `2.54.0`, Python `3.13.13`, account `nemotron_n3_post`) and records RL sampling (`temperature=1.0`, `top_p=1.0`, `top_k=-1`). It has six rows: poll existing suffix r16 `3351394`, launch static Eagle-3 K3, static PARD K5, static PARD-2 K3, online PARD K5, and online PARD-2 K3. Each launch row includes the exact env overrides, expected Hydra overrides, validation gates, log directory, and expected CSV artifacts.
- `scripts/build_specdec_metrics_dashboard.py` now regenerates the SWE-RL qwen30 launch manifest automatically and renders a new `SWE-RL Qwen30 SpecDec Launch Manifest` section in `docs/specdec_benchmark_metrics_dashboard_20260616.html`.

- Log probing was split into small `sacct`, `squeue`, `grep`, and `sed` calls so timeout-truncated SSH output still leaves usable rows. New/updated local artifacts:
  - `docs/oci_hsg_failed_job_triage_20260616.csv`
  - `docs/oci_hsg_failed_job_triage_20260616.md`
  - `docs/oci_hsg_nemorl_partial_log_manifest_20260616.csv`
  - `docs/oci_hsg_nemorl_partial_artifacts_20260616.csv`
  - `docs/oci_hsg_nemorl_partial_step_index_20260616.csv`
  - `docs/specdec_goal_evidence_matrix_20260616.csv`
  - `docs/specdec_next_experiment_queue_20260616.csv`
  - `docs/oci_hsg_mathrl_qwen235b_baseline_step20_3334220_partial_metrics_20260616.csv`
  - `docs/oci_hsg_mathrl_qwen235b_baseline_step20_3334220_partial_summary_20260616.csv`
  - `docs/oci_hsg_mathrl_qwen235b_suffix_step20_3333717_partial_metrics_20260616.csv`
  - `docs/oci_hsg_mathrl_qwen235b_suffix_step20_3333717_partial_summary_20260616.csv`
  - `docs/oci_hsg_mathrl_qwen235b_eagle3_step20_3333537_partial_metrics_20260616.csv`
  - `docs/oci_hsg_mathrl_qwen235b_eagle3_step20_3333537_partial_summary_20260616.csv`
  - `docs/oci_hsg_mathrl_qwen235b_main_baseline_fixed256_3342356_metrics_20260616.csv`
  - `docs/oci_hsg_mathrl_qwen235b_main_baseline_fixed256_3342356_summary_20260616.csv`
  - `docs/oci_hsg_swerl_qwen30ba3b_baseline_ctx40k_3344823_metrics_20260616.csv`
  - `docs/oci_hsg_swerl_qwen30ba3b_baseline_ctx40k_3344823_summary_20260616.csv`
  - `docs/oci_hsg_live_job_status_20260616.csv`
- Added `scripts/refresh_oci_nemorl_partial_logs.py` so copied `ray-driver.log` files can be refreshed into per-step metrics, summary CSVs, the Math RL dashboard summary, and the HTML dashboard with one command:
  `python3 scripts/refresh_oci_nemorl_partial_logs.py --build-dashboard`.
- Timeout/failed NeMo-RL logs are now indexed in `docs/oci_hsg_nemorl_partial_artifacts_20260616.csv` with `partial_result_state`, completed/incomplete step spans, output CSV paths, and the prioritized root-cause line. They are also split into a finer step-level index at `docs/oci_hsg_nemorl_partial_step_index_20260616.csv`; rows with `usable_for_metrics=true` completed a full step and can be consumed immediately, while the incomplete final step is retained for triage but excluded from mean timing/throughput summaries when completed steps exist.
- Added `scripts/build_specdec_next_experiment_queue.py` and wired it into the HTML dashboard. The current queue makes the remaining goal gaps explicit: poll SWE-RL suffix r16 `3351394`, then launch Qwen3-30B-A3B SWE-RL step1 ctx40k Eagle-3 K3, PARD K5, and PARD-2 K3 smokes under `nemotron_n3_post`; it also records a shorter qwen235B MathRL gate as the next clean-up path for the partial step20 results.
- Added `scripts/build_specdec_goal_evidence_matrix.py` and wired it into `scripts/build_specdec_metrics_dashboard.py`, so the dashboard `Goal Evidence Matrix` is regenerated from current CSV artifacts instead of maintained by hand. It currently says: qwen32 online PARD-2 is functionally proven for 5 MathRL steps but not a speedup win yet; qwen30/qwen32 static MathRL SpecDec is positive; qwen235B MathRL is partial-positive but not clean; SWE-RL baseline is proven; SWE-RL specdec integrated proof is still missing.
- qwen235B MathRL step20 baseline `3334220` is now confirmed `FAILED 1:0` after `01:14:56`. It completed `8/20` steps and failed during Step 9 policy training from Megatron `ProcessGroupNCCL` watchdog collective timeouts. Partial metrics over completed steps: Max OSL `1024`, mean E2E `12.45 tok/s/GPU`, generation worker `17.67 tok/s/GPU`, mean generation time `127.53s`.
- qwen235B MathRL step20 suffix K32 `3333717` is now confirmed `FAILED 1:0` after `01:24:00`. It completed `14/20` steps and failed during Step 15 policy training from NCCL watchdog timeouts including expert `ALLTOALL_BASE`, tensor `_ALLGATHER_BASE`, and pipeline `COALESCED`. Partial metrics: Max OSL `1024`, mean E2E `16.69 tok/s/GPU`, generation worker `24.19 tok/s/GPU`, weighted acceptance `26.35%`, mean accept len `1.74`.
- qwen235B MathRL step20 Eagle-3 `3333537` is now confirmed `FAILED 1:0` after `01:21:59`. It completed `14/20` steps and failed during Step 15 policy training from Megatron `ProcessGroupNCCL` watchdog collective timeouts across tensor/context/expert/pipeline groups. Partial metrics over completed steps: mean E2E `20.83 tok/s/GPU`, generation worker `34.16 tok/s/GPU`, weighted acceptance `47.42%`, mean accept len `2.42`, mean generation time `66.25s`.
- qwen235B main fixed256 baseline `3342356` completed and was parsed: Step 1 total `287.79s`, generation `36.33s`, E2E `2.57 tok/s/GPU`, generation worker `20.37 tok/s/GPU`, mean generation length `256`.
- SWE-RL qwen30BA3B ctx40k baseline `3344823` completed and was parsed: Step 1 total `141.04s`, E2E `190.8 tok/s/GPU`, generation worker `559.5 tok/s/GPU`.
- SWE-RL qwen30BA3B suffix K32 arctic py313 retry `3351394` was last confirmed `RUNNING`; it has been added to `latest_oci_hsg_swerl_qwen30ba3b_step1_smoke_20260616_jobs.csv`. At `00:09:04` elapsed, Ray had reached `64/64` actors and `All workers connected!`; the previous `arctic-inference` ImportError had not reappeared in copied logs.
- `3342358` PARD K3 fixed256 triage is updated: the ray-driver failure appears as `ActorDiedError` during `prepare_refit_info()`, and `sacct` now confirms child steps `.5` and `.8` ended `OUT_OF_MEMORY 0:125`.
- Current local access blocker: `oci-hsg-cs-001-vscode-0[1-3]` and `login-lyris` do not resolve in DNS from this shell. `ssh -O check login-lyris` also has no ControlMaster socket, so no further remote polling was possible in this pass.

## Lyris Standalone

- Original SWE OSL32K jobs `2133882`, `2133883`-`2133892` failed before benchmark execution because `/lustre/fsw/coreai_dlalgo_llm` hit the user inode hard limit. `sacct` shows the listed jobs all started at `2026-06-15T23:38:31`, failed after `00:00:17`, and the batch steps were `CANCELLED` with exit code `0:53`. SLURM could not create the configured `slurm-%j.out`; the run dirs exist, but `slurm-<jobid>.out` and `breakdown.json` are missing for all listed jobs. Current quota evidence shows `31,457,280` files against an inode hard limit of `31,457,280`.
- Retry jobs under `/home/sna/vllm-runs` are the valid source for current standalone metrics.
- Latest temp1 `/home` retry status for the failed `2133883`-`2133892` replacement matrix:
  - qwen235B suffix `2133936` completed.
  - qwen235B baseline `2133935` and Eagle-3 `2133937` reached the 5h walltime and timed out.
  - qwen32 baseline/suffix/Eagle-3 `2133938`/`2133939`/`2133940` completed.
  - qwen30BA3B baseline/suffix/Eagle-3 `2133941`/`2133942`/`2133943` completed.
- Some early `/home` retries wrote `breakdown.json` inside the container namespace because `/home/sna/vllm-runs` was not mounted into the container. Those runs can show `wrote .../breakdown.json` in `slurm-*.out` while the host file is missing. New retries explicitly mount `/home/sna/vllm-runs:/home/sna/vllm-runs`.
- Refreshed CSVs:
  - `docs/lyris_swebench_osl32k_temp01_core_matrix_20260616_metrics_live.csv`
  - `docs/lyris_math500_osl32k_temp01_home_retry_metrics_live_20260616.csv`
  - `docs/lyris_swebench_osl32k_temp01_core_matrix_20260616_status_live.csv`
- Dashboard refreshed at `docs/specdec_benchmark_metrics_dashboard_20260616.html` at `2026-06-16 20:51 PDT`. It now includes `/home` retry metrics, `/project` public-PARD Math rows, timeout/live Slurm telemetry, OCI qwen8 PARD-1 r4 no-profiler JSON-backed metrics, and qwen32 online PARD-2 r11 final 5-step metrics. The SWE partial telemetry section is structured so timeout jobs can expose progress, latest vLLM throughput, acceptance, and latest error before `breakdown.json` exists. `scripts/refresh_lyris_qwen235b_standalone_fast.py` now scans in smaller live batches by default and flushes `_live_progress.csv` plus `_completed_runs.txt` after every completed batch or fallback single-job scan, so timeout-interrupted refreshes preserve the partial rows already obtained. It was further updated so the remote scanner emits one JSONL record per job and flushes immediately; if SSH times out, parsed stdout rows are preserved and only unscanned jobs fall back to single-job probes. Refresh scripts: `scripts/refresh_lyris_temp01_live_metrics.py`, `scripts/refresh_lyris_qwen235b_standalone_fast.py`, `scripts/refresh_oci_hsg_qwen8_pard1_standalone.py`, and `scripts/build_specdec_metrics_dashboard.py`.
- Last successful Lyris refresh has `41` SWE metric rows and `38` Math metric rows from `48` completed `breakdown.json` files; the SWE status table has `52` terminal rows (`26` completed, `21` failed, `5` timed out) and no remaining active jobs in the `/home` tracked matrix. The newer `/project` public-PARD Math jobs are still active in the last local snapshot, with eight public-PARD batch-1 rows and two public-PARD batch-2 rows parsed. Current shell check at `2026-06-16 16:25 PDT` did not have a live `login-lyris` ControlMaster socket (`Permission denied (keyboard-interactive)`), so new Lyris tails could not be fetched in this pass. Speedups are only computed against a matched baseline with the same domain, model, temperature, batch size, TP/PP, OSL, and prompt shape.
- Timeout/failed standalone rows are now also scanned for partial vLLM logger telemetry. The partial CSV `docs/lyris_swebench_osl32k_temp01_core_matrix_20260616_partial_live_progress.csv` has `52` SWE launch rows, `35` rows with live generation telemetry, and `9` rows with live telemetry but no final `breakdown.json`. Examples now visible in the dashboard include timed-out qwen235B baselines `2133935` (`8.3` live gen tok/s), `2136598` (`8.1`), and `2136600` (`8.6`), plus failed-but-informative Eagle-3 rows `2133875` (`20.1` gen tok/s, `63.8%` acceptance) and `2133879` (`24.1` gen tok/s, `54.7%` acceptance). These are provisional logger aggregates, not final tok/s/GPU rows.
- OCI-HSG supplemental qwen3-8B standalone PARD-1 r3 jobs for Math/SWE at temperature `0.0` and `1.0` with ISL/OSL `4096/32768` completed prompt generation in their logs but wrote `0` final `breakdown.json` files; several children ended with `EngineCore_DP0 died unexpectedly`. `driver.log` provisional parsing yields all 12 rows in `docs/oci_hsg_qwen8_pard1_standalone_temp01_20260616_r3_metrics.csv`, but these are not final JSON-backed rows.
- OCI-HSG qwen3-8B standalone PARD-1 r4 no-profiler retry was launched under `nemotron_n3_post` with the same Math/SWE, temp `0.0`/`1.0`, ISL/OSL `4096/32768`, baseline plus PARD K3/K5 matrix. Tracker: `latest_oci_hsg_qwen8_pard1_standalone_temp01_20260616_r4_noprof_jobs.csv`. All `12` parent jobs are `COMPLETED 0:0`, and `24` child rows are parsed from final `breakdown.json` files in `docs/oci_hsg_qwen8_pard1_standalone_temp01_20260616_r4_noprof_metrics.csv`.
- OCI qwen8 PARD-1 r4 JSON-backed highlights:
  - Math temp0: batch1 baseline `41.49 tok/s/GPU`, PARD K3 `1.13x`, PARD K5 `1.23x`; batch2 baseline `84.1 tok/s/GPU`, PARD K3 `0.94x`, PARD K5 `1.07x`.
  - Math temp1: batch1 baseline `41.80 tok/s/GPU`, PARD K3 `0.92x`, PARD K5 `1.24x`; batch2 baseline `83.7 tok/s/GPU`, PARD K3 `0.86x`, PARD K5 `0.97x`.
  - SWE temp0: batch1 baseline `42.26 tok/s/GPU`, PARD K3 `1.28x`, PARD K5 `1.48x`; batch2 baseline `74.1 tok/s/GPU`, PARD K3 `1.11x`, PARD K5 `0.95x`.
  - SWE temp1: batch1 baseline `41.07 tok/s/GPU`, PARD K3 `1.07x`, PARD K5 `1.37x`; batch2 baseline `79.8 tok/s/GPU`, PARD K3 `0.87x`, PARD K5 `1.00x`.
- Lyris Math PARD-1 gap: previous qwen30BA3B PARD K3/K5 jobs `2135744`/`2135745` failed before generation because the configured local PARD draft checkpoint path was not a valid HF directory/config. A new qwen8/qwen30BA3B Math PARD K3/K5 Lustre retry was attempted with `amd/PARD-Qwen3-0.6B`, but it could not create the run directory because `/lustre/fsw` user inode quota is at the hard limit: `files=31,457,280`, `ilimit=31,457,280`. `lfs quota -v` shows the pressure concentrated on `lfs5-MDT0013`/MDT index `19`, and all existing `/lustre/fsw/coreai_dlalgo_llm/users/sna` top-level directories are on that MDT. `vllm-benchmark/vllm-runs` itself is about `77k` inodes, so the pressure is from the broader user tree or MDT accounting, not the current standalone results directory alone.
- Because `/lustre/fsw` cannot create any new inode for user `sna`, the missing Math PARD-1 Lyris jobs were resubmitted with code/model reads from `/lustre` but logs/results under writable `/project/coreai_dlalgo_llm/sna/vllm-runs/math_temp01_core_20260616_project_publicpard`, mounted into the container via `/project:/project`. Tracker: `latest_lyris_math500_osl32k_temp01_project_publicpard_20260616_jobs.csv`. Jobs: qwen8 temp0 K3/K5 `2139656`/`2139657`, qwen8 temp1 K3/K5 `2139658`/`2139659`, qwen30BA3B temp0 K3/K5 `2139660`/`2139661`, qwen30BA3B temp1 K3/K5 `2139662`/`2139663`. Initial `squeue` showed qwen8 K3/K5 jobs running and qwen30BA3B jobs configuring.
- Latest Lyris `/project` poll at `2026-06-16 15:54 PDT`: qwen8 temp1 PARD K3 `2139658` is `COMPLETED 0:0` after `01:35:40`; the other 7 public-PARD Math jobs are still `RUNNING` after about `109` minutes, with about `3:11` walltime left. The latest refresh parsed a second public-PARD batch-2 row from qwen8 temp1 PARD K5 `2139659`: `77.0 tok/s/GPU`, `1.06x`, `37.7%` acceptance, mean accept len `2.89`. qwen8 temp1 PARD K3 `2139658` batch2 remains `91.7 tok/s/GPU`, `1.27x`, `72.8%` acceptance, mean accept len `3.18`. Other batch-2 rows are still pending. Parsed batch-1 highlights: qwen8 temp0 PARD K3/K5 `1.03x`/`1.09x` (`50.7%`/`34.5%` acceptance); qwen8 temp1 PARD K3/K5 `1.32x`/`1.04x` (`68.6%`/`29.1%` acceptance); qwen30BA3B temp0 PARD K3/K5 `1.92x`/`2.20x` (`70.7%`/`52.2%` acceptance); qwen30BA3B temp1 PARD K3/K5 `1.87x`/`1.91x` (`63.7%`/`41.1%` acceptance).
- SWE temp1 Qwen3-30B-A3B has matched completed baseline/suffix/Eagle-3 rows:
  - suffix K32: `8.61x` at batch 1, `7.91x` at batch 2; acceptance about `90-91%`.
  - Eagle-3 K3: `1.09x` at batch 1, `1.04x` at batch 2; acceptance about `11%`.
- SWE temp0 Qwen3-32B Eagle-3 replacement has completed rows:
  - Eagle-3 K3: `19.4 tok/s/GPU` batch 1 and `39.2 tok/s/GPU` batch 2; acceptance `66.9%`/`71.3%`. Same-temperature speedups are waiting on a completed temp0 baseline.
- Math temp1 Qwen3-8B PARD2 standalone rows are now parsed:
  - PARD2 K3: `28.1 tok/s/GPU`, `26.9%` acceptance, mean accept len `1.81`.
  - PARD2 K5: `27.5 tok/s/GPU` batch 1 and `34.6 tok/s/GPU` batch 2; acceptance `16.3%`/`8.4%`, mean accept len `1.81`/`1.42`.
  - K3 and K5 are now keyed by the unique `logs_dir`/K-bearing label and no longer collapse into one PARD2 method row.
- Math temp1 Qwen3-30B-A3B matched baseline/suffix/Eagle-3 rows are parsed:
  - baseline `2135741`: `19.36`/`38.82 tok/s/GPU` at batch 1/2.
  - suffix K32: `144.2`/`181.4 tok/s/GPU`, `7.45x`/`4.67x` at batch 1/2; acceptance `87.6%`/`80.1%`.
  - Eagle-3 K3: `22.0`/`42.9 tok/s/GPU`, `1.14x`/`1.11x` at batch 1/2; acceptance about `10-11%`.
- New Math temp0 standalone rows parsed from the fixed `/home` mount:
  - Qwen3-8B baseline `2136605`: temp0 batch 1/2 parsed at `37.2`/`74.4 tok/s/GPU`.
  - Qwen3-8B suffix K32 `2136608`: `216.3 tok/s/GPU` at batch 1 and `409.4 tok/s/GPU` at batch 2; batch-1 speedup is `5.81x`; acceptance `84.5%`/`87.4%`, mean accept len `6.88`/`9.31`.
  - Qwen3-8B Eagle-3 K3 `2136610`: batch 1/2 rows are parsed at `73.0`/`116.6 tok/s/GPU`; batch-1 speedup is `1.96x`; acceptance `62.9%`/`63.6%`, mean accept len about `2.89`.
  - Qwen3-30B-A3B baseline `2136607`: temp0 batch 1/2 parsed at `19.38`/`39.00 tok/s/GPU`.
  - Qwen3-30B-A3B Eagle-3 K3 `2135739`: batch 1/2 rows are parsed at `21.4`/`40.7 tok/s/GPU`, speedup `1.11x`/`1.04x`, acceptance `10.1%`/`10.3%`, mean accept len about `1.30`.
  - Qwen3-30B-A3B suffix K32 `2136609`: batch 1/2 rows are available at `143.2`/`269.2 tok/s/GPU`, speedup `7.39x`/`6.90x`, `88.4%`/`88.8%` acceptance, mean accept len `7.85`/`8.96`.
- SWE temp0 Qwen3-235B Eagle-3 `2135970` now has parsed batch 1/2 rows: `5.35`/`8.33 tok/s/GPU`, acceptance `57.2%`/`49.4%`. Strict speedup is still waiting on the matching temp0 qwen235B baseline split jobs `2136598`/`2136599`.
- Qwen3-235B SWE temp1 suffix K32 is not missing: `/home` retry `2133936` is parsed with batch 1/2 rows (`4.30`/`7.80 tok/s/GPU`, acceptance `46.8%`/`50.6%`). The current split baseline jobs `2136600`/`2136601` are still running, so strict matched speedups remain blank until those complete.
- SWE temp0/temp1 follow-up rows with the fixed `/home/sna/vllm-runs` mount:
  - qwen30 temp0 baseline `2135585` completed.
  - qwen235B temp0 Eagle-3 `2135970` completed and is parsed.
  - qwen32 Eagle-3 `2135971` completed and is included in the refreshed metrics.
  - qwen235B bs2 baseline split jobs completed and are parsed: temp0 `2136599` and temp1 `2136601`. These now provide matched qwen235B bs2 speedups for suffix/Eagle-3.
  - qwen235B bs1 baseline split jobs with prompt_count=4 timed out at the 5h walltime: temp0 `2136598` and temp1 `2136600`. The prompt_count=1 matched mini-matrix completed for qwen235B SWE temp0/temp1 with baseline, suffix K32, and Eagle-3 K3. It now fills strict bs1 speedups: temp0 n=1 suffix K32 `5.96x` and Eagle-3 K3 `2.66x`; temp1 n=1 suffix K32 `2.86x` and Eagle-3 K3 `1.61x`. The first suffix submissions `2138632`/`2138635` failed immediately because the new outdirs missed `arctic-inference==0.1.1`; they were relaunched after writing that requirement. The refresh script includes `prompt_count_used` in the baseline matching key so n=1 and n=4 rows do not share speedups.
  - qwen235B temp1 Eagle-3 split by batch: `2136602` batch 1 and `2136603` batch 2 completed and are parsed. Batch 1 is `3.22 tok/s/GPU` with `21.1%` acceptance and mean accept len `1.63`; batch 2 is `6.14 tok/s/GPU`, `23.1%` acceptance, mean accept len `1.69`. Strict batch-1 speedup is still waiting on a matched qwen235B bs1 baseline because `2136600` timed out.
  - qwen32/qwen30BA3B temp0 baseline retries with the fixed `/home/sna/vllm-runs` mount completed and are included in the `41` SWE metric rows. These replace failed/missing baseline artifacts where the earlier jobs finished generation but failed or lost `breakdown.json` due mount/quota issues.
  - Mount validation: sampled `run.sbatch` files include `--container-mounts='/lustre:/lustre,/home/sna/vllm-runs:/home/sna/vllm-runs,...'`; qwen235B baseline has loaded all 118 checkpoint shards and is generating.
- Additional Math standalone rows launched with the fixed `/home` mount:
  - qwen8 Temp 0 baseline `2136605` and qwen8 Temp 1 baseline `2136606` completed and both batch 1/2 rows are parsed. Temp1 baseline is `36.3`/`72.4 tok/s/GPU`.
  - qwen8 Temp 0 suffix/Eagle-3 `2136608`/`2136610` completed and are parsed.
  - qwen30 Temp 0 baseline `2136607` completed in `03:46:36`; batch 1/2 parsed at `19.38`/`39.00 tok/s/GPU`.
  - qwen30 Temp 0 suffix `2136609` completed and both batch 1/2 rows are parsed.

## Nemo-RL Online PARD/PARD-2

- Existing qwen8 official online PARD-2 comparison remains the strongest completed evidence:
  - `online_pard2` job `3288183` completed parsed post-step rows.
  - Draft refit ran on 9 steps.
  - Acceptance length improved from static `1.836` to online `2.553`.
  - Throughput did not improve versus baseline: generation-worker TPS was `0.5887x` of baseline.
- qwen32 online PARD2 jobs with target PP>1 failed by design guard:
  - `3338453`: `pipeline_model_parallel_size=4`
  - qwen235B `3337599`: `pipeline_model_parallel_size=8`
- New qwen32 PP=1 retry:
  - `3339651`, `20260616_mathrl_qwen32_online_pard2_pp1_tp4_gate1_r1`
  - Failed after reaching vLLM actor creation.
  - Confirmed final config has target `pipeline_model_parallel_size=1`, target `tensor_model_parallel_size=4`, vLLM `tensor_parallel_size=4`.
  - This passed the previous PP guard and connected to a 4-node Ray cluster.
  - Failure: vLLM `SpeculativeConfig` rejected `speculative_draft_tensor_parallel_size=2`; this vLLM build requires draft TP to be `1` or target TP (`4`).
- qwen32 PP=1 retry with draft TP 1:
  - `3340096`, `20260616_mathrl_qwen32_online_pard2_pp1_tp4_drafttp1_gate1_r2`
  - Failed during vLLM worker initialization after actor venv setup.
  - Failure: vLLM requires `draft_tensor_parallel_size` to match target `tensor_parallel_size`; this run used draft TP `1` with target TP `4`.
- qwen32 PP=1 retry with draft TP 4:
  - `3340497`, `20260616_mathrl_qwen32_online_pard2_pp1_tp4_drafttp4_gate1_r3`
  - Submitted under account `nemotron_n3_post` with target TP `4` and draft TP `4`.
  - Failed after passing the previous draft-TP mismatch point: vLLM accepted the PARD2 `SpeculativeConfig` and initialized the engine.
  - New failure: GRPO ratio assertion requested `loss_fn.force_on_policy_ratio=true` when skipping behavior logprobs. Existing batch sizing already matches `16 prompts * 32 generations = train_global_batch_size 512`.
- qwen32 PP=1 retry with draft TP 4 and force-on-policy ratio:
  - `3340709`, `20260616_mathrl_qwen32_online_pard2_pp1_tp4_drafttp4_forceonpolicy_r4`
  - Submitted under account `nemotron_n3_post`; failed after reaching the first policy-training phase.
  - Added `loss_fn.force_on_policy_ratio=true` and disabled draft CAT weighting for this smoke so the next PARD2 draft-loss step does not require real `prev_logprobs`.
  - The driver connected to Ray, loaded the 950k-sample Math dataset, printed `force_on_policy_ratio enabled`, initialized all 16 vLLM workers, created vLLM with `SpeculativeConfig(method='pard2', num_spec_tokens=3)`, injected `VllmInternalWorkerExtension`, loaded verifier/drafter safetensors shards, selected PARD-2 target layers `(64, 57, 49, 41)`, initialized all 16 Megatron policy workers, and loaded the policy/reference models.
  - It reached `SETUP COMPLETE`, entered `Step 1/1`, exported `312` PARD draft weights, merged refit info from all `16` workers, completed generation for `128/128`, processed rewards, computed logprobs, skipped prev-logprobs due to force-on-policy ratio, and started `Training policy...`.
  - Failure: `ValueError: Cannot infer PARD-2 target hidden-state layout: hidden=(294, 1, 20480) batch=1 sequence=1176`. This indicates sequence-parallel local hidden states were not gathered before PARD2 target feature alignment.
  - Parsed r4 metrics were saved to `docs/oci_hsg_mathrl_qwen32_online_pard2_force_r4_metrics_20260616.csv` and `docs/oci_hsg_mathrl_qwen32_online_pard2_force_r4_summary_20260616.csv`; visible weighted acceptance was about `1.90%`, mean accept len `1.06`.
- qwen32 sequence-parallel gather fix:
  - Patched remote `nemo_rl/models/megatron/draft/pard.py` to gather sequence-parallel hidden states when `local_seq_len * TP == full_sequence_length` before PARD2 target feature alignment; remote `python3 -m py_compile` passed.
  - `3341175`, `...seqpargather_r5`, failed in `00:00:20` from sbatch syntax only: `export COMMAND=bash /path/driver_command.sh` made the path look like an invalid export identifier.
  - `3341220`, `...seqpargather_r6`, fixed the `COMMAND` quoting and reached the deepest point so far under `nemotron_n3_post`.
  - r6 reached `SETUP COMPLETE`, initialized 16/16 vLLM workers and 16/16 Megatron policy workers, exported `312` PARD draft weights, merged refit info from all 16 workers, completed generation for `128/128`, processed rewards, computed logprobs, and entered `Training policy...`.
  - r6 cleared the previous hidden-state gather failure, then failed in the draft embedding hook: `ValueError: Projected PARD-2 target features do not align with draft embeddings: target_feat=(1, 768, 1024) embedding=(192, 1, 1024)`. This is a sequence-parallel local-shard layout mismatch between global PARD2 target features and draft embedding output.
  - Patched remote `nemo_rl/models/megatron/draft/pard.py` again so projected PARD2 target features are sliced by tensor-parallel rank when the draft embedding output is sequence-parallel local `[local_S, B, H]`; remote `python3 -m py_compile` passed.
  - Relaunched qwen32 PP=1/TP=4/draftTP=4/force-on-policy MathRL smoke as `3341685`, `20260616_mathrl_qwen32_online_pard2_pp1_tp4_drafttp4_forceonpolicy_seqpargather_localspfeat_r7`, under `nemotron_n3_post`.
  - r7 reached Ray `16/16` actors and started the driver, but was cancelled after the new actor venv stalled before `READY_ENV_BUILDER`.
  - r8 `3341877`, `20260616_mathrl_qwen32_online_pard2_pp1_tp4_drafttp4_forceonpolicy_seqpargather_localspfeat_reusevenv_r8`, was relaunched with the r6 ready actor-venv suffix and cleared the previous startup/sequence-layout failures.
  - r8 failed after reaching the draft-loss call: `DraftCrossEntropyLossFn.__call__()` did not accept the hard-label PARD/PARD-2 input `teacher_token_ids`. This means the run reached beyond vLLM setup, generation, logprob processing, and into policy training.
  - Patched remote `nemo_rl/algorithms/loss/loss_functions.py` so draft loss supports hard `teacher_token_ids`, optional `draft_token_weights`, and `draft_global_valid_toks`; TP-sharded logits use `DistributedLogprob` for `-log p(target)`. Remote `python3 -m py_compile nemo_rl/algorithms/loss/loss_functions.py` passed.
  - Relaunched qwen32 PP=1/TP=4/draftTP=4/force-on-policy MathRL smoke as r9 `3342483`, `20260616_mathrl_qwen32_online_pard2_pp1_tp4_drafttp4_forceonpolicy_seqpargather_localspfeat_reusevenv_hardce_r9`, under `nemotron_n3_post`. It completed successfully (`COMPLETED 0:0`) at `2026-06-16T10:04:01`.
  - r9 completed `Step 1/1` end to end: setup, PARD2 vLLM generation, reward/advantage processing, logprob computation, hard-label draft CE policy training, metrics logging, and clean early stop at max steps. This clears the previous `teacher_token_ids` draft-loss blocker.
  - Parsed r9 metrics were saved to `docs/oci_hsg_mathrl_qwen32_online_pard2_hardce_r9_metrics_20260616.csv` and `docs/oci_hsg_mathrl_qwen32_online_pard2_hardce_r9_summary_20260616.csv`: total step time `1392.98s`, generation `1298.15s`, draft loss `4.5748`, E2E `25.84 tok/s/GPU`, generation worker `27.73 tok/s/GPU`, weighted acceptance `1.52%`, mean accept len `1.04`.
  - Local patch tree was synchronized with the r9-successful remote state so follow-up submissions do not regress: `remote_patch_pard2_official/nemo_rl/models/megatron/draft/pard.py` now includes sequence-parallel local target-feature slicing, `remote_patch_pard2_official/nemo_rl/algorithms/loss/loss_functions.py` now includes hard-label draft CE support, and the multimodel submit script now stages that loss file.
  - To test whether online PARD-2 keeps working beyond a one-step smoke, qwen32 PP=1/TP=4/draftTP=4/force-on-policy MathRL was relaunched as 5-step r10 `3344974`, `20260616_mathrl_qwen32_online_pard2_pp1_tp4_drafttp4_forceonpolicy_seqpargather_localspfeat_reusevenv_hardce_step5_r10`, under `nemotron_n3_post`. Dry-run validation confirmed `grpo.max_num_steps=5`, target/generation/draft TP `4`, `loss_fn.force_on_policy_ratio=true`, `policy.draft.enabled=true`, `policy.draft.loss=pard2`, train/refit interval `1`, and `policy.draft.cat_weighting=false`.
  - r10 reached the deepest multi-step gate so far but failed before Step 2. It connected to the external Ray cluster, initialized all `16/16` vLLM and `16/16` Megatron policy workers, created vLLM engines with `SpeculativeConfig(method='pard2', num_spec_tokens=3)`, warmed engines, exported and merged `312` PARD draft weights, reached `SETUP COMPLETE` after `1080.0s`, completed Step 1/5 generation for `128/128` prompts, processed rewards, computed logprobs, and entered `Training policy...`.
  - r10 failed after `00:44:15` with `ray.exceptions.RayTaskError(NameError)` in `MegatronPolicyWorker.train()`: `get_tensor_model_parallel_rank` was used in `nemo_rl/models/megatron/draft/pard.py` by PARD2 target-feature alignment but was not imported. This is an import regression in the patched sequence-parallel alignment helper, not a new Ray/vLLM setup failure.
  - Patched local `remote_patch_pard2_official/nemo_rl/models/megatron/draft/pard.py` and the OCI remote checkout to import `get_tensor_model_parallel_rank`; both local and remote `python3 -m py_compile` checks passed.
  - Relaunched the same qwen32 online PARD2 5-step MathRL follow-up as r11 `3345352`, `20260616_mathrl_qwen32_online_pard2_pp1_tp4_drafttp4_forceonpolicy_seqpargather_localspfeat_reusevenv_hardce_step5_rankimport_r11`, under `nemotron_n3_post`. Latest `sacct` at `2026-06-16 20:51 PDT`: r11 is `COMPLETED 0:0` after `02:14:28` on `nvl72034-T[09,11-12,16]`. It completed all `5/5` MathRL online PARD-2 steps end to end. Step metrics:
    - Step 1: total `1511.62s`, generation `1409.76s`, E2E `23.81 tok/s/GPU`, generation worker `25.53 tok/s/GPU`, weighted acceptance `2.06%`, mean accept len `1.06`.
    - Step 2: total `1495.65s`, generation `1403.96s`, E2E `23.97 tok/s/GPU`, generation worker `25.54 tok/s/GPU`, weighted acceptance `1.93%`, mean accept len `1.06`.
    - Step 3: total `1521.55s`, generation `1428.07s`, E2E `23.86 tok/s/GPU`, generation worker `25.43 tok/s/GPU`, weighted acceptance `1.33%`, mean accept len `1.04`.
    - Step 4: total `1500.57s`, generation `1406.57s`, E2E `24.01 tok/s/GPU`, generation worker `25.62 tok/s/GPU`, weighted acceptance `1.78%`, mean accept len `1.05`.
    - Step 5: total `1503.19s`, generation `1409.38s`, E2E `24.06 tok/s/GPU`, generation worker `25.66 tok/s/GPU`, weighted acceptance `1.53%`, mean accept len `1.05`.
  - r11 5-step mean: total `1506.52s`, generation `1411.55s`, E2E `23.94 tok/s/GPU`, generation worker `25.56 tok/s/GPU`, draft loss `4.6821`, avg reward `0.2871`, weighted acceptance `1.72%`, mean accept len `1.05`. No Traceback/RayTaskError/EngineDead and no `get_tensor_model_parallel_rank` failure appeared.
- qwen235B MathRL step20 status from OCI-HSG logs/sacct at `2026-06-16 21:26 PDT`:
  - baseline `3334220` failed after `01:14:56`; it completed `8/20` steps, then hit Megatron NCCL watchdog timeouts during Step 9 policy training. Partial metrics are saved in `docs/oci_hsg_mathrl_qwen235b_baseline_step20_3334220_partial_summary_20260616.csv`.
  - suffix K32 `3333717` failed after `01:24:00`; it completed `14/20` steps, then hit Megatron NCCL watchdog timeouts during Step 15 policy training. Partial metrics are saved in `docs/oci_hsg_mathrl_qwen235b_suffix_step20_3333717_partial_summary_20260616.csv`.
  - PARD K5 `3333535` failed after `00:22:00`; root cause is vLLM `wake_up()` CuMem allocator OOM during policy preparation: `CUDA Error: out of memory at /workspace/csrc/cumem_allocator.cpp:139`.
  - Eagle-3 `3333537` failed after `01:21:59`; it completed `14/20` steps and died during Step 15 policy training from Megatron NCCL watchdog timeouts. Partial metrics are saved in `docs/oci_hsg_mathrl_qwen235b_eagle3_step20_3333537_partial_summary_20260616.csv`.
- qwen235B main fixed256 gate:
  - baseline `3342356` completed after `00:19:31`.
  - PARD K3 `3342358` failed after `00:33:17`; it reached refit-info merge (`workers=128 keys=36945 draft_keys=0`) and then `VllmGenerationWorker.prepare_refit_info()` failed with `RayTaskError(ActorDiedError)`. `sacct` now shows child steps `.5` and `.8` as `OUT_OF_MEMORY 0:125`, so this is consistent with worker OOM.
- Failure triage rows were saved to `docs/oci_hsg_failed_job_triage_20260616.csv` and `docs/oci_hsg_failed_job_triage_20260616.md`.

## SWE-RL Qwen3-30B-A3B

- Fixed Ray reuse logic in OCI remote `nemo_rl/distributed/virtual_cluster.py`:
  - If `RAY_ADDRESS` is set, reuse the externally managed ray.sub cluster instead of starting a local driver-node-only cluster.
- Failed/cancelled attempts:
  - `3339062`: Ray `resources` error while connecting to existing cluster.
  - `3339063`: moved past that, then failed with generation placement group needing 16 GPUs while only 4 were visible after local-only Ray init.
  - `3339569`: cancelled after it bootstrapped with Python/Ray `3.12.13/2.49.2` and stayed at actor count `0/32`.
- py313/Ray 2.55 and external Ray reuse patch reached the SWE rollout path:
  - baseline `3339703` and suffix `3339704` both connected to the external Ray cluster, loaded the SWE config, completed data setup, installed actor deps, and reached `AsyncTrajectoryCollector`.
  - Both then looped on `Top k is not supported in the generation config in NeMo-Gym path!`. The config uses `top_k: null`, but the vLLM path normalizes disabled top-k to `-1`; NeMo-Gym's assertion used `assert not generation_config["top_k"]`, so `-1` failed even though it means disabled filtering.
  - Patched remote `nemo_rl/experience/rollouts.py` to allow disabled top-k sentinels `(None, 0, -1)` while still rejecting real top-k filtering; remote `python3 -m py_compile` passed.
  - Cancelled stuck r5 jobs and relaunched same Qwen3-30B-A3B SWE step1 smoke geometry (`TP=4`, `CP=1`, `EP=16`, `VLLM_TP=1`, `SEQLEN=8192`, `TOTAL_NODES=8`, `GBS=64`, `max_steps=1`) under `nemotron_n3_post`:
    - baseline `3341240`, `20260616_oci_hsg_swerl_qwen30ba3b_baseline_step1_smoke_topkfix_r6`
    - suffix `3341241`, `20260616_oci_hsg_swerl_qwen30ba3b_suffix_step1_smoke_topkfix_r6`
  - r6 did not reach the SWE rollout path. It repeatedly tried `/tmp/nemo_rl_ray_<job>_3.12.13_2.49.2/bin/ray` and stayed at `0/32` actors because the Ray/Python version override from r5 was missing.
  - Cancelled r6 and relaunched r7 with the r5-compatible Ray bootstrap values `RAY_PYTHON_VERSION=3.13.13`, `RAY_PYTHON_SPEC=3.13.13`, `RAY_VERSION=2.55.1`, and `UV_PYTHON=3.13.13`:
    - baseline `3341315`, `20260616_oci_hsg_swerl_qwen30ba3b_baseline_step1_smoke_topkfix_py313_r7`
    - suffix `3341316`, `20260616_oci_hsg_swerl_qwen30ba3b_suffix_step1_smoke_topkfix_py313_r7`
  - r7 was cancelled after a Ray readiness race: `STARTED_RAY_HEAD` was touched before `/tmp/nemo_rl_ray_<job>_3.13.13_2.55.1/bin/ray` existed, so the launcher repeatedly polled a non-existent Ray CLI and stayed at actor count `0/32`.
  - Patched remote `ray.sub` so `STARTED_RAY_HEAD` is created only after the Ray CLI exists and is executable; remote `bash -n` passed.
  - r8 jobs under `nemotron_n3_post`:
    - baseline `3341457`, `20260616_oci_hsg_swerl_qwen30ba3b_baseline_step1_smoke_rayready_py313_r8`
    - suffix `3341458`, `20260616_oci_hsg_swerl_qwen30ba3b_suffix_step1_smoke_rayready_py313_r8`
  - r8 cleared the Ray readiness race: `STARTED_RAY_HEAD` and `ray-driver.log` were present for both jobs, and actor counts reached `32/32` with `All workers connected!` in both baseline and suffix slurm logs.
  - r8 then failed because the driver could not attach to the external `RAY_ADDRESS` cluster, fell back to starting local Ray on the driver node, and only saw 4 GPUs. Both baseline and suffix hit `Not enough GPUs available. Requested 16 GPUs, but only 4 are available in the cluster`, then `ResourceInsufficientError: Maximum number of retries reached (6)`.
  - Patched OCI remote `nemo_rl/distributed/virtual_cluster.py` so `RAY_ADDRESS` is strict: it retries the external cluster attach, passes `_node_ip_address` when available, and refuses to start local Ray if external attach fails. Remote `python3 -m py_compile` passed; local patch bundle was updated as well.
  - r9 jobs were relaunched with the same Qwen3-30B-A3B SWE step1 smoke geometry (`TP=4`, `CP=1`, `EP=16`, `VLLM_TP=1`, `SEQLEN=8192`, `TOTAL_NODES=8`, `GBS=64`, `max_steps=1`) under `nemotron_n3_post`:
    - baseline `3341660`, `20260616_oci_hsg_swerl_qwen30ba3b_baseline_step1_smoke_raystrict_py313_r9`
    - suffix `3341661`, `20260616_oci_hsg_swerl_qwen30ba3b_suffix_step1_smoke_raystrict_py313_r9`
  - Latest r9 status: baseline `3341660` and suffix `3341661` both failed after external Ray startup. The strict `RAY_ADDRESS` guard prevented an incorrect driver-node-only fallback, but the driver could not read `temp_dir` from GCS and refused to start local Ray. This is now a Ray/GCS attach timing issue, not the previous top-k or local-Ray GPU visibility failure.
  - r10 Ray-head readiness patch:
    - Inspected r9 `ray-head.log`/`ray-worker-1.log`: Ray CLI existed before `STARTED_RAY_HEAD`, but workers and the driver could still race GCS readiness; driver attempts repeatedly failed with `Could not read 'temp_dir' from GCS`.
    - Patched OCI remote `ray.sub` so `STARTED_RAY_HEAD` is touched only after `ray status --address "$ip_head"` succeeds from inside the head container. Remote `bash -n ray.sub` passed.
    - Re-submitted the same Qwen3-30B-A3B SWE step1 smoke geometry (`HYBRIDEP=0`, `TP=4`, `CP=1`, `EP=16`, `VLLM_TP=1`, `GEN_TRAIN_RATIO=1`, `SEQLEN=8192`, `TOTAL_NODES=8`, `GBS=64`, `max_steps=1`) under `nemotron_n3_post`:
      - baseline `3343371`, `20260616_oci_hsg_swerl_qwen30ba3b_baseline_step1_smoke_rayheadready_py313_r10`
      - suffix `3343372`, `20260616_oci_hsg_swerl_qwen30ba3b_suffix_step1_smoke_rayheadready_py313_r10`
    - r10 baseline `3343371` failed after `00:19:05`. The Ray head and workers came up and the cluster reached `32/32` actors with `All workers connected!`, but the driver failed every external attach attempt with `ConnectionError: Could not read 'temp_dir' from GCS`. The job used a Ray 2.55.1 head venv while the Nemo-RL `uv.lock` driver environment resolves Ray 2.54.0, so the next retry pins the head to Ray 2.54.0 to match the driver lock.
    - r10 suffix `3343372` was still pending and was cancelled before it could repeat the same Ray attach failure.
    - r11 was submitted under `nemotron_n3_post` with the same Qwen3-30B-A3B SWE step1 smoke geometry (`HYBRIDEP=0`, `TP=4`, `CP=1`, `EP=16`, `VLLM_TP=1`, `GEN_TRAIN_RATIO=1`, `SEQLEN=8192`, `TOTAL_NODES=8`, `GBS=64`, `max_steps=1`) but with `RAY_VERSION=2.54.0`, `RAY_PYTHON_VERSION=3.13.13`, and `RAY_PYTHON_SPEC=3.13.13`:
      - baseline `3343655`, `20260616_oci_hsg_swerl_qwen30ba3b_baseline_step1_smoke_ray254_py313_r11`
      - suffix `3343656`, `20260616_oci_hsg_swerl_qwen30ba3b_suffix_step1_smoke_ray254_py313_r11`
    - r11 baseline `3343655` failed after `00:04:39` from Ray head startup timeout on the head node before the full cluster became ready: `The current node timed out during startup`. This is different from the r10 driver attach failure.
    - r11 suffix `3343656` moved beyond setup and entered the actual SWE rollout path. It reached `SETUP COMPLETE`, initialized vLLM/Megatron workers, started Nemo-Gym, collected rollouts to `32/32`, and entered `Step 1/1`.
    - r11 then stalled in replay-buffer sampling because one rollout group returned no generation data. The concrete vLLM error was `You passed 8193 input tokens and requested 0 output tokens. However, the model's context length is only 8192 tokens`. Nemo-Gym surfaced this as `NeMo Gym returned a result with no generation data`, with the explicit suggested fix to increase `policy.max_total_sequence_length` and `policy.generation.vllm_cfg.max_model_len`.
    - r11 suffix was also only a plumbing smoke despite the suffix label: its Hydra overrides had no `policy.generation.vllm_kwargs.speculative_config.*`, and vLLM initialized with `speculative_config=None`.
    - r12 baseline `3344309`, `20260616_oci_hsg_swerl_qwen30ba3b_baseline_step1_smoke_ray254_startwait_py313_r12`, was submitted with the same `SEQLEN=8192` geometry plus `RAY_raylet_start_wait_time_s=180` and `RAY_HEAD_READY_ATTEMPTS=240`; it is expected to hit the same context limit if it starts.
    - r13 32K retries were submitted under `nemotron_n3_post` with the same 8-node Qwen3-30B-A3B SWE smoke geometry but `SEQLEN=32768`, Ray `2.54.0`, Python `3.13.13`, and the Ray start-wait overrides:
      - baseline `3344379`, `20260616_oci_hsg_swerl_qwen30ba3b_baseline_step1_smoke_ctx32k_ray254_py313_r13`
      - suffix specdec `3344390`, `20260616_oci_hsg_swerl_qwen30ba3b_suffix_step1_smoke_ctx32k_specdec_ray254_py313_r13`, with explicit `speculative_config.method=suffix`, `num_speculative_tokens=32`, and the suffix tree/cache knobs.
    - Immediate `squeue` polling timed out during SLURM controller delay, but both r13 `sbatch` calls returned job ids and the tracker was updated.
    - Tracker updated: `latest_oci_hsg_swerl_qwen30ba3b_step1_smoke_20260616_jobs.csv`.
    - r13 baseline `3344379` reached `Step 1/1`, started SWE rollout collection, reached at least `13/32` rollouts, and reported policy training telemetry (`Training Worker Group` about `1733 tok/s/GPU`). It then failed at the 32K context boundary: vLLM rejected a prompt with `32769` input tokens against `max_model_len=32768`, followed by `EngineDeadError` and Nemo-Gym empty-generation failures.
    - r13 suffix `3344390` was still pending with the same 32K context setting, so it was cancelled before starting.
    - r14 context-margin retries were submitted under `nemotron_n3_post` with the same Qwen3-30B-A3B 8-node smoke geometry, `SEQLEN=40960`, explicit `policy.generation.vllm_cfg.max_model_len=40960`, and `max_num_batched_tokens=65536`:
      - baseline `3344823`, `20260616_oci_hsg_swerl_qwen30ba3b_baseline_step1_smoke_ctx40k_vllm40k_ray254_py313_r14`
      - suffix K32 `3344824`, `20260616_oci_hsg_swerl_qwen30ba3b_suffix_step1_smoke_ctx40k_vllm40k_specdec_ray254_py313_r14`
    - Follow-up audit found the r14 suffix submission inherited the suffix launcher env but did not explicitly append the `policy.generation.vllm_kwargs.speculative_config.*` Hydra overrides. Since it was still pending, `3344824` was cancelled before using GPUs.
    - r15 suffix K32 replacement `3344863`, `20260616_oci_hsg_swerl_qwen30ba3b_suffix_step1_smoke_ctx40k_vllm40k_specdec_k32_ray254_py313_r15`, was submitted with explicit `speculative_config.method=suffix`, `num_speculative_tokens=32`, `suffix_decoding_max_tree_depth=24`, `suffix_decoding_max_cached_requests=10000`, `suffix_decoding_max_spec_factor=1.0`, and `suffix_decoding_min_token_prob=0.1`, plus the same 40K context overrides.
    - Latest OCI-HSG poll at about `2026-06-16 21:00 PDT`: baseline ctx40k `3344823` completed after `00:26:47`; suffix K32 `3344863` failed after `00:11:11`.
    - r15 suffix failed before rollout because vLLM suffix decoding validates that `arctic-inference==0.1.1` is installed in the actor venv. The log shows `ImportError: Arctic Inference is required for suffix decoding. Install via pip install arctic-inference==0.1.1`.
    - r16 suffix K32 `3351394`, `20260616_oci_hsg_swerl_qwen30ba3b_suffix_step1_smoke_ctx40k_vllm40k_specdec_k32_arcticpy313_ray254_py313_r16`, was submitted with the `arcticpy313_r16` actor venv suffix. Last copied logs showed Ray actors reached `64/64`, `All workers connected!`, and no `arctic-inference` ImportError before local DNS blocked further polling.
- Older Lyris Qwen3-30B-A3B integrated step10 matrix attempts are not reusable:
  - `2109933`/`2109990` suffix, `2109934`/`2109991` PARD, and `2109935`/`2109992` Eagle-3 all failed in an enroot step.

## 2026-06-16 23:01 PDT Timeout/Partial-Log Handling Update

- Hardened the MathRL and SWE-RL submit/poll helpers so SSH connect timeout and remote command timeout are separate. This prevents a slow log poll from being controlled only by the short SSH connect timeout.
- Poll commands now read at most a small number of recent log files and split bounded `tail` output into labelled `[tail-chunk]` blocks. Current dry-run used `--tail-lines 40 --log-chunk-lines 20 --max-log-files 2 --remote-log-timeout 10`; default next-window contract uses `--tail-lines 80 --log-chunk-lines 20 --max-log-files 3`.
- Timeout output is written to reusable stdout/stderr files immediately, and status CSV/Markdown is flushed per selected row. Contract checks now explicitly verify timeout stdout/stderr persistence for both MathRL and SWE-RL helpers.
- The integrated next-access-window driver now forwards the chunk/timeout controls to both helpers. Latest local dry-run finished all six steps with summary `driver_child_remote_unreachable=5, driver_step_ok=1`; remote preflight is still DNS-blocked from this local session, but the timeout/partial-output path is validated.
- Dashboard regenerated and opened: `docs/specdec_benchmark_metrics_dashboard_20260616.html`. It now shows the integrated driver contract and the timeout/log chunk guarantees.

## 2026-06-16 23:06 PDT Timeout Command Output Parser

- Added `scripts/refresh_specdec_timeout_command_outputs.py` so saved submit/poll stdout logs are scanned into `docs/specdec_timeout_command_output_artifacts_20260616.csv` and `docs/specdec_timeout_command_output_step_index_20260616.csv` on every dashboard rebuild.
- The parser reuses the existing NeMo-RL step parser; if a timeout stdout contains complete `Step N/M` blocks, completed-step metrics become immediately usable without waiting for a later full-log rsync.
- Current dashboard scan found `24` saved stdout logs and `0` parsed step rows because the latest local run was DNS-blocked dry-run output. A smoke test using the existing `3334220` MathRL ray-driver log copied as `.stdout.log` parsed `9` step rows and recovered job id `3334220`, proving the command-output path can consume real timeout log chunks.

## 2026-06-16 23:08 PDT Objective Evidence/Queue Update

- Updated `docs/specdec_goal_evidence_matrix_20260616.csv` so the top-level goal evidence now explicitly includes current execution readiness: `remote DNS blocked`, with child preflight counts from `docs/specdec_goal_next_access_window_status_20260616.csv` (`driver_child_remote_unreachable=5`, `driver_step_ok=1`).
- Added the saved stdout parser to the objective evidence matrix: `scripts/refresh_specdec_timeout_command_outputs.py` is recorded as a proven parser that is waiting for real remote tail chunks.
- Updated `docs/specdec_next_experiment_queue_20260616.csv` with a P0 `preserve_timeout_command_outputs` row. The queue now says to rebuild the dashboard or run the timeout-output refresh after any submit/poll timeout, using `docs/specdec_timeout_command_output_artifacts_20260616.csv` and `docs/specdec_timeout_command_output_step_index_20260616.csv`.
- Dashboard regenerated: `docs/specdec_benchmark_metrics_dashboard_20260616.html` now shows the remote DNS blocker and the timeout-output reuse path directly in the Goal Evidence Matrix and Next Experiment Queue.

## 2026-06-16 23:11 PDT Strict Goal Completion Audit

- Added `scripts/build_specdec_goal_completion_audit.py`, producing `docs/specdec_goal_completion_audit_20260616.csv` and `docs/specdec_goal_completion_audit_20260616.md`.
- Current strict audit summary: `complete=3`, `partial=3`, `incomplete=2`, `blocked_by_external_access=1`.
- Complete rows are implementation contracts, completed qwen32 online PARD-2 functional proof, and timeout/failed-log preservation. Incomplete rows remain online PARD functional proof (`R3`) and integrated SWE-RL PARD/PARD-2 proof beyond baseline (`R5`). Partial rows cover MathRL static SpecDec, online-training performance impact, and standalone Math/SWE vLLM benchmark coverage.
- Dashboard regenerated: `docs/specdec_benchmark_metrics_dashboard_20260616.html` now includes the `Goal Completion Audit` section directly below the Goal Evidence Matrix.

## 2026-06-16 23:14 PDT Timeout Output Immediate Refresh

- Tightened timeout log chunking so both MathRL and SWE-RL submit/poll helpers default to `--log-chunk-lines 20`; the integrated next-access-window driver now forwards `--tail-lines 80 --log-chunk-lines 20 --max-log-files 3`.
- Updated `scripts/run_specdec_goal_next_access_window.py` to run `scripts/refresh_specdec_timeout_command_outputs.py` immediately after each child command stdout/stderr file is written. This means a timeout that still prints complete `Step N/M` blocks is indexed into `docs/specdec_timeout_command_output_step_index_20260616.csv` before the next step runs, not only during the final dashboard rebuild.
- Latest dry-run status has `timeout_refresh_state=refresh_ok` for all six driver steps. The current parser scan sees `30` saved stdout artifacts and `0` parsed step rows because this local run was DNS-blocked before any remote log tail could be read.
- Contract regenerated with the new guarantees: `docs/specdec_goal_next_access_window_contract_20260616.json` / `.md` now pass `64/64` checks, including the timeout refresh fields.

## 2026-06-16 23:18 PDT Remote Access Host Audit

- Rechecked local SLURM mode and SSH access: `squeue` is still unavailable locally, so commands must go through SSH.
- Added `scripts/audit_specdec_remote_access_hosts.py`, producing `docs/specdec_remote_access_host_audit_20260616.csv` and `docs/specdec_remote_access_host_audit_20260616.md`.
- Current audit covers 14 configured/candidate OCI-HSG and Lyris hosts, including `oci-hsg-cs-001-vscode-01/02/03`, `oci-hsg-cs-001-vscode-02.nvidia.com`, `login-lyris`, `login-lyris.nvidia.com`, `lyris`, and `lyris.nvidia.com`.
- Result: all 14 candidates are still `failed_dns`; no ControlMaster socket is available for the configured OCI-HSG/Lyris aliases, and no batch SSH connection reached auth or SLURM.
- Dashboard regenerated with a `Remote Access Host Audit` section: `docs/specdec_benchmark_metrics_dashboard_20260616.html`.

## 2026-06-16 23:21 PDT Remote Host Override Readiness

- Added runtime host override support to both active-goal submit helpers and the integrated driver. The helpers now preserve `manifest_remote_host` while using an overridden `remote_host` for preflight/execution.
- `scripts/submit_mathrl_qwen32_online_pard_gate_from_manifest.py` supports `--remote-host-override`, `SPECDEC_REMOTE_HOST_OVERRIDE`, or `SPECDEC_OCI_HOST`; for MathRL online launch rows it also rewrites the leading `ssh <old-host> ...` inside the manifest command.
- `scripts/submit_swerl_qwen30_specdec_from_manifest.py` supports the same override for preflight and the SSH wrapper used for poll/launch rows.
- `scripts/run_specdec_goal_next_access_window.py` now forwards `--remote-host-override`, plus method-specific `--math-remote-host-override` and `--swerl-remote-host-override`.
- Isolated dry-runs under `tmp/override_*` verified that `manifest_remote_host=oci-hsg-cs-001-vscode-02` and `remote_host=replacement-oci-host` are both recorded, and that the MathRL online PARD command starts with `ssh replacement-oci-host ...`.
- Updated `docs/specdec_goal_next_access_window_runbook_20260616.md` with override examples. The next executable command when a reachable alias is available is:
  `python3 scripts/run_specdec_goal_next_access_window.py --execute --scope all --math-phase all --swerl-phase all --keep-going --remote-host-override <reachable-oci-host>`.

## 2026-06-16 23:25 PDT Remote-Ready Auto Driver

- Extended `scripts/audit_specdec_remote_access_hosts.py` so successful SSH probes also record `remote_hostname`, `remote_user`, `remote_squeue_state`, and `remote_squeue_path`. This prevents treating a non-SLURM endpoint as launch-ready.
- Added `scripts/run_specdec_goal_when_remote_ready.py`. It runs the host audit, selects the first host matching the OCI/HSG/vscode pattern with `connect_state=ok` and `remote_squeue_state=ok`, then invokes `scripts/run_specdec_goal_next_access_window.py` with `--remote-host-override <selected-host>`.
- Current wrapper status is `no_ready_host` with `ready_host_count=0`, as expected from the current DNS-blocked local session. Artifacts: `docs/specdec_goal_remote_ready_driver_status_20260616.csv` and `.md`.
- The command to use once access is expected to work is now:
  `python3 scripts/run_specdec_goal_when_remote_ready.py --execute --scope all --math-phase all --swerl-phase all --keep-going`.
