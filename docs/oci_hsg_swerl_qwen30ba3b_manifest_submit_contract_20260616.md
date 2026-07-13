# SWE-RL Qwen30 Manifest Submit Contract

Overall: **FAIL**

Manifest: `/Users/sna/Nemo-RL_Qwen3_Roadmap/docs/oci_hsg_swerl_qwen30ba3b_specdec_launch_manifest_20260616.csv`
Submit helper: `/Users/sna/Nemo-RL_Qwen3_Roadmap/scripts/submit_swerl_qwen30_specdec_from_manifest.py`

| check | status | detail |
| --- | --- | --- |
| manifest has six qwen30 rows | PASS | `6` |
| submit status includes stdout_path | PASS | `stdout_path` |
| submit status includes stderr_path | PASS | `stderr_path` |
| submit helper exposes command output directory | PASS | `/Users/sna/Nemo-RL_Qwen3_Roadmap/docs/oci_hsg_swerl_qwen30ba3b_specdec_manifest_command_outputs_20260616` |
| submit helper flushes status after each row | PASS | `True` |
| manifest method/action coverage | PASS | `[('poll_existing', 'suffix', 'false'), ('launch_static', 'eagle3', 'false'), ('launch_static', 'pard', 'false'), ('launch_static', 'pard2', 'false'), ('launch_online', 'pard', 'true'), ('launch_online', 'pard2', 'true')]` |
| poll_existing:suffix:false helper validation | PASS | `ok` |
| poll_existing:suffix:false dry-run command bash -n | PASS | `` |
| poll_existing:suffix:false execute command bash -n | PASS | `` |
| poll_existing:suffix:false polls suffix r16 job | PASS | `3351394` |
| poll_existing:suffix:false checks squeue | PASS | `squeue -j 3351394` |
| poll_existing:suffix:false checks sacct | PASS | `sacct -X -j 3351394` |
| poll_existing:suffix:false tails bounded log chunk | PASS | `tail -n 80` |
| poll_existing:suffix:false emits reusable tail chunk markers | PASS | `[tail-chunk] lines` |
| poll_existing:suffix:false limits recent log files | PASS | `tail -n 3` |
| poll_existing:suffix:false bounds remote log probes | PASS | `timeout 20s` |
| launch_static:eagle3:false helper validation | PASS | `ok; stages patched non-suffix launcher` |
| launch_static:eagle3:false dry-run command bash -n | PASS | `` |
| launch_static:eagle3:false execute command bash -n | PASS | `` |
| launch_static:eagle3:false dry-run is non-submitting | PASS | `DRY_RUN=true` |
| launch_static:eagle3:false dry-run avoids ambiguous truthy flag | PASS | `not present: DRY_RUN=1` |
| launch_static:eagle3:false execute enables submission | PASS | `DRY_RUN=false` |
| launch_static:eagle3:false execute does not stay dry-run | PASS | `not present: DRY_RUN=true` |
| launch_static:eagle3:false keeps step1 smoke | PASS | `MAX_STEPS=1` |
| launch_static:eagle3:false keeps ctx40k | PASS | `MAX_TOTAL_SEQUENCE_LENGTH=40960` |
| launch_static:eagle3:false uses RL temperature | PASS | `GENERATION_TEMPERATURE=1.0` |
| launch_static:eagle3:false uses RL top_p | PASS | `GENERATION_TOP_P=1.0` |
| launch_static:eagle3:false uses requested account | PASS | `SBATCH_ACCOUNT=nemotron_n3_post` |
| launch_static:eagle3:false stages patched non-suffix launcher | PASS | `PY_PATCH_SWERL_SPECDEC_LAUNCHER` |
| launch_static:eagle3:false patched gate admits non-suffix methods | PASS | `ngram/draft_model/pard2/eagle3` |
| launch_static:eagle3:false refuses unknown launcher shape | FAIL | `missing token: unsupported launcher gate shape; refusing unsafe non-suffix launch` |
| launch_static:eagle3:false disables online policy draft | PASS | `policy.draft.enabled=false` |
| launch_static:eagle3:false uses eagle3 specdec | PASS | `speculative_config.method=eagle3` |
| launch_static:pard:false helper validation | PASS | `ok; stages patched non-suffix launcher` |
| launch_static:pard:false dry-run command bash -n | PASS | `` |
| launch_static:pard:false execute command bash -n | PASS | `` |
| launch_static:pard:false dry-run is non-submitting | PASS | `DRY_RUN=true` |
| launch_static:pard:false dry-run avoids ambiguous truthy flag | PASS | `not present: DRY_RUN=1` |
| launch_static:pard:false execute enables submission | PASS | `DRY_RUN=false` |
| launch_static:pard:false execute does not stay dry-run | PASS | `not present: DRY_RUN=true` |
| launch_static:pard:false keeps step1 smoke | PASS | `MAX_STEPS=1` |
| launch_static:pard:false keeps ctx40k | PASS | `MAX_TOTAL_SEQUENCE_LENGTH=40960` |
| launch_static:pard:false uses RL temperature | PASS | `GENERATION_TEMPERATURE=1.0` |
| launch_static:pard:false uses RL top_p | PASS | `GENERATION_TOP_P=1.0` |
| launch_static:pard:false uses requested account | PASS | `SBATCH_ACCOUNT=nemotron_n3_post` |
| launch_static:pard:false stages patched non-suffix launcher | PASS | `PY_PATCH_SWERL_SPECDEC_LAUNCHER` |
| launch_static:pard:false patched gate admits non-suffix methods | PASS | `ngram/draft_model/pard2/eagle3` |
| launch_static:pard:false refuses unknown launcher shape | FAIL | `missing token: unsupported launcher gate shape; refusing unsafe non-suffix launch` |
| launch_static:pard:false disables online policy draft | PASS | `policy.draft.enabled=false` |
| launch_static:pard:false uses PARD draft_model specdec | PASS | `speculative_config.method=draft_model` |
| launch_static:pard:false enables PARD parallel drafting | PASS | `parallel_drafting=true` |
| launch_static:pard2:false helper validation | PASS | `ok; stages patched non-suffix launcher` |
| launch_static:pard2:false dry-run command bash -n | PASS | `` |
| launch_static:pard2:false execute command bash -n | PASS | `` |
| launch_static:pard2:false dry-run is non-submitting | PASS | `DRY_RUN=true` |
| launch_static:pard2:false dry-run avoids ambiguous truthy flag | PASS | `not present: DRY_RUN=1` |
| launch_static:pard2:false execute enables submission | PASS | `DRY_RUN=false` |
| launch_static:pard2:false execute does not stay dry-run | PASS | `not present: DRY_RUN=true` |
| launch_static:pard2:false keeps step1 smoke | PASS | `MAX_STEPS=1` |
| launch_static:pard2:false keeps ctx40k | PASS | `MAX_TOTAL_SEQUENCE_LENGTH=40960` |
| launch_static:pard2:false uses RL temperature | PASS | `GENERATION_TEMPERATURE=1.0` |
| launch_static:pard2:false uses RL top_p | PASS | `GENERATION_TOP_P=1.0` |
| launch_static:pard2:false uses requested account | PASS | `SBATCH_ACCOUNT=nemotron_n3_post` |
| launch_static:pard2:false stages patched non-suffix launcher | PASS | `PY_PATCH_SWERL_SPECDEC_LAUNCHER` |
| launch_static:pard2:false patched gate admits non-suffix methods | PASS | `ngram/draft_model/pard2/eagle3` |
| launch_static:pard2:false refuses unknown launcher shape | FAIL | `missing token: unsupported launcher gate shape; refusing unsafe non-suffix launch` |
| launch_static:pard2:false disables online policy draft | PASS | `policy.draft.enabled=false` |
| launch_static:pard2:false uses PARD-2 specdec | PASS | `speculative_config.method=pard2` |
| launch_static:pard2:false exports PARD-2 vLLM site | PASS | `SOURCE_VLLM_SITE=` |
| launch_online:pard:true helper validation | PASS | `ok; stages patched non-suffix launcher` |
| launch_online:pard:true dry-run command bash -n | PASS | `` |
| launch_online:pard:true execute command bash -n | PASS | `` |
| launch_online:pard:true dry-run is non-submitting | PASS | `DRY_RUN=true` |
| launch_online:pard:true dry-run avoids ambiguous truthy flag | PASS | `not present: DRY_RUN=1` |
| launch_online:pard:true execute enables submission | PASS | `DRY_RUN=false` |
| launch_online:pard:true execute does not stay dry-run | PASS | `not present: DRY_RUN=true` |
| launch_online:pard:true keeps step1 smoke | PASS | `MAX_STEPS=1` |
| launch_online:pard:true keeps ctx40k | PASS | `MAX_TOTAL_SEQUENCE_LENGTH=40960` |
| launch_online:pard:true uses RL temperature | PASS | `GENERATION_TEMPERATURE=1.0` |
| launch_online:pard:true uses RL top_p | PASS | `GENERATION_TOP_P=1.0` |
| launch_online:pard:true uses requested account | PASS | `SBATCH_ACCOUNT=nemotron_n3_post` |
| launch_online:pard:true stages patched non-suffix launcher | PASS | `PY_PATCH_SWERL_SPECDEC_LAUNCHER` |
| launch_online:pard:true patched gate admits non-suffix methods | PASS | `ngram/draft_model/pard2/eagle3` |
| launch_online:pard:true refuses unknown launcher shape | FAIL | `missing token: unsupported launcher gate shape; refusing unsafe non-suffix launch` |
| launch_online:pard:true enables online policy draft | PASS | `policy.draft.enabled=true` |
| launch_online:pard:true exports online PARD flag | PASS | `PARD_ONLINE_TRAINING=true` |
| launch_online:pard:true uses PARD draft_model specdec | PASS | `speculative_config.method=draft_model` |
| launch_online:pard:true enables PARD parallel drafting | PASS | `parallel_drafting=true` |
| launch_online:pard2:true helper validation | PASS | `ok; stages patched non-suffix launcher` |
| launch_online:pard2:true dry-run command bash -n | PASS | `` |
| launch_online:pard2:true execute command bash -n | PASS | `` |
| launch_online:pard2:true dry-run is non-submitting | PASS | `DRY_RUN=true` |
| launch_online:pard2:true dry-run avoids ambiguous truthy flag | PASS | `not present: DRY_RUN=1` |
| launch_online:pard2:true execute enables submission | PASS | `DRY_RUN=false` |
| launch_online:pard2:true execute does not stay dry-run | PASS | `not present: DRY_RUN=true` |
| launch_online:pard2:true keeps step1 smoke | PASS | `MAX_STEPS=1` |
| launch_online:pard2:true keeps ctx40k | PASS | `MAX_TOTAL_SEQUENCE_LENGTH=40960` |
| launch_online:pard2:true uses RL temperature | PASS | `GENERATION_TEMPERATURE=1.0` |
| launch_online:pard2:true uses RL top_p | PASS | `GENERATION_TOP_P=1.0` |
| launch_online:pard2:true uses requested account | PASS | `SBATCH_ACCOUNT=nemotron_n3_post` |
| launch_online:pard2:true stages patched non-suffix launcher | PASS | `PY_PATCH_SWERL_SPECDEC_LAUNCHER` |
| launch_online:pard2:true patched gate admits non-suffix methods | PASS | `ngram/draft_model/pard2/eagle3` |
| launch_online:pard2:true refuses unknown launcher shape | FAIL | `missing token: unsupported launcher gate shape; refusing unsafe non-suffix launch` |
| launch_online:pard2:true enables online policy draft | PASS | `policy.draft.enabled=true` |
| launch_online:pard2:true exports online PARD flag | PASS | `PARD_ONLINE_TRAINING=true` |
| launch_online:pard2:true uses PARD-2 specdec | PASS | `speculative_config.method=pard2` |
| launch_online:pard2:true exports PARD-2 vLLM site | PASS | `SOURCE_VLLM_SITE=` |
| timeout stdout is written to a reusable file | PASS | `.tmp_online_gate/swerl_qwen30_submit_contract_outputs/2026-06-16T000000-0700_P3_launch_online_pard_pard_online_step1_smoke_ctx40k_vllm40k_specdec_k5_ray254_py313_r20.stdout.log` |
| timeout stderr is written to a reusable file | PASS | `.tmp_online_gate/swerl_qwen30_submit_contract_outputs/2026-06-16T000000-0700_P3_launch_online_pard_pard_online_step1_smoke_ctx40k_vllm40k_specdec_k5_ray254_py313_r20.stderr.log` |
| failed preflight does not submit on execute | PASS | `blocked_remote_unreachable` |
