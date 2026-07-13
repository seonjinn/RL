# MathRL Qwen32 Online PARD Gate Submit Contract

Overall: **PASS**

Manifest: `/Users/sna/Nemo-RL_Qwen3_Roadmap/docs/oci_hsg_mathrl_qwen32_online_pard_gate_manifest_20260616.csv`
Submit helper: `/Users/sna/Nemo-RL_Qwen3_Roadmap/scripts/submit_mathrl_qwen32_online_pard_gate_from_manifest.py`
Status CSV: `/Users/sna/Nemo-RL_Qwen3_Roadmap/docs/oci_hsg_mathrl_qwen32_online_pard_gate_submit_status_20260616.csv`

| check | status | detail |
| --- | --- | --- |
| manifest has five qwen32 gate rows | PASS | `5` |
| manifest gate coverage | PASS | `online_pard2_step5_r11_completed, static_pard_step20_reference, static_pard2_14b_step20_reference, online_pard_step5_k5_tp4_r1, online_pard2_step5_r12_replay` |
| submit status includes stdout_path | PASS | `stdout_path` |
| submit status includes stderr_path | PASS | `stderr_path` |
| submit helper exposes command output directory | PASS | `/Users/sna/Nemo-RL_Qwen3_Roadmap/docs/oci_hsg_mathrl_qwen32_online_pard_gate_command_outputs_20260616` |
| submit helper flushes status after each row | PASS | `True` |
| completed_reference:online_pard2:online_pard2_step5_r11_completed helper validation | PASS | `ok` |
| completed_reference:online_pard2:online_pard2_step5_r11_completed command bash -n | PASS | `` |
| completed_reference:online_pard2:online_pard2_step5_r11_completed command does not serialize WANDB key | PASS | `not present: WANDB_API_KEY` |
| completed_reference:online_pard2:online_pard2_step5_r11_completed command does not serialize HF token | PASS | `not present: HUGGINGFACE_TOKEN` |
| completed_reference:online_pard2:online_pard2_step5_r11_completed polls squeue | PASS | `squeue -j 3345352` |
| completed_reference:online_pard2:online_pard2_step5_r11_completed polls sacct | PASS | `sacct -X -j 3345352` |
| completed_reference:online_pard2:online_pard2_step5_r11_completed tails bounded log chunk | PASS | `tail -n 80` |
| completed_reference:online_pard2:online_pard2_step5_r11_completed emits reusable tail chunk markers | PASS | `[tail-chunk] lines` |
| completed_reference:online_pard2:online_pard2_step5_r11_completed limits recent log files | PASS | `tail -n 3` |
| completed_reference:online_pard2:online_pard2_step5_r11_completed bounds remote log probes | PASS | `timeout 20s` |
| static_reference:pard:static_pard_step20_reference helper validation | PASS | `ok` |
| static_reference:pard:static_pard_step20_reference command bash -n | PASS | `` |
| static_reference:pard:static_pard_step20_reference command does not serialize WANDB key | PASS | `not present: WANDB_API_KEY` |
| static_reference:pard:static_pard_step20_reference command does not serialize HF token | PASS | `not present: HUGGINGFACE_TOKEN` |
| static_reference:pard:static_pard_step20_reference polls squeue | PASS | `squeue -j 3333531` |
| static_reference:pard:static_pard_step20_reference polls sacct | PASS | `sacct -X -j 3333531` |
| static_reference:pard:static_pard_step20_reference tails bounded log chunk | PASS | `tail -n 80` |
| static_reference:pard:static_pard_step20_reference emits reusable tail chunk markers | PASS | `[tail-chunk] lines` |
| static_reference:pard:static_pard_step20_reference limits recent log files | PASS | `tail -n 3` |
| static_reference:pard:static_pard_step20_reference bounds remote log probes | PASS | `timeout 20s` |
| static_reference:pard2:static_pard2_14b_step20_reference helper validation | PASS | `ok` |
| static_reference:pard2:static_pard2_14b_step20_reference command bash -n | PASS | `` |
| static_reference:pard2:static_pard2_14b_step20_reference command does not serialize WANDB key | PASS | `not present: WANDB_API_KEY` |
| static_reference:pard2:static_pard2_14b_step20_reference command does not serialize HF token | PASS | `not present: HUGGINGFACE_TOKEN` |
| static_reference:pard2:static_pard2_14b_step20_reference polls squeue | PASS | `squeue -j 3334113` |
| static_reference:pard2:static_pard2_14b_step20_reference polls sacct | PASS | `sacct -X -j 3334113` |
| static_reference:pard2:static_pard2_14b_step20_reference tails bounded log chunk | PASS | `tail -n 80` |
| static_reference:pard2:static_pard2_14b_step20_reference emits reusable tail chunk markers | PASS | `[tail-chunk] lines` |
| static_reference:pard2:static_pard2_14b_step20_reference limits recent log files | PASS | `tail -n 3` |
| static_reference:pard2:static_pard2_14b_step20_reference bounds remote log probes | PASS | `timeout 20s` |
| launch_online:online_pard:online_pard_step5_k5_tp4_r1 helper validation | PASS | `ok` |
| launch_online:online_pard:online_pard_step5_k5_tp4_r1 command bash -n | PASS | `` |
| launch_online:online_pard:online_pard_step5_k5_tp4_r1 command does not serialize WANDB key | PASS | `not present: WANDB_API_KEY` |
| launch_online:online_pard:online_pard_step5_k5_tp4_r1 command does not serialize HF token | PASS | `not present: HUGGINGFACE_TOKEN` |
| launch_online:online_pard:online_pard_step5_k5_tp4_r1 uses generic online launcher | PASS | `submit_nemorl_online_draft_specdec.sh` |
| launch_online:online_pard:online_pard_step5_k5_tp4_r1 carries account env | PASS | `ACCOUNT=` |
| launch_online:online_pard:online_pard_step5_k5_tp4_r1 uses requested account | PASS | `nemotron_n3_post` |
| launch_online:online_pard:online_pard_step5_k5_tp4_r1 carries max steps | PASS | `MAX_STEPS=` |
| launch_online:online_pard:online_pard_step5_k5_tp4_r1 carries min generation | PASS | `NRL_VLLM_GENERATION_MIN_TOKENS=` |
| launch_online:online_pard2:online_pard2_step5_r12_replay helper validation | PASS | `ok` |
| launch_online:online_pard2:online_pard2_step5_r12_replay command bash -n | PASS | `` |
| launch_online:online_pard2:online_pard2_step5_r12_replay command does not serialize WANDB key | PASS | `not present: WANDB_API_KEY` |
| launch_online:online_pard2:online_pard2_step5_r12_replay command does not serialize HF token | PASS | `not present: HUGGINGFACE_TOKEN` |
| launch_online:online_pard2:online_pard2_step5_r12_replay uses generic online launcher | PASS | `submit_nemorl_online_draft_specdec.sh` |
| launch_online:online_pard2:online_pard2_step5_r12_replay carries account env | PASS | `ACCOUNT=` |
| launch_online:online_pard2:online_pard2_step5_r12_replay uses requested account | PASS | `nemotron_n3_post` |
| launch_online:online_pard2:online_pard2_step5_r12_replay carries max steps | PASS | `MAX_STEPS=` |
| launch_online:online_pard2:online_pard2_step5_r12_replay carries min generation | PASS | `NRL_VLLM_GENERATION_MIN_TOKENS=` |
| online PARD launch exports PARD draft format | PASS | `DRAFT_FORMAT='"'"'pard` |
| online PARD launch uses draft_model specdec | PASS | `SPECDEC_METHOD='"'"'draft_model` |
| online PARD launch uses hard CE | PASS | `POLICY_DRAFT_LOSS='"'"'hard_ce` |
| online PARD launch uses K5 | PASS | `NUM_SPECULATIVE_TOKENS='"'"'5` |
| online PARD launch uses draft TP4 | PASS | `DRAFT_TP='"'"'4` |
| online PARD launch enables online training | PASS | `PARD_ONLINE_TRAINING='"'"'true` |
| online PARD-2 replay exports PARD-2 draft format | PASS | `DRAFT_FORMAT='"'"'pard2` |
| online PARD-2 replay uses PARD-2 specdec | PASS | `SPECDEC_METHOD='"'"'pard2` |
| online PARD-2 replay carries source vLLM site | PASS | `SOURCE_VLLM_SITE=` |
| online PARD-2 replay uses K3 | PASS | `NUM_SPECULATIVE_TOKENS='"'"'3` |
| online PARD-2 replay disables fallback | PASS | `POLICY_DRAFT_ALLOW_GENERIC_PARD2_FALLBACK='"'"'false` |
| submit status has five rows | PASS | `5` |
| submit status follows manifest gate order | PASS | `online_pard2_step5_r11_completed, static_pard_step20_reference, static_pard2_14b_step20_reference, online_pard_step5_k5_tp4_r1, online_pard2_step5_r12_replay` |
| submit status CSV includes stdout_path | PASS | `stdout_path` |
| submit status CSV includes stderr_path | PASS | `stderr_path` |
| submit status CSV includes operation_state | PASS | `operation_state` |
| submit status CSV includes remote_command | PASS | `remote_command` |
| submit status operation summary | PASS | `{'not_selected': 4, 'dry_run_ready': 1}` |
| execute mode requires an explicit row filter | PASS | `--execute requires --action, --gate-name, or --method unless --allow-all-execute is set` |
| timeout stdout is written to a reusable file | PASS | `.tmp_online_gate/mathrl_qwen32_submit_contract_outputs/2026-06-16T000000-0700_P2_launch_online_online_pard_online_pard_step5_k5_tp4_r1.stdout.log` |
| timeout stderr is written to a reusable file | PASS | `.tmp_online_gate/mathrl_qwen32_submit_contract_outputs/2026-06-16T000000-0700_P2_launch_online_online_pard_online_pard_step5_k5_tp4_r1.stderr.log` |
| failed preflight does not submit on execute | PASS | `blocked_remote_unreachable` |
