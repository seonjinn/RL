# MathRL Qwen32 Online PARD Gate Contract

Overall: **PASS**

Manifest: `/Users/sna/Nemo-RL_Qwen3_Roadmap/docs/oci_hsg_mathrl_qwen32_online_pard_gate_manifest_20260616.csv`

| check | status | detail |
| --- | --- | --- |
| manifest has five qwen32 gate rows | PASS | `5` |
| manifest gate coverage | PASS | `online_pard2_step5_r11_completed, static_pard_step20_reference, static_pard2_14b_step20_reference, online_pard_step5_k5_tp4_r1, online_pard2_step5_r12_replay` |
| manifest includes priority | PASS | `priority` |
| manifest includes action | PASS | `action` |
| manifest includes gate_name | PASS | `gate_name` |
| manifest includes method | PASS | `method` |
| manifest includes k | PASS | `k` |
| manifest includes online_training | PASS | `online_training` |
| manifest includes proof_state | PASS | `proof_state` |
| manifest includes env_overrides | PASS | `env_overrides` |
| manifest includes hydra_overrides | PASS | `hydra_overrides` |
| manifest includes remote_command | PASS | `remote_command` |
| completed online PARD-2 reference is r11 job 3345352 | PASS | `3345352` |
| static PARD reference is completed qwen32 job | PASS | `3333531` |
| static PARD-2 14B reference is tracked | PASS | `3334113` |
| completed_reference:online_pard2:online_pard2_step5_r11_completed uses MathRL Qwen3-32B | PASS | `Qwen3-32B` |
| completed_reference:online_pard2:online_pard2_step5_r11_completed remote_host=oci-hsg-cs-001-vscode-02 | PASS | `oci-hsg-cs-001-vscode-02` |
| completed_reference:online_pard2:online_pard2_step5_r11_completed account=nemotron_n3_post | PASS | `nemotron_n3_post` |
| completed_reference:online_pard2:online_pard2_step5_r11_completed partition=batch | PASS | `batch` |
| completed_reference:online_pard2:online_pard2_step5_r11_completed max_new_tokens=1024 | PASS | `1024` |
| completed_reference:online_pard2:online_pard2_step5_r11_completed min_tokens=1024 | PASS | `1024` |
| completed_reference:online_pard2:online_pard2_step5_r11_completed temperature=1.0 | PASS | `1.0` |
| completed_reference:online_pard2:online_pard2_step5_r11_completed top_p=1.0 | PASS | `1.0` |
| completed_reference:online_pard2:online_pard2_step5_r11_completed top_k=-1 | PASS | `-1` |
| completed_reference:online_pard2:online_pard2_step5_r11_completed max_model_len=4096 | PASS | `4096` |
| completed_reference:online_pard2:online_pard2_step5_r11_completed num_nodes=4 | PASS | `4` |
| completed_reference:online_pard2:online_pard2_step5_r11_completed gpus_per_node=4 | PASS | `4` |
| completed_reference:online_pard2:online_pard2_step5_r11_completed train_global_batch_size=512 | PASS | `512` |
| completed_reference:online_pard2:online_pard2_step5_r11_completed does not serialize WANDB key | PASS | `not present: WANDB_API_KEY` |
| completed_reference:online_pard2:online_pard2_step5_r11_completed command does not serialize WANDB key | PASS | `not present: WANDB_API_KEY` |
| static_reference:pard:static_pard_step20_reference uses MathRL Qwen3-32B | PASS | `Qwen3-32B` |
| static_reference:pard:static_pard_step20_reference remote_host=oci-hsg-cs-001-vscode-02 | PASS | `oci-hsg-cs-001-vscode-02` |
| static_reference:pard:static_pard_step20_reference account=nemotron_n3_post | PASS | `nemotron_n3_post` |
| static_reference:pard:static_pard_step20_reference partition=batch | PASS | `batch` |
| static_reference:pard:static_pard_step20_reference max_new_tokens=1024 | PASS | `1024` |
| static_reference:pard:static_pard_step20_reference min_tokens=1024 | PASS | `1024` |
| static_reference:pard:static_pard_step20_reference temperature=1.0 | PASS | `1.0` |
| static_reference:pard:static_pard_step20_reference top_p=1.0 | PASS | `1.0` |
| static_reference:pard:static_pard_step20_reference top_k=-1 | PASS | `-1` |
| static_reference:pard:static_pard_step20_reference max_model_len=4096 | PASS | `4096` |
| static_reference:pard:static_pard_step20_reference num_nodes=4 | PASS | `4` |
| static_reference:pard:static_pard_step20_reference gpus_per_node=4 | PASS | `4` |
| static_reference:pard:static_pard_step20_reference train_global_batch_size=512 | PASS | `512` |
| static_reference:pard:static_pard_step20_reference does not serialize WANDB key | PASS | `not present: WANDB_API_KEY` |
| static_reference:pard:static_pard_step20_reference command does not serialize WANDB key | PASS | `not present: WANDB_API_KEY` |
| static_reference:pard2:static_pard2_14b_step20_reference uses MathRL Qwen3-32B | PASS | `Qwen3-32B` |
| static_reference:pard2:static_pard2_14b_step20_reference remote_host=oci-hsg-cs-001-vscode-02 | PASS | `oci-hsg-cs-001-vscode-02` |
| static_reference:pard2:static_pard2_14b_step20_reference account=nemotron_n3_post | PASS | `nemotron_n3_post` |
| static_reference:pard2:static_pard2_14b_step20_reference partition=batch | PASS | `batch` |
| static_reference:pard2:static_pard2_14b_step20_reference max_new_tokens=1024 | PASS | `1024` |
| static_reference:pard2:static_pard2_14b_step20_reference min_tokens=1024 | PASS | `1024` |
| static_reference:pard2:static_pard2_14b_step20_reference temperature=1.0 | PASS | `1.0` |
| static_reference:pard2:static_pard2_14b_step20_reference top_p=1.0 | PASS | `1.0` |
| static_reference:pard2:static_pard2_14b_step20_reference top_k=-1 | PASS | `-1` |
| static_reference:pard2:static_pard2_14b_step20_reference max_model_len=4096 | PASS | `4096` |
| static_reference:pard2:static_pard2_14b_step20_reference num_nodes=4 | PASS | `4` |
| static_reference:pard2:static_pard2_14b_step20_reference gpus_per_node=4 | PASS | `4` |
| static_reference:pard2:static_pard2_14b_step20_reference train_global_batch_size=512 | PASS | `512` |
| static_reference:pard2:static_pard2_14b_step20_reference does not serialize WANDB key | PASS | `not present: WANDB_API_KEY` |
| static_reference:pard2:static_pard2_14b_step20_reference command does not serialize WANDB key | PASS | `not present: WANDB_API_KEY` |
| launch_online:online_pard:online_pard_step5_k5_tp4_r1 uses MathRL Qwen3-32B | PASS | `Qwen3-32B` |
| launch_online:online_pard:online_pard_step5_k5_tp4_r1 remote_host=oci-hsg-cs-001-vscode-02 | PASS | `oci-hsg-cs-001-vscode-02` |
| launch_online:online_pard:online_pard_step5_k5_tp4_r1 account=nemotron_n3_post | PASS | `nemotron_n3_post` |
| launch_online:online_pard:online_pard_step5_k5_tp4_r1 partition=batch | PASS | `batch` |
| launch_online:online_pard:online_pard_step5_k5_tp4_r1 max_new_tokens=1024 | PASS | `1024` |
| launch_online:online_pard:online_pard_step5_k5_tp4_r1 min_tokens=1024 | PASS | `1024` |
| launch_online:online_pard:online_pard_step5_k5_tp4_r1 temperature=1.0 | PASS | `1.0` |
| launch_online:online_pard:online_pard_step5_k5_tp4_r1 top_p=1.0 | PASS | `1.0` |
| launch_online:online_pard:online_pard_step5_k5_tp4_r1 top_k=-1 | PASS | `-1` |
| launch_online:online_pard:online_pard_step5_k5_tp4_r1 max_model_len=4096 | PASS | `4096` |
| launch_online:online_pard:online_pard_step5_k5_tp4_r1 num_nodes=4 | PASS | `4` |
| launch_online:online_pard:online_pard_step5_k5_tp4_r1 gpus_per_node=4 | PASS | `4` |
| launch_online:online_pard:online_pard_step5_k5_tp4_r1 train_global_batch_size=512 | PASS | `512` |
| launch_online:online_pard:online_pard_step5_k5_tp4_r1 does not serialize WANDB key | PASS | `not present: WANDB_API_KEY` |
| launch_online:online_pard:online_pard_step5_k5_tp4_r1 command does not serialize WANDB key | PASS | `not present: WANDB_API_KEY` |
| launch_online:online_pard:online_pard_step5_k5_tp4_r1 remote command bash -n | PASS | `` |
| launch_online:online_pard:online_pard_step5_k5_tp4_r1 uses generic online launcher | PASS | `submit_nemorl_online_draft_specdec.sh` |
| launch_online:online_pard:online_pard_step5_k5_tp4_r1 uses requested account | PASS | `nemotron_n3_post` |
| launch_online:online_pard2:online_pard2_step5_r12_replay uses MathRL Qwen3-32B | PASS | `Qwen3-32B` |
| launch_online:online_pard2:online_pard2_step5_r12_replay remote_host=oci-hsg-cs-001-vscode-02 | PASS | `oci-hsg-cs-001-vscode-02` |
| launch_online:online_pard2:online_pard2_step5_r12_replay account=nemotron_n3_post | PASS | `nemotron_n3_post` |
| launch_online:online_pard2:online_pard2_step5_r12_replay partition=batch | PASS | `batch` |
| launch_online:online_pard2:online_pard2_step5_r12_replay max_new_tokens=1024 | PASS | `1024` |
| launch_online:online_pard2:online_pard2_step5_r12_replay min_tokens=1024 | PASS | `1024` |
| launch_online:online_pard2:online_pard2_step5_r12_replay temperature=1.0 | PASS | `1.0` |
| launch_online:online_pard2:online_pard2_step5_r12_replay top_p=1.0 | PASS | `1.0` |
| launch_online:online_pard2:online_pard2_step5_r12_replay top_k=-1 | PASS | `-1` |
| launch_online:online_pard2:online_pard2_step5_r12_replay max_model_len=4096 | PASS | `4096` |
| launch_online:online_pard2:online_pard2_step5_r12_replay num_nodes=4 | PASS | `4` |
| launch_online:online_pard2:online_pard2_step5_r12_replay gpus_per_node=4 | PASS | `4` |
| launch_online:online_pard2:online_pard2_step5_r12_replay train_global_batch_size=512 | PASS | `512` |
| launch_online:online_pard2:online_pard2_step5_r12_replay does not serialize WANDB key | PASS | `not present: WANDB_API_KEY` |
| launch_online:online_pard2:online_pard2_step5_r12_replay command does not serialize WANDB key | PASS | `not present: WANDB_API_KEY` |
| launch_online:online_pard2:online_pard2_step5_r12_replay remote command bash -n | PASS | `` |
| launch_online:online_pard2:online_pard2_step5_r12_replay uses generic online launcher | PASS | `submit_nemorl_online_draft_specdec.sh` |
| launch_online:online_pard2:online_pard2_step5_r12_replay uses requested account | PASS | `nemotron_n3_post` |
| online PARD uses PARD draft format | PASS | `DRAFT_FORMAT=pard` |
| online PARD uses draft_model specdec | PASS | `SPECDEC_METHOD=draft_model` |
| online PARD uses K5 | PASS | `NUM_SPECULATIVE_TOKENS=5` |
| online PARD uses draft TP4 | PASS | `DRAFT_TP=4` |
| online PARD enables online training | PASS | `PARD_ONLINE_TRAINING=true` |
| online PARD draft type is PARD | PASS | `POLICY_DRAFT_TYPE=pard` |
| online PARD loss is hard CE | PASS | `POLICY_DRAFT_LOSS=hard_ce` |
| online PARD caps k-slot sequence | PASS | `POLICY_DRAFT_MAX_TRAINING_SEQUENCE_LENGTH=256` |
| online PARD uses generation TP4 | PASS | `policy.generation.vllm_cfg.tensor_parallel_size=4` |
| online PARD uses training TP4 | PASS | `policy.megatron_cfg.tensor_model_parallel_size=4` |
| online PARD disables context parallelism | PASS | `policy.megatron_cfg.context_parallel_size=1` |
| online PARD disables sequence packing | PASS | `policy.sequence_packing.enabled=false` |
| online PARD keeps force-on-policy ratio | PASS | `loss_fn.force_on_policy_ratio=true` |
| online PARD row is the missing submit-ready proof | PASS | `when_ssh_dns_recovers` |
| online PARD-2 replay uses PARD-2 draft format | PASS | `DRAFT_FORMAT=pard2` |
| online PARD-2 replay uses PARD-2 specdec | PASS | `SPECDEC_METHOD=pard2` |
| online PARD-2 replay uses K3 | PASS | `NUM_SPECULATIVE_TOKENS=3` |
| online PARD-2 replay uses draft TP4 | PASS | `DRAFT_TP=4` |
| online PARD-2 replay carries source vLLM site | PASS | `SOURCE_VLLM_SITE=` |
| online PARD-2 replay draft type is PARD-2 | PASS | `POLICY_DRAFT_TYPE=pard2` |
| online PARD-2 replay disables generic fallback | PASS | `POLICY_DRAFT_ALLOW_GENERIC_PARD2_FALLBACK=false` |
| online PARD-2 replay uses local transformer spec | PASS | `NRL_FORCE_LOCAL_TRANSFORMER_SPEC=true` |
| online PARD-2 replay reuses unlocked actor venv | PASS | `NRL_ACTOR_UV_LOCK_MODE=unlocked` |
| online PARD-2 replay serializes actor venv | PASS | `NRL_SERIALIZE_ACTOR_VENV_CREATION=true` |
