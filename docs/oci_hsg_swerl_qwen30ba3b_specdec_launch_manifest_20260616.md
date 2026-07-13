# SWE-RL Qwen3-30B-A3B SpecDec Launch Manifest

Generated: `2026-06-17T14:33:05-07:00`

These are submit-ready rows for the missing integrated SWE-RL proof points. They keep the successful ctx40k/vLLM40k qwen30 baseline geometry and use RL sampling (`temperature=1.0`, `top_p=1.0`, `top_k=-1`).

| priority | action | method | K | online | submit | depends on |
| --- | --- | --- | ---: | --- | --- | --- |
| P0 | `poll_existing` | `suffix` | 32 | false | already_submitted | remote SSH/DNS recovery |
| P1 | `launch_static` | `eagle3` | 3 | false | when_ssh_dns_recovers | can run in parallel with suffix r16 poll |
| P2 | `launch_static` | `pard` | 5 | false | when_ssh_dns_recovers | can run after Eagle-3 or in parallel if nodes are available |
| P2 | `launch_static` | `pard2` | 3 | false | when_ssh_dns_recovers | PARD2 patched vLLM source site available in SWE repo runtime |
| P3 | `launch_online` | `pard` | 5 | true | when_ssh_dns_recovers | static PARD or explicit decision to test online path directly |
| P3 | `launch_online` | `pard2` | 3 | true | when_ssh_dns_recovers | qwen32 online PARD-2 hard-CE patch staged in SWE repo plus static PARD-2 gate |

## Override Details

### suffix_step1_smoke_ctx40k_vllm40k_specdec_k32_arcticpy313_ray254_py313_r16

- launcher: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_launchers/20260616_oci_hsg_swerl_qwen30ba3b_suffix_step1_smoke_py313_r1/run_qwen30ba3b_swerl_gb200_suffix_smoke.sh`
- logs: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_suffix_step1_smoke_ctx40k_vllm40k_specdec_k32_arcticpy313_ray254_py313_r16/suffix_step1`
- expected artifacts: `docs/oci_hsg_swerl_qwen30ba3b_suffix_ctx40k_3351394_{metrics,summary}_20260616.csv`

Env overrides:

```bash
poll sacct/squeue; copy ray-driver.log; parse partial steps immediately
```

Hydra overrides:

```bash
already submitted; suffix K32 ctx40k/vllm40k
```

### eagle3_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r17

- launcher: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_launchers/20260616_oci_hsg_swerl_qwen30ba3b_suffix_step1_smoke_py313_r1/run_qwen30ba3b_swerl_gb200_suffix_smoke.sh`
- logs: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_eagle3_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r17/eagle3_step1`
- expected artifacts: `docs/oci_hsg_swerl_qwen30ba3b_eagle3_ctx40k_step1_{metrics,summary}_20260616.csv`

Env overrides:

```bash
ACCOUNT=nemotron_n3_post PARTITION=batch MAX_STEPS=1 SEQLEN=40960 MAX_MODEL_LEN=40960 MAX_TOTAL_SEQUENCE_LENGTH=40960 MAX_NUM_BATCHED_TOKENS=65536 TOTAL_NODES=8 GBS=64 GEN_TRAIN_RATIO=1 VLLM_TP=1 HYBRIDEP=0 RAY_VERSION=2.54.0 RAY_PYTHON_VERSION=3.13.13 RAY_PYTHON_SPEC=3.13.13 UV_PYTHON=3.13.13 GENERATION_TEMPERATURE=1.0 GENERATION_TOP_P=1.0 GENERATION_TOP_K=-1 METHOD=eagle3 RUN_ID=20260616_oci_hsg_swerl_qwen30ba3b_eagle3_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r17 BASE_LOG_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_eagle3_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r17/eagle3_step1 DRAFT_FORMAT=eagle3 ENABLE_VLLM_SPECDEC=true POLICY_DRAFT_ENABLED=false SPECDEC_METHOD=eagle3 NUM_SPECULATIVE_TOKENS=3 DRAFT_MODEL=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--RedHatAI--Qwen3-30B-A3B-Thinking-2507-speculator.eagle3/snapshots/a7ec796dd65236f1ecd4ed2958a7f0689e5da5cf DRAFT_TP=1 INCLUDE_DRAFT_TP=true SPECDEC_PARALLEL_DRAFTING=false
```

Hydra overrides:

```bash
grpo.max_num_steps=1 policy.max_total_sequence_length=40960 policy.generation.vllm_cfg.max_model_len=40960 policy.generation.vllm_cfg.max_num_batched_tokens=65536 policy.generation.vllm_cfg.tensor_parallel_size=1 policy.generation.temperature=1.0 policy.generation.top_p=1.0 policy.generation.top_k=-1 policy.draft.enabled=false ++policy.generation.vllm_kwargs.speculative_config.method=eagle3 ++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=3 ++policy.generation.vllm_kwargs.speculative_config.model=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--RedHatAI--Qwen3-30B-A3B-Thinking-2507-speculator.eagle3/snapshots/a7ec796dd65236f1ecd4ed2958a7f0689e5da5cf ++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1
```

### pard_step1_smoke_ctx40k_vllm40k_specdec_k5_ray254_py313_r18

- launcher: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_launchers/20260616_oci_hsg_swerl_qwen30ba3b_suffix_step1_smoke_py313_r1/run_qwen30ba3b_swerl_gb200_suffix_smoke.sh`
- logs: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_pard_step1_smoke_ctx40k_vllm40k_specdec_k5_ray254_py313_r18/pard_step1`
- expected artifacts: `docs/oci_hsg_swerl_qwen30ba3b_pard_ctx40k_step1_{metrics,summary}_20260616.csv`

Env overrides:

```bash
ACCOUNT=nemotron_n3_post PARTITION=batch MAX_STEPS=1 SEQLEN=40960 MAX_MODEL_LEN=40960 MAX_TOTAL_SEQUENCE_LENGTH=40960 MAX_NUM_BATCHED_TOKENS=65536 TOTAL_NODES=8 GBS=64 GEN_TRAIN_RATIO=1 VLLM_TP=1 HYBRIDEP=0 RAY_VERSION=2.54.0 RAY_PYTHON_VERSION=3.13.13 RAY_PYTHON_SPEC=3.13.13 UV_PYTHON=3.13.13 GENERATION_TEMPERATURE=1.0 GENERATION_TOP_P=1.0 GENERATION_TOP_K=-1 METHOD=pard RUN_ID=20260616_oci_hsg_swerl_qwen30ba3b_pard_step1_smoke_ctx40k_vllm40k_specdec_k5_ray254_py313_r18 BASE_LOG_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_pard_step1_smoke_ctx40k_vllm40k_specdec_k5_ray254_py313_r18/pard_step1 DRAFT_FORMAT=pard ENABLE_VLLM_SPECDEC=true SPECDEC_METHOD=draft_model NUM_SPECULATIVE_TOKENS=5 DRAFT_MODEL=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--amd--PARD-Qwen3-0.6B/snapshots/34e73caf94c021c3d4b8bad86ebd3616850ab4fd DRAFT_TP=1 INCLUDE_DRAFT_TP=true SPECDEC_PARALLEL_DRAFTING=true
```

Hydra overrides:

```bash
grpo.max_num_steps=1 policy.max_total_sequence_length=40960 policy.generation.vllm_cfg.max_model_len=40960 policy.generation.vllm_cfg.max_num_batched_tokens=65536 policy.generation.vllm_cfg.tensor_parallel_size=1 policy.generation.temperature=1.0 policy.generation.top_p=1.0 policy.generation.top_k=-1 policy.draft.enabled=false ++policy.generation.vllm_kwargs.speculative_config.method=draft_model ++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=5 ++policy.generation.vllm_kwargs.speculative_config.model=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--amd--PARD-Qwen3-0.6B/snapshots/34e73caf94c021c3d4b8bad86ebd3616850ab4fd ++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1 ++policy.generation.vllm_kwargs.speculative_config.parallel_drafting=true
```

### pard2_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r19

- launcher: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_launchers/20260616_oci_hsg_swerl_qwen30ba3b_suffix_step1_smoke_py313_r1/run_qwen30ba3b_swerl_gb200_suffix_smoke.sh`
- logs: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_pard2_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r19/pard2_step1`
- expected artifacts: `docs/oci_hsg_swerl_qwen30ba3b_pard2_ctx40k_step1_{metrics,summary}_20260616.csv`

Env overrides:

```bash
ACCOUNT=nemotron_n3_post PARTITION=batch MAX_STEPS=1 SEQLEN=40960 MAX_MODEL_LEN=40960 MAX_TOTAL_SEQUENCE_LENGTH=40960 MAX_NUM_BATCHED_TOKENS=65536 TOTAL_NODES=8 GBS=64 GEN_TRAIN_RATIO=1 VLLM_TP=1 HYBRIDEP=0 RAY_VERSION=2.54.0 RAY_PYTHON_VERSION=3.13.13 RAY_PYTHON_SPEC=3.13.13 UV_PYTHON=3.13.13 GENERATION_TEMPERATURE=1.0 GENERATION_TOP_P=1.0 GENERATION_TOP_K=-1 METHOD=pard2 RUN_ID=20260616_oci_hsg_swerl_qwen30ba3b_pard2_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r19 BASE_LOG_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_pard2_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r19/pard2_step1 DRAFT_FORMAT=pard2 ENABLE_VLLM_SPECDEC=true SPECDEC_METHOD=pard2 NUM_SPECULATIVE_TOKENS=3 DRAFT_MODEL=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--amd--PARD2-Qwen3-8B/snapshots/6cf9edc27c8afa8088a6f61fc0edb875a621c43d DRAFT_TP=1 INCLUDE_DRAFT_TP=true SPECDEC_PARALLEL_DRAFTING=true SOURCE_VLLM_SITE=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-pard2-official-smoke-20260612/.container_cache/vllm_pard2_official_target_feat_pyoverlay_nostable_nofp4out_basefa_nofp4fusion_20260614
```

Hydra overrides:

```bash
grpo.max_num_steps=1 policy.max_total_sequence_length=40960 policy.generation.vllm_cfg.max_model_len=40960 policy.generation.vllm_cfg.max_num_batched_tokens=65536 policy.generation.vllm_cfg.tensor_parallel_size=1 policy.generation.temperature=1.0 policy.generation.top_p=1.0 policy.generation.top_k=-1 policy.draft.enabled=false ++policy.generation.vllm_kwargs.speculative_config.method=pard2 ++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=3 ++policy.generation.vllm_kwargs.speculative_config.model=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--amd--PARD2-Qwen3-8B/snapshots/6cf9edc27c8afa8088a6f61fc0edb875a621c43d ++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1 ++policy.generation.vllm_kwargs.speculative_config.parallel_drafting=true
```

### pard_online_step1_smoke_ctx40k_vllm40k_specdec_k5_ray254_py313_r20

- launcher: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_launchers/20260616_oci_hsg_swerl_qwen30ba3b_suffix_step1_smoke_py313_r1/run_qwen30ba3b_swerl_gb200_suffix_smoke.sh`
- logs: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_pard_online_step1_smoke_ctx40k_vllm40k_specdec_k5_ray254_py313_r20/pard_step1`
- expected artifacts: `docs/oci_hsg_swerl_qwen30ba3b_pard_online_ctx40k_step1_{metrics,summary}_20260616.csv`

Env overrides:

```bash
ACCOUNT=nemotron_n3_post PARTITION=batch MAX_STEPS=1 SEQLEN=40960 MAX_MODEL_LEN=40960 MAX_TOTAL_SEQUENCE_LENGTH=40960 MAX_NUM_BATCHED_TOKENS=65536 TOTAL_NODES=8 GBS=64 GEN_TRAIN_RATIO=1 VLLM_TP=1 HYBRIDEP=0 RAY_VERSION=2.54.0 RAY_PYTHON_VERSION=3.13.13 RAY_PYTHON_SPEC=3.13.13 UV_PYTHON=3.13.13 GENERATION_TEMPERATURE=1.0 GENERATION_TOP_P=1.0 GENERATION_TOP_K=-1 METHOD=pard RUN_ID=20260616_oci_hsg_swerl_qwen30ba3b_pard_online_step1_smoke_ctx40k_vllm40k_specdec_k5_ray254_py313_r20 BASE_LOG_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_pard_online_step1_smoke_ctx40k_vllm40k_specdec_k5_ray254_py313_r20/pard_step1 DRAFT_FORMAT=pard ENABLE_VLLM_SPECDEC=true SPECDEC_METHOD=draft_model NUM_SPECULATIVE_TOKENS=5 DRAFT_MODEL=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--amd--PARD-Qwen3-0.6B/snapshots/34e73caf94c021c3d4b8bad86ebd3616850ab4fd DRAFT_TP=1 INCLUDE_DRAFT_TP=true SPECDEC_PARALLEL_DRAFTING=true PARD_ONLINE_TRAINING=true POLICY_DRAFT_ENABLED=true POLICY_DRAFT_TYPE=pard POLICY_DRAFT_LOSS=hard_ce PARD_TRAINING_MODE=k_slot POLICY_DRAFT_MAX_TRAINING_SEQUENCE_LENGTH=128 POLICY_DRAFT_TRAIN_INTERVAL=1 POLICY_DRAFT_REFIT_INTERVAL=1 POLICY_DRAFT_CAT_WEIGHTING=false
```

Hydra overrides:

```bash
grpo.max_num_steps=1 policy.max_total_sequence_length=40960 policy.generation.vllm_cfg.max_model_len=40960 policy.generation.vllm_cfg.max_num_batched_tokens=65536 policy.generation.vllm_cfg.tensor_parallel_size=1 policy.generation.temperature=1.0 policy.generation.top_p=1.0 policy.generation.top_k=-1 ++policy.generation.vllm_kwargs.speculative_config.method=draft_model ++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=5 ++policy.generation.vllm_kwargs.speculative_config.model=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--amd--PARD-Qwen3-0.6B/snapshots/34e73caf94c021c3d4b8bad86ebd3616850ab4fd ++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1 ++policy.generation.vllm_kwargs.speculative_config.parallel_drafting=true policy.draft.enabled=true ++policy.draft.type=pard ++policy.draft.loss=hard_ce ++policy.draft.training_mode=k_slot ++policy.draft.max_training_sequence_length=128 ++policy.draft.train_interval=1 ++policy.draft.refit_interval=1 ++policy.draft.cat_weighting=false
```

### pard2_online_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r21

- launcher: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_launchers/20260616_oci_hsg_swerl_qwen30ba3b_suffix_step1_smoke_py313_r1/run_qwen30ba3b_swerl_gb200_suffix_smoke.sh`
- logs: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_pard2_online_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r21/pard2_step1`
- expected artifacts: `docs/oci_hsg_swerl_qwen30ba3b_pard2_online_ctx40k_step1_{metrics,summary}_20260616.csv`

Env overrides:

```bash
ACCOUNT=nemotron_n3_post PARTITION=batch MAX_STEPS=1 SEQLEN=40960 MAX_MODEL_LEN=40960 MAX_TOTAL_SEQUENCE_LENGTH=40960 MAX_NUM_BATCHED_TOKENS=65536 TOTAL_NODES=8 GBS=64 GEN_TRAIN_RATIO=1 VLLM_TP=1 HYBRIDEP=0 RAY_VERSION=2.54.0 RAY_PYTHON_VERSION=3.13.13 RAY_PYTHON_SPEC=3.13.13 UV_PYTHON=3.13.13 GENERATION_TEMPERATURE=1.0 GENERATION_TOP_P=1.0 GENERATION_TOP_K=-1 METHOD=pard2 RUN_ID=20260616_oci_hsg_swerl_qwen30ba3b_pard2_online_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r21 BASE_LOG_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/swerl_qwen30ba3b_logs/20260616_oci_hsg_swerl_qwen30ba3b_pard2_online_step1_smoke_ctx40k_vllm40k_specdec_k3_ray254_py313_r21/pard2_step1 DRAFT_FORMAT=pard2 ENABLE_VLLM_SPECDEC=true SPECDEC_METHOD=pard2 NUM_SPECULATIVE_TOKENS=3 DRAFT_MODEL=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--amd--PARD2-Qwen3-8B/snapshots/6cf9edc27c8afa8088a6f61fc0edb875a621c43d DRAFT_TP=1 INCLUDE_DRAFT_TP=true SPECDEC_PARALLEL_DRAFTING=true SOURCE_VLLM_SITE=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-pard2-official-smoke-20260612/.container_cache/vllm_pard2_official_target_feat_pyoverlay_nostable_nofp4out_basefa_nofp4fusion_20260614 PARD_ONLINE_TRAINING=true POLICY_DRAFT_ENABLED=true POLICY_DRAFT_TYPE=pard2 POLICY_DRAFT_LOSS=pard2 PARD_TRAINING_MODE=k_slot POLICY_DRAFT_MAX_TRAINING_SEQUENCE_LENGTH=128 POLICY_DRAFT_TRAIN_INTERVAL=1 POLICY_DRAFT_REFIT_INTERVAL=1 POLICY_DRAFT_CAT_WEIGHTING=true POLICY_DRAFT_ALLOW_GENERIC_PARD2_FALLBACK=false
```

Hydra overrides:

```bash
grpo.max_num_steps=1 policy.max_total_sequence_length=40960 policy.generation.vllm_cfg.max_model_len=40960 policy.generation.vllm_cfg.max_num_batched_tokens=65536 policy.generation.vllm_cfg.tensor_parallel_size=1 policy.generation.temperature=1.0 policy.generation.top_p=1.0 policy.generation.top_k=-1 ++policy.generation.vllm_kwargs.speculative_config.method=pard2 ++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=3 ++policy.generation.vllm_kwargs.speculative_config.model=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--amd--PARD2-Qwen3-8B/snapshots/6cf9edc27c8afa8088a6f61fc0edb875a621c43d ++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1 ++policy.generation.vllm_kwargs.speculative_config.parallel_drafting=true policy.draft.enabled=true ++policy.draft.type=pard2 ++policy.draft.loss=pard2 ++policy.draft.training_mode=k_slot ++policy.draft.max_training_sequence_length=128 ++policy.draft.train_interval=1 ++policy.draft.refit_interval=1 ++policy.draft.cat_weighting=true ++policy.draft.allow_generic_pard2_fallback=false
```
