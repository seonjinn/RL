# Qwen8 Official PARD-2 Comparison Contract Validation

Overall: **PASS**

Launcher: `/Users/sna/Nemo-RL_Qwen3_Roadmap/experiments/eagle3_online/submit_lyris_qwen8_pard2_official_comparison_20260613.sh`

| check | status | detail |
| --- | --- | --- |
| default draft_format="auto" | PASS | `local draft_format="auto"` |
| default draft_model="" | PASS | `local draft_model=""` |
| default policy_draft_enabled="false" | PASS | `local policy_draft_enabled="false"` |
| default pard_online_training="false" | PASS | `local pard_online_training="false"` |
| default enable_vllm_specdec="false" | PASS | `local enable_vllm_specdec="false"` |
| default specdec_method="eagle3" | PASS | `local specdec_method="eagle3"` |
| default source_vllm_site="" | PASS | `local source_vllm_site=""` |
| default pard2_patch_dir="" | PASS | `local pard2_patch_dir=""` |
| default num_spec_tokens="0" | PASS | `local num_spec_tokens="0"` |
| default include_draft_tp="false" | PASS | `local include_draft_tp="false"` |
| default parallel_drafting="false" | PASS | `local parallel_drafting="false"` |
| default debug_draft_refit="false" | PASS | `local debug_draft_refit="false"` |
| default force_local_transformer_spec="false" | PASS | `local force_local_transformer_spec="false"` |
| default actor_uv_lock_mode="--locked" | PASS | `local actor_uv_lock_mode="--locked"` |
| default force_rebuild_actor_venvs="false" | PASS | `local force_rebuild_actor_venvs="false"` |
| default policy_draft_loss="hard_ce" | PASS | `local policy_draft_loss="hard_ce"` |
| default policy_draft_cat_weighting="false" | PASS | `local policy_draft_cat_weighting="false"` |
| default variants include baseline/static/online PARD-2 | PASS | `VARIANTS="${VARIANTS:-baseline static_pard2 online_pard2}"` |
| has explicit Lyris/OCI profile selector | PASS | `CLUSTER_PROFILE="${CLUSTER_PROFILE:-lyris}"` |
| OCI profile uses OCI-HSG host | PASS | `DEFAULT_REMOTE_HOST="oci-hsg-cs-001-vscode-02"` |
| OCI profile uses batch partition | PASS | `DEFAULT_PARTITION="batch"` |
| OCI profile stays within batch walltime limit | PASS | `DEFAULT_WALLTIME="04:00:00"` |
| OCI profile uses existing Qwen8 smoke Python setting | PASS | `DEFAULT_UV_PYTHON="3.13.13"` |
| OCI profile uses existing Qwen8 smoke Ray setting | PASS | `DEFAULT_RAY_VERSION="2.55.1"` |
| OCI profile uses verified PARD-2 vLLM site | PASS | `DEFAULT_SOURCE_VLLM_SITE="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-pard2-official-smoke-20260612/.container_cache/vllm_pard2_official_target_feat"` |
| walltime default is profile-specific | PASS | `WALLTIME="${WALLTIME:-${DEFAULT_WALLTIME}}"` |
| baseline case leaves no-spec defaults untouched | PASS | `` |
| static_pard2 sets draft_format="pard2" | PASS | `draft_format="pard2"` |
| static_pard2 sets draft_model="${DRAFTER}" | PASS | `draft_model="${DRAFTER}"` |
| static_pard2 sets enable_vllm_specdec="true" | PASS | `enable_vllm_specdec="true"` |
| static_pard2 sets specdec_method="pard2" | PASS | `specdec_method="pard2"` |
| static_pard2 sets source_vllm_site="${SOURCE_VLLM_SITE}" | PASS | `source_vllm_site="${SOURCE_VLLM_SITE}"` |
| static_pard2 sets pard2_patch_dir="${STAGE_REPO}/experiments/eagle3_qwen3_235b/patches" | PASS | `pard2_patch_dir="${STAGE_REPO}/experiments/eagle3_qwen3_235b/patches"` |
| static_pard2 sets num_spec_tokens="${SPEC_NUM_TOKENS}" | PASS | `num_spec_tokens="${SPEC_NUM_TOKENS}"` |
| static_pard2 sets include_draft_tp="true" | PASS | `include_draft_tp="true"` |
| static_pard2 sets parallel_drafting="true" | PASS | `parallel_drafting="true"` |
| static_pard2 sets force_local_transformer_spec="true" | PASS | `force_local_transformer_spec="true"` |
| static_pard2 does not set policy_draft_enabled="true" | PASS | `` |
| static_pard2 does not set pard_online_training="true" | PASS | `` |
| online_pard2 sets draft_format="pard2" | PASS | `draft_format="pard2"` |
| online_pard2 sets draft_model="${DRAFTER}" | PASS | `draft_model="${DRAFTER}"` |
| online_pard2 sets enable_vllm_specdec="true" | PASS | `enable_vllm_specdec="true"` |
| online_pard2 sets specdec_method="pard2" | PASS | `specdec_method="pard2"` |
| online_pard2 sets source_vllm_site="${SOURCE_VLLM_SITE}" | PASS | `source_vllm_site="${SOURCE_VLLM_SITE}"` |
| online_pard2 sets pard2_patch_dir="${STAGE_REPO}/experiments/eagle3_qwen3_235b/patches" | PASS | `pard2_patch_dir="${STAGE_REPO}/experiments/eagle3_qwen3_235b/patches"` |
| online_pard2 sets num_spec_tokens="${SPEC_NUM_TOKENS}" | PASS | `num_spec_tokens="${SPEC_NUM_TOKENS}"` |
| online_pard2 sets include_draft_tp="true" | PASS | `include_draft_tp="true"` |
| online_pard2 sets parallel_drafting="true" | PASS | `parallel_drafting="true"` |
| online_pard2 sets force_local_transformer_spec="true" | PASS | `force_local_transformer_spec="true"` |
| online_pard2 sets policy_draft_enabled="true" | PASS | `policy_draft_enabled="true"` |
| online_pard2 sets pard_online_training="true" | PASS | `pard_online_training="true"` |
| online_pard2 sets debug_draft_refit="true" | PASS | `debug_draft_refit="true"` |
| online_pard2 sets actor_uv_lock_mode="unlocked" | PASS | `actor_uv_lock_mode="unlocked"` |
| online_pard2 sets force_rebuild_actor_venvs="${NRL_FORCE_REBUILD_ACTOR_VENVS}" | PASS | `force_rebuild_actor_venvs="${NRL_FORCE_REBUILD_ACTOR_VENVS}"` |
| online_pard2 sets policy_draft_loss="pard2" | PASS | `policy_draft_loss="pard2"` |
| online_pard2 sets policy_draft_cat_weighting="true" | PASS | `policy_draft_cat_weighting="true"` |
| remote command exports NRL_VLLM_OMIT_GENERATION_LOGPROBS=false | PASS | `NRL_VLLM_OMIT_GENERATION_LOGPROBS=false` |
| remote command exports NRL_VLLM_DISABLE_LOG_STATS=false | PASS | `NRL_VLLM_DISABLE_LOG_STATS=false` |
| remote command exports NRL_DEBUG_DRAFT_REFIT='${debug_draft_refit}' | PASS | `NRL_DEBUG_DRAFT_REFIT='${debug_draft_refit}'` |
| remote command exports DRAFT_FORMAT='${draft_format}' | PASS | `DRAFT_FORMAT='${draft_format}'` |
| remote command exports PARD_ONLINE_TRAINING='${pard_online_training}' | PASS | `PARD_ONLINE_TRAINING='${pard_online_training}'` |
| remote command exports POLICY_DRAFT_ENABLED='${policy_draft_enabled}' | PASS | `POLICY_DRAFT_ENABLED='${policy_draft_enabled}'` |
| remote command exports POLICY_DRAFT_TYPE=pard2 | PASS | `POLICY_DRAFT_TYPE=pard2` |
| remote command exports POLICY_DRAFT_LOSS='${policy_draft_loss}' | PASS | `POLICY_DRAFT_LOSS='${policy_draft_loss}'` |
| remote command exports POLICY_DRAFT_CAT_WEIGHTING='${policy_draft_cat_weighting}' | PASS | `POLICY_DRAFT_CAT_WEIGHTING='${policy_draft_cat_weighting}'` |
| remote command exports POLICY_DRAFT_MAX_TRAINING_SEQUENCE_LENGTH='${PARD_MAX_TRAINING_SEQUENCE_LENGTH}' | PASS | `POLICY_DRAFT_MAX_TRAINING_SEQUENCE_LENGTH='${PARD_MAX_TRAINING_SEQUENCE_LENGTH}'` |
| remote command exports PARD_TRAINING_MODE=k_slot | PASS | `PARD_TRAINING_MODE=k_slot` |
| remote command exports POLICY_DRAFT_ALLOW_GENERIC_PARD2_FALLBACK=false | PASS | `POLICY_DRAFT_ALLOW_GENERIC_PARD2_FALLBACK=false` |
| remote command exports ENABLE_VLLM_SPECDEC='${enable_vllm_specdec}' | PASS | `ENABLE_VLLM_SPECDEC='${enable_vllm_specdec}'` |
| remote command exports SPECDEC_METHOD='${specdec_method}' | PASS | `SPECDEC_METHOD='${specdec_method}'` |
| remote command exports SOURCE_VLLM_SITE='${source_vllm_site}' | PASS | `SOURCE_VLLM_SITE='${source_vllm_site}'` |
| remote command exports PARD2_OFFICIAL_VLLM_PATCH_DIR='${pard2_patch_dir}' | PASS | `PARD2_OFFICIAL_VLLM_PATCH_DIR='${pard2_patch_dir}'` |
| remote command exports SBATCH_EXTRA_ARGS='${SBATCH_EXTRA_ARGS}' | PASS | `SBATCH_EXTRA_ARGS='${SBATCH_EXTRA_ARGS}'` |
| remote command exports NUM_SPECULATIVE_TOKENS='${num_spec_tokens}' | PASS | `NUM_SPECULATIVE_TOKENS='${num_spec_tokens}'` |
| remote command exports DRAFT_TP=1 | PASS | `DRAFT_TP=1` |
| remote command exports INCLUDE_DRAFT_TP='${include_draft_tp}' | PASS | `INCLUDE_DRAFT_TP='${include_draft_tp}'` |
| remote command exports SPECDEC_PARALLEL_DRAFTING='${parallel_drafting}' | PASS | `SPECDEC_PARALLEL_DRAFTING='${parallel_drafting}'` |
| remote command exports NRL_FORCE_LOCAL_TRANSFORMER_SPEC='${force_local_transformer_spec}' | PASS | `NRL_FORCE_LOCAL_TRANSFORMER_SPEC='${force_local_transformer_spec}'` |
| remote command exports NRL_ACTOR_UV_LOCK_MODE='${actor_uv_lock_mode}' | PASS | `NRL_ACTOR_UV_LOCK_MODE='${actor_uv_lock_mode}'` |
| remote command exports NRL_FORCE_REBUILD_ACTOR_VENVS='${force_rebuild_actor_venvs}' | PASS | `NRL_FORCE_REBUILD_ACTOR_VENVS='${force_rebuild_actor_venvs}'` |
| staging/preflight includes "${ROOT_DIR}/remote_patch_pard2_official/" | PASS | `"${ROOT_DIR}/remote_patch_pard2_official/"` |
| staging/preflight includes "${REMOTE_HOST}:${STAGE_REPO}/experiments/eagle3_qwen3_235b/remote_patches/SpecDec-RL/" | PASS | `"${REMOTE_HOST}:${STAGE_REPO}/experiments/eagle3_qwen3_235b/remote_patches/SpecDec-RL/"` |
| staging/preflight includes "${ROOT_DIR}/experiments/eagle3_online/prepare_pard2_official_vllm_site.sh" | PASS | `"${ROOT_DIR}/experiments/eagle3_online/prepare_pard2_official_vllm_site.sh"` |
| staging/preflight includes "${ROOT_DIR}/scripts/test_pard2_target_feature_alignment.py" | PASS | `"${ROOT_DIR}/scripts/test_pard2_target_feature_alignment.py"` |
| staging/preflight includes "${ROOT_DIR}/scripts/test_vllm_draft_refit_target_proj.py" | PASS | `"${ROOT_DIR}/scripts/test_vllm_draft_refit_target_proj.py"` |
| staging/preflight includes nemo_rl/models/megatron/setup.py | PASS | `nemo_rl/models/megatron/setup.py` |
| staging/preflight includes nemo_rl/models/megatron/community_import.py | PASS | `nemo_rl/models/megatron/community_import.py` |
| staging/preflight includes PYTHONPATH='${SOURCE_VLLM_SITE}':\${PYTHONPATH:-} python3 experiments/eagle3_qwen3_235b/patches/check_pard2_official_patch.py | PASS | `PYTHONPATH='${SOURCE_VLLM_SITE}':\${PYTHONPATH:-} python3 experiments/eagle3_qwen3_235b/patches/check_pard2_official_patch.py` |
| staging/preflight includes test -d '${SOURCE_VLLM_SITE}/vllm' | PASS | `test -d '${SOURCE_VLLM_SITE}/vllm'` |
| staging/preflight includes nemo_rl/algorithms/utils.py | PASS | `nemo_rl/algorithms/utils.py` |
| tracker CSV header exists | PASS | `job_id,method,model,run_id,target_model,draft_model,num_speculative_tokens,max_steps,max_new_tokens,min_tokens,num_prompts,num_generations,train_global_batch_size,base_log_dir` |
| tracker CSV row printf exists | PASS | `%s,%s,qwen8,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n` |
| tracker CSV header/row column count match | PASS | `14 columns` |
| tracker CSV printf placeholder/argument count match | PASS | `13 args` |
| tracker CSV keeps base_log_dir column | PASS | `base_log_dir` |
| OCI overlay setup.py exists | PASS | `/Users/sna/Nemo-RL_Qwen3_Roadmap/.tmp_remote_current_oci/nemo_rl/models/megatron/setup.py` |
| OCI overlay defines gradient-accumulation fusion helper | PASS | `def _set_gradient_accumulation_fusion(model_cfg: Any, enabled: bool) -> None:` |
| OCI overlay sets provider or nested TransformerConfig fusion flag | PASS | `obj.gradient_accumulation_fusion = enabled` |
| OCI overlay descends into provider transformer holders | PASS | `for attr in ("transformer", "transformer_config", "_model_config"):` |
| OCI overlay reapplies fusion flag before policy get_model | PASS | `policy_cfg["megatron_cfg"]["gradient_accumulation_fusion"]` |
| OCI overlay reapplies fusion flag before reference get_model | PASS | `config["megatron_cfg"]["gradient_accumulation_fusion"]` |
| community_import.py source exists | PASS | `/Users/sna/Nemo-RL_Qwen3_Roadmap/experiments/eagle3_qwen3_235b/remote_patches/SpecDec-RL/nemo_rl/models/megatron/community_import.py` |
| community import defines gradient-accumulation fusion helper | PASS | `def _set_gradient_accumulation_fusion(model_provider: Any, enabled: bool) -> None:` |
| community import descends into provider transformer holders | PASS | `for attr in ("transformer", "transformer_config", "_model_config"):` |
| community import applies megatron_cfg fusion flag to provider | PASS | `model_provider, megatron_config["gradient_accumulation_fusion"]` |
| community import finalizes provider | PASS | `model_provider.finalize()` |
| community import reapplies fusion flag after provider finalize | PASS | `model_provider\.finalize\(\).*?_set_gradient_accumulation_fusion\(` |
| algorithm utils.py source exists | PASS | `/Users/sna/Nemo-RL_Qwen3_Roadmap/experiments/eagle3_qwen3_235b/remote_patches/SpecDec-RL/nemo_rl/algorithms/utils.py` |
| advantage helper accepts std_rewards keyword | PASS | `std_rewards: torch.Tensor / None = None,` |
| advantage helper preserves default std source | PASS | `std_source_rewards = rewards if std_rewards is None else std_rewards` |
| advantage helper computes separate std mean | PASS | `prompt_std_mean =` |
| advantage helper std uses std_rewards mean | PASS | `prompt_baseline_square - prompt_std_mean.square()` |
| performance metrics accepts pydantic MasterConfig | PASS | `if hasattr(master_config, "model_dump"):` |
| performance metrics normalizes MasterConfig to dict | PASS | `master_config = master_config.model_dump()` |
