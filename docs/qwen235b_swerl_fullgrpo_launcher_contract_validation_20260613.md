# SWE-RL Full-GRPO SpecDec Launcher Contract Validation

Overall: **PASS**

Launcher: `/Users/sna/Nemo-RL_Qwen3_Roadmap/experiments/eagle3_online/submit_lyris_swerl_qwen235b_fullgrpo_specdec_matrix_20260613.sh`

| check | status | detail |
| --- | --- | --- |
| covers baseline/suffix/PARD/PARD-2/Eagle3 | PASS | `METHODS="${METHODS:-baseline suffix pard pard2 eagle3}"` |
| covers max_steps 10 and 20 | PASS | `MAX_STEPS_SWEEP="${MAX_STEPS_SWEEP:-10 20}"` |
| uses user-requested train replica default | PASS | `NUM_TRAIN_REPLICAS="${NUM_TRAIN_REPLICAS:-1}"` |
| uses user-requested gen/train ratio default | PASS | `GEN_TRAIN_RATIO="${GEN_TRAIN_RATIO:-1}"` |
| uses user-requested vLLM TP default | PASS | `VLLM_TP="${VLLM_TP:-4}"` |
| uses user-requested HYBRIDEP default | PASS | `HYBRIDEP="${HYBRIDEP:-0}"` |
| does not use generation-only skip-training mode | PASS | `SKIP_TRAINING=0 \` |
| does not stop after generation | PASS | `NRL_STOP_AFTER_GENERATION=false \` |
| uses full-GRPO-compatible SpecDec logprob mode | PASS | `SPECDEC_GRPO_MODE=strict_request_logprobs \` |
| keeps generation logprobs for training | PASS | `NRL_VLLM_OMIT_GENERATION_LOGPROBS=false \` |
| passes speculative/reconvert Hydra overrides through EXTRA_ARGS | PASS | `EXTRA_ARGS='${hydra_extra}' \` |
| passes speculative/reconvert Hydra overrides through EXTRA_HYDRA_OVERRIDES | PASS | `EXTRA_HYDRA_OVERRIDES='${hydra_extra}' \` |
| accepts W&B key from caller environment | PASS | `WANDB_API_KEY=*** |
| passes W&B key into launcher environment | PASS | `WANDB_API_KEY=*** \` |
| redacts secrets from launcher diagnostics | PASS | `redact_secrets()` |
| redacts W&B key values in dry-run/error output | PASS | `s/(WANDB_API_KEY=*** |
| redacts Hugging Face token values in dry-run/error output | PASS | `s/(HUGGINGFACE_TOKEN=)[^[:space:]` |
| redacts HF_TOKEN values in dry-run/error output | PASS | `s/(HF_TOKEN=)[^[:space:]` |
| redacts GitHub token values in dry-run/error output | PASS | `s/(GITHUB_TOKEN=*** |
| redacts GitLab token values in dry-run/error output | PASS | `s/(GITLAB_TOKEN=*** |
| targets Rui Qwen235B SWE scale-gen launcher | PASS | `run_grpo_qwen3_235b_swe_scale_gen.sh` |
| has explicit cluster profile selector | PASS | `CLUSTER_PROFILE="${CLUSTER_PROFILE:-lyris}"` |
| defaults Lyris profile to requested SLURM account | PASS | `coreai_dlalgo_llm` |
| defaults OCI-HSG profile to known SSH host | PASS | `oci-hsg-cs-001-vscode-02` |
| defaults OCI-HSG SWE-RL profile to user-approved account | PASS | `nemotron_n3_post` |
| defaults OCI-HSG profile to batch partition | PASS | `DEFAULT_PARTITION="batch"` |
| keeps OCI-HSG jobs within batch partition limit | PASS | `DEFAULT_SBATCH_TIME="04:00:00"` |
| keeps Lyris profile free of invalid GRES requests | PASS | `DEFAULT_SBATCH_GPU_FLAG=""` |
| keeps OCI-HSG profile on explicit GPU GRES | PASS | `DEFAULT_SBATCH_GPU_FLAG='--gres=gpu:${NUM_GPU}'` |
| supports explicit Megatron reconvert prewarm | PASS | `FORCE_RECONVERT_FROM_HF="${FORCE_RECONVERT_FROM_HF:-false}"` |
| can gate launch on converted Megatron cache | PASS | `REQUIRE_MEGATRON_RUN_CONFIG="${REQUIRE_MEGATRON_RUN_CONFIG:-false}"` |
| supports Slurm dependency injection for after-prewarm matrix | PASS | `SBATCH_DEPENDENCY="${SBATCH_DEPENDENCY:-}"` |
| defaults OCI-HSG profile to visible Rui SWE container | PASS | `ruit-swe_bench-6dc8fabea-aarch64-060426-mcore-apptainer.squashfs` |
| writes separate OCI-HSG tracker | PASS | `latest_oci_hsg_swerl_qwen235b_fullgrpo_specdec_20260613_jobs.csv` |
| defaults OCI-HSG profile to verified py312 arctic suffix site | PASS | `vllm-benchmark/.container_cache/arctic-inference-0.1.1` |
| preflights arctic native suffix extension | PASS | `find '${ARCTIC_SITE}/arctic_inference/suffix_decoding' -maxdepth 1 -name '_C*.so' / grep -q .` |
| defaults OCI-HSG profile to verified official PARD-2 vLLM site | PASS | `SpecDec-RL-pard2-official-smoke-20260612/.container_cache/vllm_pard2_official_target_feat` |
| stages patched remote launcher in writable user area | PASS | `REMOTE_LAUNCHER_DIR="${REMOTE_LAUNCHER_DIR:-${HF_HOME%/}/../swerl_fullgrpo_launchers/${RUN_ID}}"` |
| keeps checkpoints/logs in writable user area | PASS | `REMOTE_OUTPUT_ROOT="${REMOTE_OUTPUT_ROOT:-${HF_HOME%/}/..}"` |
| preflights repo ray.sub | PASS | `test -s '${REMOTE_REPO}/ray.sub'` |
| preflights SWE config file | PASS | `test -s '${REMOTE_REPO}/test_assets/qwen-235B/grpo_qwen3_235b_async_swe.yaml'` |
| checks converted Megatron run_config when requested | PASS | `run_config='${HF_HOME}/nemo_rl/'\"\${model_subdir}\"'/iter_0000000/run_config.yaml'` |
| preflights pyxis container image path | PASS | `container_file=\"\${container_image}\"` |
| preflights writable HF cache | PASS | `mkdir -p '${HF_HOME}'` |
| preflights writable remote launcher directory | PASS | `mkdir -p '${REMOTE_LAUNCHER_DIR}'` |
| passes account/partition/GRES/dependency into remote launcher patch | PASS | `awk -v account='${ACCOUNT}' -v partition='${PARTITION}' -v gpu_flag='${SBATCH_GPU_FLAG}' -v dependency='${SBATCH_DEPENDENCY}'` |
| patches hardcoded Rui account line | PASS | `/^SBATCH_ACCOUNT=/ {print "SBATCH_ACCOUNT=\"" account "\""` |
| patches hardcoded Rui partition line | PASS | `/^SBATCH_PARTITION=/ {print "SBATCH_PARTITION=\"" partition "\""` |
| does not turn dry-run GRES display into a command | PASS | `index(\$0, "[DRY_RUN]") && index(\$0, "--gres=gpu:")` |
| patches or removes Rui GRES line per cluster profile | PASS | `index(\$0, "--gres=gpu:") && index(\$0, "NUM_GPU")` |
| patches Rui dependency line for after-prewarm jobs | PASS | `index(\$0, "--dependency=singleton")` |
| propagates source vLLM/arctic site into PYTHONPATH | PASS | `print "  export PYTHONPATH=\"\${SOURCE_VLLM_SITE}:\${PYTHONPATH:-}\""` |
| matches Rui job id pipeline line | PASS | `index(\$0, "ray.sub / tee /dev/stderr / grep -o")` |
| redirects Rui job id file into writable log dir | PASS | `gsub("> latest_235b_scale_gen_job_id.txt", "> \"\${BASE_LOG_DIR}/latest_235b_scale_gen_job_id.txt\"")` |
| reads Rui job id file from writable log dir | PASS | `print "JOB_ID=\"\$(cat \"\${BASE_LOG_DIR}/latest_235b_scale_gen_job_id.txt\")\""` |
| syntax checks patched remote launcher | PASS | `bash -n "\${patched_launcher}"` |
| runs patched launcher against Rui repo assets | PASS | `REPO_ROOT='${REMOTE_REPO}' \` |
| passes writable checkpoint root | PASS | `CHECKPOINT_ROOT='${REMOTE_OUTPUT_ROOT}/results' \` |
| passes writable log root | PASS | `BASE_LOG_DIR='${REMOTE_LOG_ROOT}/${method}_steps${max_steps}' \` |
| passes HF datasets cache | PASS | `HF_DATASETS_CACHE='${HF_HOME}/cache' \` |
| passes HF hub cache | PASS | `HUGGINGFACE_HUB_CACHE='${HF_HOME}/hub' \` |
| passes writable persistent cache override | PASS | `PERSISTENT_CACHE='${PERSISTENT_CACHE}' \` |
| passes partition-appropriate walltime | PASS | `SBATCH_TIME='${SBATCH_TIME}' \` |
| allows SpecDec request logprobs only for SpecDec methods | PASS | `NRL_ALLOW_SPECDEC_REQUEST_LOGPROBS='${request_logprobs}' \` |
| baseline method case exists | PASS | `` |
| baseline sets specdec_method="" | PASS | `specdec_method=""` |
| baseline sets draft_model="" | PASS | `draft_model=""` |
| baseline sets spec_tokens="0" | PASS | `spec_tokens="0"` |
| baseline sets enable_specdec="false" | PASS | `enable_specdec="false"` |
| baseline sets request_logprobs="false" | PASS | `request_logprobs="false"` |
| baseline sets extra_args="" | PASS | `extra_args=""` |
| suffix method case exists | PASS | `` |
| suffix sets specdec_method="suffix" | PASS | `specdec_method="suffix"` |
| suffix sets spec_tokens="${SUFFIX_SPEC_TOKENS}" | PASS | `spec_tokens="${SUFFIX_SPEC_TOKENS}"` |
| suffix sets source_vllm_site="${ARCTIC_SITE}" | PASS | `source_vllm_site="${ARCTIC_SITE}"` |
| suffix sets enable_specdec="true" | PASS | `enable_specdec="true"` |
| suffix sets request_logprobs="true" | PASS | `request_logprobs="true"` |
| suffix sets speculative_config.method=suffix | PASS | `speculative_config.method=suffix` |
| suffix sets suffix_decoding_max_tree_depth | PASS | `suffix_decoding_max_tree_depth` |
| pard method case exists | PASS | `` |
| pard sets specdec_method="draft_model" | PASS | `specdec_method="draft_model"` |
| pard sets draft_model="${PARD_DRAFT_MODEL}" | PASS | `draft_model="${PARD_DRAFT_MODEL}"` |
| pard sets spec_tokens="${PARD_SPEC_TOKENS}" | PASS | `spec_tokens="${PARD_SPEC_TOKENS}"` |
| pard sets draft_tp="${PARD_DRAFT_TP}" | PASS | `draft_tp="${PARD_DRAFT_TP}"` |
| pard sets parallel_drafting="true" | PASS | `parallel_drafting="true"` |
| pard sets speculative_config.method=draft_model | PASS | `speculative_config.method=draft_model` |
| pard sets speculative_config.parallel_drafting=true | PASS | `speculative_config.parallel_drafting=true` |
| pard2 method case exists | PASS | `` |
| pard2 sets specdec_method="pard2" | PASS | `specdec_method="pard2"` |
| pard2 sets draft_model="${PARD2_DRAFT_MODEL}" | PASS | `draft_model="${PARD2_DRAFT_MODEL}"` |
| pard2 sets spec_tokens="${PARD2_SPEC_TOKENS}" | PASS | `spec_tokens="${PARD2_SPEC_TOKENS}"` |
| pard2 sets draft_tp="${PARD2_DRAFT_TP}" | PASS | `draft_tp="${PARD2_DRAFT_TP}"` |
| pard2 sets source_vllm_site="${PARD2_SOURCE_VLLM_SITE}" | PASS | `source_vllm_site="${PARD2_SOURCE_VLLM_SITE}"` |
| pard2 sets speculative_config.method=pard2 | PASS | `speculative_config.method=pard2` |
| pard2 sets speculative_config.parallel_drafting=true | PASS | `speculative_config.parallel_drafting=true` |
| eagle3 method case exists | PASS | `` |
| eagle3 sets specdec_method="eagle3" | PASS | `specdec_method="eagle3"` |
| eagle3 sets draft_model="${EAGLE3_DRAFT_MODEL}" | PASS | `draft_model="${EAGLE3_DRAFT_MODEL}"` |
| eagle3 sets spec_tokens="${EAGLE3_SPEC_TOKENS}" | PASS | `spec_tokens="${EAGLE3_SPEC_TOKENS}"` |
| eagle3 sets draft_tp="${EAGLE3_DRAFT_TP}" | PASS | `draft_tp="${EAGLE3_DRAFT_TP}"` |
| eagle3 sets speculative_config.method=eagle3 | PASS | `speculative_config.method=eagle3` |
| tracker header has expected width | PASS | `17 columns` |
| tracker row includes exp_suffix | PASS | `exp_suffix` |
