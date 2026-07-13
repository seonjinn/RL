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
| can preserve remote W&B key environment | PASS | `USE_REMOTE_WANDB_API_KEY="${USE_REMOTE_WANDB_API_KEY:-false}"` |
| accepts W&B key from caller environment | PASS | `WANDB_API_KEY="${WANDB_API_KEY:-}"` |
| passes selected W&B key source into launcher environment | PASS | `${wandb_api_key_assignment} \` |
| defaults UV_PYTHON_DOWNLOADS to a valid non-empty value | PASS | `UV_PYTHON_DOWNLOADS="${UV_PYTHON_DOWNLOADS:-auto}"` |
| enables explicit Ray/Python sbatch export by default | PASS | `SBATCH_EXPORT_RAY_ENV="${SBATCH_EXPORT_RAY_ENV:-true}"` |
| has helper for explicit Ray/Python sbatch export | PASS | `append_sbatch_ray_export()` |
| runs explicit Ray/Python sbatch export helper before submit | PASS | `append_sbatch_ray_export` |
| includes RAY_VERSION in explicit sbatch export list | PASS | `    RAY_VERSION \` |
| includes RAY_PYTHON_VERSION in explicit sbatch export list | PASS | `    RAY_PYTHON_VERSION \` |
| includes RAY_PYTHON_SPEC in explicit sbatch export list | PASS | `    RAY_PYTHON_SPEC \` |
| includes RAY_USE_EXISTING_ENV in explicit sbatch export list | PASS | `    RAY_USE_EXISTING_ENV \` |
| includes UV_PYTHON in explicit sbatch export list | PASS | `    UV_PYTHON \` |
| includes UV_PYTHON_DOWNLOADS in explicit sbatch export list | PASS | `    UV_PYTHON_DOWNLOADS; do` |
| uses shell indirect expansion so defaulted env vars are exported | PASS | `env_value="${!env_name-}"` |
| appends explicit Ray export to sbatch extra args | PASS | `SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:+${SBATCH_EXTRA_ARGS} }${export_arg}"` |
| keeps default dry-run output out of latest job trackers | PASS | `OUT="${ROOT_DIR}/tmp/$(basename "${DEFAULT_OUT%.csv}")_dryrun.csv"` |
| creates dry-run output directory | PASS | `mkdir -p "$(dirname "${OUT}")"` |
| keeps Lyris compatible with MFA ControlMaster sessions | PASS | `DEFAULT_SSH_DISABLE_CONTROLMASTER="false"` |
| keeps OCI-HSG long-command submissions off ControlMaster by default | PASS | `DEFAULT_SSH_DISABLE_CONTROLMASTER="true"` |
| allows profile-specific SSH ControlMaster behavior | PASS | `SSH_DISABLE_CONTROLMASTER="${SSH_DISABLE_CONTROLMASTER:-${DEFAULT_SSH_DISABLE_CONTROLMASTER}}"` |
| uses noninteractive SSH base options | PASS | `SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10)` |
| can disable ControlMaster when requested | PASS | `SSH_OPTS+=(-S none)` |
| redacts secrets from launcher diagnostics | PASS | `redact_secrets()` |
| redacts W&B key values in dry-run/error output | PASS | `s/(WANDB_API_KEY=)[^[:space:]` |
| redacts Hugging Face token values in dry-run/error output | PASS | `s/(HUGGINGFACE_TOKEN=)[^[:space:]` |
| redacts HF_TOKEN values in dry-run/error output | PASS | `s/(HF_TOKEN=)[^[:space:]` |
| redacts GitHub token values in dry-run/error output | PASS | `s/(GITHUB_TOKEN=)[^[:space:]` |
| redacts GitLab token values in dry-run/error output | PASS | `s/(GITLAB_TOKEN=)[^[:space:]` |
| targets Rui Qwen235B SWE scale-gen launcher | PASS | `run_grpo_qwen3_235b_swe_scale_gen.sh` |
| has explicit cluster profile selector | PASS | `CLUSTER_PROFILE="${CLUSTER_PROFILE:-lyris}"` |
| defaults Lyris profile to requested SLURM account | PASS | `coreai_dlalgo_llm` |
| defaults to a writable persistent cache | PASS | `qwen235b_swerl_specdec` |
| keeps Lyris jobs within gb200 partition limit | PASS | `DEFAULT_SBATCH_TIME="05:00:00"` |
| defaults OCI-HSG profile to known SSH host | PASS | `oci-hsg-cs-001-vscode-02` |
| defaults OCI-HSG SWE-RL profile to user-approved account | PASS | `nemotron_n3_post` |
| defaults OCI-HSG profile to batch partition | PASS | `DEFAULT_PARTITION="batch"` |
| keeps OCI-HSG jobs within batch partition limit | PASS | `DEFAULT_SBATCH_TIME="04:00:00"` |
| keeps Lyris profile free of invalid GRES requests | PASS | `DEFAULT_SBATCH_GPU_FLAG=""` |
| keeps OCI-HSG profile on explicit GPU GRES | PASS | `DEFAULT_SBATCH_GPU_FLAG='--gres=gpu:${NUM_GPU}'` |
| supports explicit Megatron reconvert prewarm | PASS | `FORCE_RECONVERT_FROM_HF="${FORCE_RECONVERT_FROM_HF:-false}"` |
| can gate launch on converted Megatron cache | PASS | `REQUIRE_MEGATRON_RUN_CONFIG="${REQUIRE_MEGATRON_RUN_CONFIG:-false}"` |
| supports Slurm dependency injection for after-prewarm matrix | PASS | `SBATCH_DEPENDENCY="${SBATCH_DEPENDENCY:-}"` |
| supports extra sbatch options such as node exclusion | PASS | `SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:-}"` |
| defaults OCI-HSG profile to visible Rui SWE container | PASS | `ruit-swe_bench-6dc8fabea-aarch64-060426-mcore-apptainer.squashfs` |
| writes separate OCI-HSG tracker | PASS | `latest_oci_hsg_swerl_qwen235b_fullgrpo_specdec_20260613_jobs.csv` |
| defaults OCI-HSG profile to verified py312 arctic suffix site | PASS | `vllm-benchmark/.container_cache/arctic-inference-0.1.1` |
| preflights arctic native suffix extension | PASS | `find '${ARCTIC_SITE}/arctic_inference/suffix_decoding' -maxdepth 1 -name '_C*.so' / grep -q .` |
| defaults Lyris profile to patched Python-overlay PARD-2 vLLM site | PASS | `vllm_pard2_official_target_feat_pyoverlay_lyris_nostable_nofp4out_basefa_nofp4fusion_20260614` |
| defaults OCI-HSG profile to patched Python-overlay PARD-2 vLLM site | PASS | `vllm_pard2_official_target_feat_pyoverlay_nostable_nofp4out_basefa_nofp4fusion_20260614` |
| rejects compiled official PARD-2 vLLM sites by default | PASS | `DEFAULT_PARD2_REJECT_COMPILED_C="true"` |
| allows PARD-2 compiled-site preflight override | PASS | `PARD2_REJECT_COMPILED_C="${PARD2_REJECT_COMPILED_C:-${DEFAULT_PARD2_REJECT_COMPILED_C}}"` |
| preflights PARD-2 source site does not contain incompatible compiled extension | PASS | `find '${PARD2_SOURCE_VLLM_SITE}/vllm' -maxdepth 1 -name '_C*.so' / grep -q .` |
| reports explicit PARD-2 source-site fallback error | PASS | `PARD2_SOURCE_VLLM_SITE must be the patched Python overlay` |
| stages patched remote launcher in writable user area | PASS | `REMOTE_LAUNCHER_DIR="${REMOTE_LAUNCHER_DIR:-${HF_HOME%/}/../swerl_fullgrpo_launchers/${RUN_ID}}"` |
| keeps checkpoints/logs in writable user area | PASS | `REMOTE_OUTPUT_ROOT="${REMOTE_OUTPUT_ROOT:-${HF_HOME%/}/..}"` |
| preflights repo ray.sub | PASS | `test -s '${REMOTE_REPO}/ray.sub'` |
| preflights SWE config file | PASS | `test -s '${REMOTE_REPO}/test_assets/qwen-235B/grpo_qwen3_235b_async_swe.yaml'` |
| preflights remote SWE Gym app.py | PASS | `gym_swe_agent_app='${REMOTE_REPO}/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/app.py'` |
| requires OpenHands PYTHONPATH patch in remote SWE Gym app.py | PASS | `PYTHONPATH=/openhands_setup/OpenHands` |
| requires poetry shebang miniforge bind patch in remote SWE Gym app.py | PASS | `poetry_shebang` |
| requires shebang miniforge mount destination in remote SWE Gym app.py | PASS | `shebang_python.parent.parent` |
| checks converted Megatron run_config when requested | PASS | `run_config='${HF_HOME}/nemo_rl/'\"\${model_subdir}\"'/iter_0000000/run_config.yaml'` |
| preflights pyxis container image path | PASS | `container_file=\"\${container_image}\"` |
| preflights writable HF cache | PASS | `mkdir -p '${HF_HOME}'` |
| preflights writable remote launcher directory | PASS | `mkdir -p '${REMOTE_LAUNCHER_DIR}'` |
| passes account/partition/GRES/dependency/extra args into remote launcher patch | PASS | `awk -v account='${ACCOUNT}' -v partition='${PARTITION}' -v gpu_flag='${SBATCH_GPU_FLAG}' -v dependency='${SBATCH_DEPENDENCY}' -v extra_sbatch_args='${SBATCH_EXTRA_ARGS}'` |
| patches hardcoded Rui account line | PASS | `/^SBATCH_ACCOUNT=/ {print "SBATCH_ACCOUNT=\"" account "\""` |
| patches hardcoded Rui partition line | PASS | `/^SBATCH_PARTITION=/ {print "SBATCH_PARTITION=\"" partition "\""` |
| patches node-local uv cache line | PASS | `/^export UV_CACHE_DIR=\/tmp\/uv_cache/` |
| defaults remote launcher to method/step-scoped uv cache | PASS | `print "UV_CACHE_SCOPE=\"\${UV_CACHE_SCOPE:-method_steps}\""` |
| supports stable method/step uv cache keys across retries | PASS | `print "  method_steps) uv_cache_key=\"\${UV_CACHE_KEY:-\${EXP_SUFFIX:-manual}}\" ;;"` |
| uses scoped persistent uv cache | PASS | `print "  export UV_CACHE_DIR=\"\${PERSISTENT_CACHE}/uv_cache/\${uv_cache_key}\""` |
| uses shared persistent pip cache | PASS | `print "  export PIP_CACHE_DIR=\"\${PERSISTENT_CACHE}/pip_cache\""` |
| uses scoped persistent torch extension cache | PASS | `print "  export TORCH_EXTENSIONS_DIR=\"\${PERSISTENT_CACHE}/torch_extensions/\${uv_cache_key}\""` |
| patches Rui uv lock timeout line | PASS | `index(\$0, "UV_LOCK_TIMEOUT=")` |
| raises Rui uv lock timeout | PASS | `gsub(/UV_LOCK_TIMEOUT=[0-9]+/, "UV_LOCK_TIMEOUT=\${UV_LOCK_TIMEOUT:-7200}")` |
| does not turn dry-run GRES display into a command | PASS | `index(\$0, "[DRY_RUN]") && index(\$0, "--gres=gpu:")` |
| patches or removes Rui GRES line per cluster profile | PASS | `index(\$0, "--gres=gpu:") && index(\$0, "NUM_GPU")` |
| patches Rui dependency line for after-prewarm jobs | PASS | `index(\$0, "--dependency=singleton")` |
| inserts optional extra sbatch args | PASS | `extra_sbatch_args != ""` |
| propagates source vLLM/arctic site into PYTHONPATH | PASS | `print "  export PYTHONPATH=\"\${SOURCE_VLLM_SITE}:\${PYTHONPATH:-}\""` |
| injects NemoGym source root into launcher and actor envs | PASS | `NRL_NEMO_GYM_SOURCE_ROOT` |
| points NemoGym source root at staged Gym workspace | PASS | `3rdparty/Gym-workspace/Gym` |
| copies NemoGym/Ray/UV env vars into the Ray driver command | PASS | `for env_name in PYTHONPATH NEMO_RL_VENV_DIR NRL_FORCE_REBUILD_VENVS NRL_FORCE_REBUILD_ACTOR_VENVS NRL_ACTOR_VENV_CACHE_SUFFIX NRL_ACTOR_UV_LOCK_MODE NEMO_RL_NEMO_GYM_EXECUTABLE_SYSTEM NRL_NEMO_GYM_SKIP_PACKAGE_BUILD NRL_NEMO_GYM_SOURCE_ROOT RAY_VERSION RAY_PYTHON_VERSION RAY_PYTHON_SPEC RAY_USE_EXISTING_ENV RAY_VENV RAY_STATUS_VENV RAY_CGRAPH_GET_TIMEOUT RAY_CGRAPH_get_timeout UV_PYTHON UV_PYTHON_INSTALL_DIR UV_PYTHON_DOWNLOADS` |
| quotes copied env vars before prepending to COMMAND | PASS | `runtime_env_prefix=\"\${runtime_env_prefix} \${env_name}=\${env_value_quoted}\"` |
| prepends copied env vars to Ray driver command | PASS | `export COMMAND=\"\${runtime_env_prefix# } \${COMMAND}\"` |
| matches Rui job id pipeline line | PASS | `index(\$0, "ray.sub / tee /dev/stderr / grep -o")` |
| redirects Rui job id file into writable log dir | PASS | `gsub("> latest_235b_scale_gen_job_id.txt", "> \"\${BASE_LOG_DIR}/latest_235b_scale_gen_job_id.txt\"")` |
| reads Rui job id file from writable log dir | PASS | `print "JOB_ID=\"\$(cat \"\${BASE_LOG_DIR}/latest_235b_scale_gen_job_id.txt\")\""` |
| syntax checks patched remote launcher | PASS | `bash -n "\${patched_launcher}"` |
| runs patched launcher against Rui repo assets | PASS | `REPO_ROOT='${REMOTE_REPO}' \` |
| passes writable checkpoint root | PASS | `CHECKPOINT_ROOT='${REMOTE_OUTPUT_ROOT}/results' \` |
| passes writable log root | PASS | `BASE_LOG_DIR='${REMOTE_LOG_ROOT}/${method}_steps${max_steps}' \` |
| passes HF datasets cache | PASS | `HF_DATASETS_CACHE='${HF_HOME}/cache' \` |
| passes HF hub cache | PASS | `HUGGINGFACE_HUB_CACHE='${HF_HOME}/hub' \` |
| defaults persistent cache to writable cluster path | PASS | `PERSISTENT_CACHE="${PERSISTENT_CACHE:-${DEFAULT_PERSISTENT_CACHE}}"` |
| passes writable persistent cache override | PASS | `PERSISTENT_CACHE='${PERSISTENT_CACHE}' \` |
| passes actor venv cache suffix for fresh NemoGym actor envs | PASS | `NRL_ACTOR_VENV_CACHE_SUFFIX='${NRL_ACTOR_VENV_CACHE_SUFFIX}' \` |
| passes NemoGym package-build skip override | PASS | `NRL_NEMO_GYM_SKIP_PACKAGE_BUILD='${NRL_NEMO_GYM_SKIP_PACKAGE_BUILD}' \` |
| passes Ray version override | PASS | `RAY_VERSION='${RAY_VERSION:-}' \` |
| passes Ray Python version override | PASS | `RAY_PYTHON_VERSION='${RAY_PYTHON_VERSION:-}' \` |
| passes Ray Python spec override | PASS | `RAY_PYTHON_SPEC='${RAY_PYTHON_SPEC:-}' \` |
| passes Ray existing-env override | PASS | `RAY_USE_EXISTING_ENV='${RAY_USE_EXISTING_ENV:-}' \` |
| passes UV Python override | PASS | `UV_PYTHON='${UV_PYTHON}' \` |
| passes UV Python downloads override | PASS | `UV_PYTHON_DOWNLOADS='${UV_PYTHON_DOWNLOADS}' \` |
| defaults uv cache scope to stable method/step cache reuse | PASS | `UV_CACHE_SCOPE="${UV_CACHE_SCOPE:-method_steps}"` |
| passes stable method/step uv cache key | PASS | `UV_CACHE_KEY='${method}_steps${max_steps}' \` |
| raises uv lock timeout above native TransformerEngine build time | PASS | `UV_LOCK_TIMEOUT="${UV_LOCK_TIMEOUT:-7200}"` |
| defaults Ray compiled-DAG timeout to non-empty value | PASS | `RAY_CGRAPH_GET_TIMEOUT="${RAY_CGRAPH_GET_TIMEOUT:-7200}"` |
| propagates lowercase Ray compiled-DAG timeout from uppercase default | PASS | `RAY_CGRAPH_get_timeout="${RAY_CGRAPH_get_timeout:-${RAY_CGRAPH_GET_TIMEOUT}}"` |
| defaults vLLM Ray compiled-DAG channel to valid auto value | PASS | `VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE="${VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE:-auto}"` |
| defaults vLLM Ray compiled-DAG overlap flag to valid integer false | PASS | `VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM="${VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM:-0}"` |
| passes uv cache scope override | PASS | `UV_CACHE_SCOPE='${UV_CACHE_SCOPE}' \` |
| passes uv lock timeout override | PASS | `UV_LOCK_TIMEOUT='${UV_LOCK_TIMEOUT}' \` |
| passes non-empty uppercase Ray compiled-DAG timeout | PASS | `RAY_CGRAPH_GET_TIMEOUT='${RAY_CGRAPH_GET_TIMEOUT}' \` |
| passes non-empty lowercase Ray compiled-DAG timeout | PASS | `RAY_CGRAPH_get_timeout='${RAY_CGRAPH_get_timeout}' \` |
| passes valid vLLM Ray compiled-DAG channel | PASS | `VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE='${VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE}' \` |
| passes valid vLLM Ray compiled-DAG overlap flag | PASS | `VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM='${VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM}' \` |
| passes partition-appropriate walltime | PASS | `SBATCH_TIME='${SBATCH_TIME}' \` |
| allows SpecDec request logprobs only for SpecDec methods | PASS | `NRL_ALLOW_SPECDEC_REQUEST_LOGPROBS='${request_logprobs}' \` |
| escapes staged SIF formatter for Hydra | PASS | `swe_rebench_formatter_for_hydra` |
| preserves staged SIF formatter quotes through bash -c | PASS | `container_formatter=[\\\"${swe_rebench_formatter_for_hydra}\\\"]` |
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
