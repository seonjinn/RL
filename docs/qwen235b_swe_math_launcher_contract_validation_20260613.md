# Qwen235B SWE/Math Launcher Contract Validation

Overall: **PASS**

| check | status | detail |
| --- | --- | --- |
| SWE batch sweep defaults PARD-2 method to pard2 | PASS | `PARD2_CONFIG_METHOD="${PARD2_CONFIG_METHOD:-pard2}"` |
| SWE batch sweep includes baseline/suffix/PARD/PARD-2/Eagle3 | PASS | `METHODS="${METHODS-baseline suffix pard pard2 eagle3}"` |
| Suffix next pass defaults to K8/K16 | PASS | `K_SWEEP="${K_SWEEP:-8 16}"` |
| Drafter next pass defaults to PARD/PARD-2/Eagle3 | PASS | `METHODS="${METHODS:-pard pard2 eagle3}"` |
| PARD next pass defaults to K9/K11 | PASS | `PARD_K_SWEEP="${PARD_K_SWEEP:-9 11}"` |
| PARD-2 next pass defaults to K9/K11 | PASS | `PARD2_K_SWEEP="${PARD2_K_SWEEP:-9 11}"` |
| Eagle3 next pass defaults to K9/K11 | PASS | `EAGLE3_K_SWEEP="${EAGLE3_K_SWEEP:-9 11}"` |
| Math500 targets Qwen3-235B-A22B | PASS | `MODEL="${MODEL:-Qwen/Qwen3-235B-A22B}"` |
| Math500 uses Math500 prompt JSONL | PASS | `PROMPT_JSONL="${PROMPT_JSONL:-${REMOTE_REPO}/data/math_500_data_prompts_20260612.jsonl}"` |
| Math500 includes baseline/suffix/PARD/PARD-2/Eagle3 | PASS | `METHODS="${METHODS:-baseline suffix pard pard2 eagle3}"` |
| Math500 defaults PARD-2 method to pard2 | PASS | `PARD2_CONFIG_METHOD="${PARD2_CONFIG_METHOD:-pard2}"` |
| base submit helper executes with fake SSH | PASS | `exit 0` |
| base submit tracker output exists | PASS | `/var/folders/fd/n6gx1vhj1gx3c675gj01f7k80000gp/T/qwen235b_launcher_contract_d7qah22x/base_submit_jobs.txt` |
| base submit tracker row count | PASS | `5 rows` |
| base submit tracker row width | PASS | `9 columns` |
| base submit tracker header width | PASS | `job_id,method,model,prompt_offset,prompt_count,isl,osl,batch_sizes,logs_dir` |
| base submit tracker expected methods present | PASS | `baseline,eagle3,pard,pard2,suffix` |
| base submit tracker metadata parsed | PASS | `account,batch_sizes,container_image,csv_header,cuda_launch_blocking,dtype,enable_pard2_method_alias,gpu_memory_utilization,isl,kv_cache_dtype,max_model_len,max_num_batched_tokens,max_num_seqs,methods,model,osl,partition,pp,prompt_count,prompt_jsonl,prompt_offset,remote_host,remote_repo,submitted_at,torch_show_cpp_stacktraces,tp,vllm_allow_long_max_model_len` |
| SWE batch sweep executes with fake SSH | PASS | `exit 0` |
| SWE batch sweep tracker output exists | PASS | `/var/folders/fd/n6gx1vhj1gx3c675gj01f7k80000gp/T/qwen235b_launcher_contract_d7qah22x/batch_sweep.csv` |
| SWE batch sweep tracker row count | PASS | `20 rows` |
| SWE batch sweep tracker row width | PASS | `12 columns` |
| SWE batch sweep tracker header width | PASS | `job_id,dataset,model_group,batch_size,method,model,prompt_offset,prompt_count,isl,osl,batch_sizes,logs_dir` |
| SWE batch sweep tracker expected methods present | PASS | `baseline,eagle3,pard,pard2,suffix` |
| Suffix K sweep executes with fake SSH | PASS | `exit 0` |
| Suffix K sweep tracker output exists | PASS | `/var/folders/fd/n6gx1vhj1gx3c675gj01f7k80000gp/T/qwen235b_launcher_contract_d7qah22x/suffix_k_sweep.csv` |
| Suffix K sweep tracker row count | PASS | `8 rows` |
| Suffix K sweep tracker row width | PASS | `14 columns` |
| Suffix K sweep tracker header width | PASS | `job_id,dataset,model_group,batch_size,method,num_speculative_tokens,model,prompt_offset,prompt_count,isl,osl,batch_sizes,logs_dir,source_tracker` |
| Suffix K sweep tracker expected methods present | PASS | `suffix_k16,suffix_k8` |
| Suffix K sweep method/K columns match | PASS | `suffix_k8/suffix_k16` |
| Drafter K sweep executes with fake SSH | PASS | `exit 0` |
| Drafter K sweep tracker output exists | PASS | `/var/folders/fd/n6gx1vhj1gx3c675gj01f7k80000gp/T/qwen235b_launcher_contract_d7qah22x/drafter_k_sweep.csv` |
| Drafter K sweep tracker row count | PASS | `12 rows` |
| Drafter K sweep tracker row width | PASS | `14 columns` |
| Drafter K sweep tracker header width | PASS | `job_id,dataset,model_group,batch_size,method,num_speculative_tokens,model,prompt_offset,prompt_count,isl,osl,batch_sizes,logs_dir,source_tracker` |
| Drafter K sweep tracker expected methods present | PASS | `eagle3_k11,eagle3_k9,pard2_k11,pard2_k9,pard_k11,pard_k9` |
| Drafter K sweep method/K columns match | PASS | `pard/pard2/eagle3 K9/K11` |
| Math500 launcher executes with fake SSH | PASS | `exit 0` |
| Math500 tracker output exists | PASS | `/var/folders/fd/n6gx1vhj1gx3c675gj01f7k80000gp/T/qwen235b_launcher_contract_d7qah22x/math500_jobs.txt` |
| Math500 tracker row count | PASS | `5 rows` |
| Math500 tracker row width | PASS | `9 columns` |
| Math500 tracker header width | PASS | `job_id,method,model,prompt_offset,prompt_count,isl,osl,batch_sizes,logs_dir` |
| Math500 tracker expected methods present | PASS | `baseline,eagle3,pard,pard2,suffix` |
| Math500 tracker metadata parsed | PASS | `account,batch_sizes,container_image,csv_header,cuda_launch_blocking,dtype,enable_pard2_method_alias,gpu_memory_utilization,isl,kv_cache_dtype,max_model_len,max_num_batched_tokens,max_num_seqs,methods,model,osl,partition,pp,prompt_count,prompt_jsonl,prompt_offset,remote_host,remote_repo,submitted_at,torch_show_cpp_stacktraces,tp,vllm_allow_long_max_model_len` |
| Math500 tracker metadata uses Math500 prompts | PASS | `/tmp/fake-vllm-benchmark/data/math_500_data_prompts_20260612.jsonl` |
| Math500 tracker metadata uses Qwen235B | PASS | `Qwen/Qwen3-235B-A22B` |

## Script Runs

| script | exit |
| --- | ---: |
| `/Users/sna/Nemo-RL_Qwen3_Roadmap/experiments/eagle3_qwen3_235b/submit_lyris_swebench32k_standalone_specdec.sh` | 0 |
| `/Users/sna/Nemo-RL_Qwen3_Roadmap/experiments/eagle3_qwen3_235b/submit_lyris_qwen235b_swebench_osl32k_batch_sweep_20260612.sh` | 0 |
| `/Users/sna/Nemo-RL_Qwen3_Roadmap/experiments/eagle3_qwen3_235b/submit_lyris_qwen235b_swebench_osl32k_suffix_k_sweep_20260613.sh` | 0 |
| `/Users/sna/Nemo-RL_Qwen3_Roadmap/experiments/eagle3_qwen3_235b/submit_lyris_qwen235b_swebench_osl32k_drafter_k_sweep_20260613.sh` | 0 |
| `/Users/sna/Nemo-RL_Qwen3_Roadmap/experiments/eagle3_qwen3_235b/submit_lyris_qwen235b_math500_osl32k_specdec_20260613.sh` | 0 |
