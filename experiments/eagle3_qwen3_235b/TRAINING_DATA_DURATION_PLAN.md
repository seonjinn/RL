# Qwen3-235B Eagle3 Training Data And Duration Plan

Last updated: 2026-05-23 01:14 PDT

## Live Gate

Qwen3-235B Eagle3 draft training has not started yet. The active work is still
the Qwen3-235B SWE rollout-capture smoke. No `train_data_step*.jsonl` or
materialized conversation corpus has been produced yet:

```text
rollout smoke job: 2861605
job name: qwen3-235b-swe-rollout-vllm0102src-swegym-fixed-instancedict-smoke1step
state at 2026-05-22 19:01 PDT: CANCELLED by 150081 before allocation
squeue start estimate: n/a
shape: NUM_GPU=4, NUM_NODES=16, NUM_GEN_NODES=4
train/val data: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/swegym_example_for_sweagent_with_instance_dict.jsonl
output target: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations_vllm0102src_swegym_fixed_instancedict_smoke.jsonl

fallback smoke job: 2863716
job name: qwen3-235b-swe-rollout-capture-balanced24n4g
state: FAILED, ExitCode=1:0, elapsed=00:04:07
start time: 2026-05-22T19:01:00
end time: 2026-05-22T19:05:07
allocated shape: 24 nodes, 96 GPUs
shape: NUM_GPU=4, NUM_NODES=24, NUM_GEN_NODES=8
train/val data: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/swegym_example_for_sweagent_with_instance_dict.jsonl
output target: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations_balanced24n4g.jsonl
failure: Ray runtime-env agent timeout on nvl72089-T06 / 10.109.17.223; no rollout JSONL produced

retry smoke job: 2864216
job name: qwen3-235b-swe-rollout-capture-balanced24n4g-excl-t06
state: FAILED, ExitCode=1:0, elapsed=00:02:47
start time: 2026-05-22T19:11:28
end time: 2026-05-22T19:14:15
allocated shape: 24 nodes, 96 GPUs
shape: NUM_GPU=4, NUM_NODES=24, NUM_GEN_NODES=8
exclude: nvl72089-T06
failure: Ray CoreWorker gRPC bind failed on head nvl72007-T01, port 10002 already in use; no rollout JSONL produced

retry smoke job: 2866525
job name: qwen3-235b-swe-rollout-capture-balanced24n4g-headport-excl-t06-t01
state: FAILED, ExitCode=1:0, elapsed=00:06:14
start time: 2026-05-22T22:50:23
end time: 2026-05-22T22:56:37
allocated shape: 24 nodes, 96 GPUs
shape: NUM_GPU=4, NUM_NODES=24, NUM_GEN_NODES=8
exclude: nvl72089-T06,nvl72007-T01
train/val data: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/swegym_example_for_sweagent_with_instance_dict.jsonl
output target: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations_balanced24n4g_headport_excl_t06_t01.jsonl
failure: Megatron Qwen3MoE provider missing finalize(); no rollout JSONL produced

compat probe job: 2866588
job name: q235b-megatron-compat-probe
state: COMPLETED, ExitCode=0:0, elapsed=00:01:44
result: PASS, qwen3_moe_provider_smoke.has_finalize=True, errors=[]
report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/megatron_compat_probe_finalize.md

retry smoke job: 2866601
job name: qwen3-235b-swe-rollout-capture-balanced24n4g-headport-finalize-excl-t06-t01
state: FAILED, ExitCode=1:0, elapsed=00:06:16
start time: 2026-05-22T23:00:19
end time: 2026-05-22T23:06:35
allocated shape: 24 nodes, 96 GPUs
shape: NUM_GPU=4, NUM_NODES=24, NUM_GEN_NODES=8
exclude: nvl72089-T06,nvl72007-T01
train/val data: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/swegym_example_for_sweagent_with_instance_dict.jsonl
output target: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations_balanced24n4g_headport_finalize_excl_t06_t01.jsonl
watcher PID: 2755376
watcher log: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/watch_rollout_capture_2866601_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_excl_t06_t01_swegym_smoke.log
state report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/rollout_capture_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_excl_t06_t01_2866601_swegym_state_advance.json
failure: Megatron saw expert_tensor_parallel_size=4, so decoder world_size 64 was not divisible by expert_tensor_model_pipeline_parallel size 256; no rollout JSONL produced

retry smoke job: 2866688
job name: qwen3-235b-swe-rollout-capture-balanced24n4g-headport-finalize-etp1-excl-t06-t01
state: FAILED, ExitCode=1:0, elapsed=00:06:21
start time: 2026-05-22T23:09:59
end time: 2026-05-22T23:16:20
allocated shape: 24 nodes, 96 GPUs
shape: NUM_GPU=4, NUM_NODES=24, NUM_GEN_NODES=8, TP=4, ETP=1, EP=16, CP=1, PP=4
exclude: nvl72089-T06,nvl72007-T01
train/val data: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/swegym_example_for_sweagent_with_instance_dict.jsonl
output target: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations_balanced24n4g_headport_finalize_etp1_excl_t06_t01.jsonl
watcher PID: 2834941
watcher log: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/watch_rollout_capture_2866688_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etp1_excl_t06_t01_swegym_smoke.log
state report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/rollout_capture_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etp1_excl_t06_t01_2866688_swegym_state_advance.json
failure: Hydra carried expert_tensor_parallel_size=1, but Bridge finalize recoupled provider ETP to TP before Megatron-Core initialize; no rollout JSONL produced

compat probe job: 2866747
job name: q235b-megatron-compat-probe
state: COMPLETED, ExitCode=0:0, elapsed=00:01:57
result: PASS, qwen3_moe_provider_smoke.expert_tensor_parallel_size_after_finalize=1, errors=[]
report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/megatron_compat_probe_etp_finalize.md

retry smoke job: 2866765
job name: qwen3-235b-swe-rollout-capture-balanced24n4g-headport-finalize-etp1-preserve-excl-t06-t01
state: CANCELLED by 150081, elapsed=00:05:50
start time: 2026-05-22T23:20:45
end time: 2026-05-22T23:26:35
allocated/requested shape: 24 nodes, 96 GPUs
shape: NUM_GPU=4, NUM_NODES=24, NUM_GEN_NODES=8, TP=4, ETP=1, EP=16, CP=1, PP=4
exclude: nvl72089-T06,nvl72007-T01
train/val data: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/swegym_example_for_sweagent_with_instance_dict.jsonl
output target: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations_balanced24n4g_headport_finalize_etp1_preserve_excl_t06_t01.jsonl
watcher PID: 2937984
watcher log: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/watch_rollout_capture_2866765_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etp1_preserve_excl_t06_t01_swegym_smoke.log
state report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/rollout_capture_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etp1_preserve_excl_t06_t01_2866765_swegym_state_advance.json
failure: source inspection showed Bridge initialize_model_parallel ignores provider.expert_tensor_parallel_size unless passed through kwargs; job was cancelled after same product-256 error appeared

source inspection job: 2866786
job name: q235b-bridge-src-inspect
state: COMPLETED, ExitCode=0:0, elapsed=00:01:12
result: Megatron-Bridge model_provider_mixin.initialize_model_parallel accepts **model_parallel_kwargs and does not pass provider.expert_tensor_parallel_size by default
report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/bridge_src_inspect_2866786.out

previous retry smoke job: 2866789
job name: qwen3-235b-swe-rollout-capture-balanced24n4g-headport-finalize-etpkwarg-excl-t06-t01
state: FAILED, ExitCode=1:0, elapsed=00:06:02
start time: 2026-05-22T23:27:44
allocated/requested shape: 24 nodes, 96 GPUs
shape: NUM_GPU=4, NUM_NODES=24, NUM_GEN_NODES=8, TP=4, ETP=1, EP=16, CP=1, PP=4
exclude: nvl72089-T06,nvl72007-T01
train/val data: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/swegym_example_for_sweagent_with_instance_dict.jsonl
output target: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations_balanced24n4g_headport_finalize_etpkwarg_excl_t06_t01.jsonl
watcher PID: 2994448
watcher log: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/watch_rollout_capture_2866789_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etpkwarg_excl_t06_t01_swegym_smoke.log
state report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/rollout_capture_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etpkwarg_excl_t06_t01_2866789_swegym_state_advance.json
failure: passed the prior ETP/world-size error, initialized 64 workers, and started vLLM generation; then failed because this Bridge provider exposes provide_models(), not provide_distributed_model()

retry smoke job: 2866871
job name: qwen3-235b-swe-rollout-capture-balanced24n4g-headport-finalize-etpkwarg-providemodels-excl-t06-t01
state at 2026-05-22 23:50 PDT: FAILED
allocated/requested shape: 24 nodes, 96 GPUs
shape: NUM_GPU=4, NUM_NODES=24, NUM_GEN_NODES=8, TP=4, ETP=1, EP=16, CP=1, PP=4
exclude: nvl72089-T06,nvl72007-T01
train/val data: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/swegym_example_for_sweagent_with_instance_dict.jsonl
output target: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations_balanced24n4g_headport_finalize_etpkwarg_providemodels_excl_t06_t01.jsonl
watcher PID: 3119496
watcher log: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/watch_rollout_capture_2866871_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etpkwarg_providemodels_excl_t06_t01_swegym_smoke.log
state report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/rollout_capture_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etpkwarg_providemodels_excl_t06_t01_2866871_swegym_state_advance.json
failure: passed Ray startup, 64 Megatron workers, 32 vLLM workers, vLLM model load, and HF shard fetch; then failed in Megatron-Bridge HF-to-TP weight scatter with NCCL duplicate GPU detection because ranks on the same node were still using cuda:0

retry smoke job: 2867033
job name: qwen3-235b-swe-rollout-capture-balanced24n4g-headport-finalize-etpkwarg-providemodels-cudadev-excl-t06-t01
state: FAILED, ExitCode=1:0, elapsed=00:06:18
start time: 2026-05-22T23:59:02
end time: 2026-05-23T00:05:20
allocated/requested shape: 24 nodes, 96 GPUs
shape: NUM_GPU=4, NUM_NODES=24, NUM_GEN_NODES=8, TP=4, ETP=1, EP=16, CP=1, PP=4
exclude: nvl72089-T06,nvl72007-T01
train/val data: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/swegym_example_for_sweagent_with_instance_dict.jsonl
output target: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations_balanced24n4g_headport_finalize_etpkwarg_providemodels_cudadev_excl_t06_t01.jsonl
watcher PID: 3325121
watcher log: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/watch_rollout_capture_2867033_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etpkwarg_providemodels_cudadev_excl_t06_t01_swegym_smoke.log
state report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/rollout_capture_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etpkwarg_providemodels_cudadev_excl_t06_t01_2867033_swegym_state_advance.json
failure: passed CUDA device binding, Ray/vLLM startup, Qwen3-235B vLLM load, and HF shard fetch; then failed in Megatron-Bridge HF-to-Megatron import because old TPAwareMapping treated TELayerNormColumnParallelLinear.layer_norm_weight as column-parallel instead of replicated

compat probe job: 2867247
job name: q235b-megatron-compat-probe
state: COMPLETED, ExitCode=0:0, elapsed=00:01:42
result: PASS, qwen3_moe_mapping_registry.layernorm_entries[0].type=AutoMapping, errors=[]
report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/megatron_compat_probe_automapping_fallback.md

retry smoke job: 2867262
job name: qwen3-235b-swe-rollout-capture-balanced24n4g-headport-finalize-etpkwarg-providemodels-cudadev-automapping-excl-t06-t01
state: FAILED, ExitCode=1:0, elapsed=00:06:18
start time: 2026-05-23T00:21:30
end time: 2026-05-23T00:27:48
allocated/requested shape: 24 nodes, 96 GPUs
shape: NUM_GPU=4, NUM_NODES=24, NUM_GEN_NODES=8, TP=4, ETP=1, EP=16, CP=1, PP=4
exclude: nvl72089-T06,nvl72007-T01
train/val data: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/swegym_example_for_sweagent_with_instance_dict.jsonl
output target: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations_balanced24n4g_headport_finalize_etpkwarg_providemodels_cudadev_automapping_excl_t06_t01.jsonl
watcher PID: 3598143
watcher log: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/watch_rollout_capture_2867262_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etpkwarg_providemodels_cudadev_automapping_excl_t06_t01_swegym_smoke.log
state report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/rollout_capture_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etpkwarg_providemodels_cudadev_automapping_excl_t06_t01_2867262_swegym_state_advance.json
failure: fallback AutoMapping selected replicated path, but old ReplicatedMapping still used megatron_module.weight for non-source TP ranks and CPU tensors for broadcast; no rollout JSONL produced

retry smoke job: 2867356
job name: qwen3-235b-swe-rollout-capture-balanced24n4g-headport-finalize-etpkwarg-providemodels-cudadev-automaprepbcast-excl-t06-t01
state: CANCELLED by 150081, elapsed=00:05:37
allocated/requested shape: 24 nodes, 96 GPUs
shape: NUM_GPU=4, NUM_NODES=24, NUM_GEN_NODES=8, TP=4, ETP=1, EP=16, CP=1, PP=4
exclude: nvl72089-T06,nvl72007-T01
train/val data: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/swegym_example_for_sweagent_with_instance_dict.jsonl
output target: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations_balanced24n4g_headport_finalize_etpkwarg_providemodels_cudadev_automaprepbcast_excl_t06_t01.jsonl
watcher PID: 3745453
watcher log: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/watch_rollout_capture_2867356_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etpkwarg_providemodels_cudadev_automaprepbcast_excl_t06_t01_swegym_smoke.log
state report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/rollout_capture_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etpkwarg_providemodels_cudadev_automaprepbcast_excl_t06_t01_2867356_swegym_state_advance.json
failure: fallback AutoMapping selected the actual target tensor shape, but some Megatron target parameters were still CPU tensors during NCCL broadcast; no rollout JSONL produced

compat probe job: 2867422
job name: q235b-megatron-compat-probe
state: COMPLETED, ExitCode=0:0, elapsed=00:01:50
result: PASS, qwen3_moe_bridge_registered=True, qwen3_moe_provider_smoke.provider=Qwen3MoEModelProvider, errors=[]
report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/megatron_compat_probe.json

retry smoke job: 2867545
job name: qwen3-235b-swe-rollout-capture-balanced24n4g-headport-finalize-etpkwarg-providemodels-cudadev-automaprepbcastcuda-excl-t06-t01
state: FAILED, ExitCode=1:0, elapsed=00:06:09
start time: 2026-05-23T00:44:36
end time: 2026-05-23T00:50:45
allocated/requested shape: 24 nodes, 96 GPUs
shape: NUM_GPU=4, NUM_NODES=24, NUM_GEN_NODES=8, TP=4, ETP=1, EP=16, CP=1, PP=4
exclude: nvl72089-T06,nvl72007-T01
train/val data: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/swegym_example_for_sweagent_with_instance_dict.jsonl
output target: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations_balanced24n4g_headport_finalize_etpkwarg_providemodels_cudadev_automaprepbcastcuda_excl_t06_t01.jsonl
watcher PID: 3855439
watcher log: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/watch_rollout_capture_2867545_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etpkwarg_providemodels_cudadev_automaprepbcastcuda_excl_t06_t01_swegym_smoke.log
state report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/rollout_capture_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etpkwarg_providemodels_cudadev_automaprepbcastcuda_excl_t06_t01_2867545_swegym_state_advance.json
failure: passed the previous CPU/NCCL fallback path and vLLM loaded the 235B model; then old TPAwareMapping could not infer TEColumnParallelGroupedLinear for decoder.layers.0.mlp.experts.linear_fc1.weight0; no rollout JSONL produced

compat probe job: 2867656
job name: q235b-megatron-compat-probe
state: COMPLETED, ExitCode=0:0, elapsed=00:01:36
result: PASS, tpaware_grouped_linear_detection={TEColumnParallelGroupedLinear: column, TERowParallelGroupedLinear: row}, errors=[]
report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/megatron_compat_probe.json

retry smoke job: 2867662
job name: qwen3-235b-swe-rollout-capture-balanced24n4g-headport-finalize-etpkwarg-providemodels-cudadev-automapgroupedlin-excl-t06-t01
state: FAILED, ExitCode=1:0, elapsed=00:05:58
start time: 2026-05-23T01:07:06
end time: 2026-05-23T01:13:04
allocated/requested shape: 24 nodes, 96 GPUs
shape: NUM_GPU=4, NUM_NODES=24, NUM_GEN_NODES=8, TP=4, ETP=1, EP=16, CP=1, PP=4
exclude: nvl72089-T06,nvl72007-T01
train/val data: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/swegym_example_for_sweagent_with_instance_dict.jsonl
output target: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations_balanced24n4g_headport_finalize_etpkwarg_providemodels_cudadev_automapgroupedlin_excl_t06_t01.jsonl
watcher PID: 4017490
watcher log: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/watch_rollout_capture_2867662_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etpkwarg_providemodels_cudadev_automapgroupedlin_excl_t06_t01_swegym_smoke.log
state report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/rollout_capture_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etpkwarg_providemodels_cudadev_automapgroupedlin_excl_t06_t01_2867662_swegym_state_advance.json
failure: grouped module type detection worked, but old ColumnParallelMapping expected .weight while TEColumnParallelGroupedLinear exposes weight0/weight1; no rollout JSONL produced

compat probe job: 2867766
job name: q235b-megatron-compat-probe
state at 2026-05-23 01:14 PDT: submitted, result pending because local SSH/DNS lookup for oci-hsg-cs-001-vscode-02 is temporarily failing
expected result: grouped_linear_temporary_weight_attr.has_weight=true and weight_is_weight0=true
```

These jobs are runtime/capture proofs only. They use five SWE-Gym rows and must not
be promoted to the canonical Eagle3 training corpus. They should prove whether the
Qwen3-235B RL path writes `train_data_step*.jsonl` and whether those logs can be
normalized into ModelOpt conversation rows.

The next retry will be submitted after thirteen infrastructure/API fixes. First,
`run_grpo_qwen3_235b_swe.sh` supports `SBATCH_EXCLUDE`, which let the retry avoid
the runtime-env timeout node `nvl72089-T06`. Second, remote
`SpecDec-RL/ray.sub` now passes `--min-worker-port=54001` and
`--max-worker-port=54513` to the Ray head command as well as worker commands, so
the driver/CoreWorker no longer defaults to binding `10002` on the head node.
The `2866525` ray-head log confirmed the patched head command includes that
worker port range. Third, the Qwen3MoE bridge shim now exposes a no-op
backward-compatible `finalize()` method, and probe `2866588` passed that
provider smoke. Fourth, the rollout launcher now carries
`ETP="${ETP:-1}"` into
`policy.megatron_cfg.expert_tensor_parallel_size=${ETP}`. Fifth, after `2866688`
proved the override reached Hydra but was lost during Bridge provider
`finalize()`, the Qwen3MoE shim and SpecDec-RL `community_import.py` now
restore `expert_tensor_parallel_size` immediately after finalize; probe
`2866747` passed with after-finalize ETP still equal to 1. Sixth, source
inspection `2866786` showed the Bridge mixin only forwards ETP through
`**model_parallel_kwargs`, so `community_import.py` now calls
`initialize_model_parallel(seed=0, expert_tensor_parallel_size=...)`. Seventh,
`2866789` showed this Bridge version exposes `provide_models()` instead of
`provide_distributed_model()`, so `community_import.py` now falls back to
`model_provider.provide_models(wrap_with_ddp=False)`. Eighth, `2866871` proved
the provider path reaches HF-to-Megatron weight scatter but exposed NCCL duplicate
GPU detection; SpecDec-RL `setup_distributed()` now binds the current CUDA device
from `LOCAL_RANK` before initializing the NCCL process group. Ninth, `2867033`
proved that CUDA binding fixed the duplicate GPU failure and exposed an old
Megatron-Bridge mapping gap: fused `TELayerNormColumnParallelLinear` layernorm
weights must be replicated. The Qwen3MoE shim now provides a fallback
`AutoMapping` that forces those layernorm/router/norm parameters through the
replicated path, and probe `2867247` passed the mapping-registry check. The
old `ReplicatedMapping` still used `megatron_module.weight` and CPU tensors for
the replicated broadcast path. The fallback `AutoMapping` now builds non-source
TP tensors from the actual target parameter such as `layer_norm_weight`. Tenth,
`2867356` showed that the target parameter can still be a CPU tensor while the
TP process group is NCCL. The fallback now performs the broadcast on the current
CUDA device and moves the result back to the target device before returning.
Probe `2867422` passed after this patch. The active next action remains
`2867545` then passed the previous CPU/NCCL failure and exposed one more old
Megatron-Bridge gap: grouped expert linear modules were not registered with
`TPAwareMapping`. The Qwen3MoE plugin now registers
`TEColumnParallelGroupedLinear` as column-parallel and
`TERowParallelGroupedLinear` as row-parallel. Probe `2867656` passed this
check. `2867662` then showed that the old column/row mappings still require
`.weight`, while Transformer Engine grouped expert modules expose numbered
weights such as `weight0`. The Qwen3MoE plugin now wraps grouped mapping calls
with a temporary `.weight` attribute pointing at the resolved target tensor.
Probe `2867766` was submitted to validate that helper before the next rollout
retry. The active next action remains to inspect `2867766`, then resubmit
rollout and wait for `train_data_step*.jsonl`.
The smoke must produce
`train_data_step*.jsonl`. The full
SWE-Gym after-smoke gate is still `waiting` / `poll_smoke`, and the Eagle3
hidden-state/train/export pipeline preflight remains incomplete because no
canonical rollout corpus exists yet.

The full SWE-Gym rollout submit preflight was regenerated at
`2026-05-22 17:50 PDT` with the real experiment name
`qwen3-235b-swe-rollout-vllm0102src-swegym-full` and the source-built vLLM
runtime passthrough env. It is PASS and `submit_ready=true`, but remains gated
behind the five-row smoke. The validator now has a synthetic case that rejects
dry-run-named full rollout preflights before any full rollout handoff. The
preflight itself also now auto-requires source-built vLLM env for `vllm0102src`
rollout names, and `validate_rollout_submit_preflight_contract.py` is part of
operator refresh with PASS status.

The earlier pending smoke `2861293` used the visible
`responses_api_agents/swe_agents/data/example.jsonl` directly. That file is
SWE-like but lacks `responses_create_params.metadata.instance_dict`, while the
OpenHands runner calls `json.loads(data_point["instance_dict"])`. The job was
cancelled before allocation and replaced by `2861605`. The repaired input was
generated by:

```bash
python3 experiments/eagle3_qwen3_235b/materialize_swegym_nemogym_dataset.py \
  --source-jsonl /lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/sdevare/repos/ultra/tk-nemo-gym/responses_api_agents/swe_agents/data/example.jsonl \
  --output-jsonl /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/swegym_example_for_sweagent_with_instance_dict.jsonl \
  --no-preserve-source-generation-params \
  --model Qwen/Qwen3-235B-A22B-Thinking-2507
```

The previous datafix attempt `2860803` reached Ray worker readiness and then
failed because the configured full SWE/R2E-Gym JSONL was not visible:

```text
/lustre/fsw/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/sdevare/repos/nano/dataset/rl/swe_all_datasets_train_w_agent_ref_r2e_gym_subset.jsonl
```

The launchers and preflight gates now validate `TRAIN_DATA_PATH` and
`VAL_DATA_PATH` before Slurm submit, including required SWE-agent metadata and a
parseable JSON-string `instance_dict`. This catches both the missing full path
and the incomplete visible example before Slurm allocation. A larger
SWE/R2E-Gym NemoGym JSONL still needs to be located or generated before
calibration or production draft training.

Hugging Face `SWE-Gym/SWE-Gym` materialization is now proven for both the small
schema proof and the usable full train split:

```text
proof output: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/swegym_train_nemogym_hf_8.jsonl
report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/swegym_hf_materialize_8.md
status: PASS, 8 rows, metadata.instance_dict present and parseable
full output: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/swegym_train_nemogym_hf_full.jsonl
full report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/swegym_hf_materialize_full.md
full status: PASS, 2438 rows, 0 validation failures, 0 warnings
full rollout preflight: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/rollout_capture_swegym_full_submit_preflight.md, PASS, submit_ready=true
full rollout after-smoke gate: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/full_swegym_after_smoke_gate.md, WAITING, next_step=poll_smoke, active smoke state is awaiting next smoke after probe 2867766
note: for SWE-Gym/SWE-bench-style eval, the runner writes instance_dict to a per-instance dataset and does not require nv-internal run_script.sh/parsing_script.py fields
```

The direct `R2E-Gym/R2E-Gym-Subset` materialization probe timed out after 300
seconds and left an empty JSONL, so R2E-Gym is not a ready source yet. The
current full rollout candidate is therefore the materialized SWE-Gym train split.
`submit_full_rollout_after_smoke_if_ready.py` now records the safety gate that
keeps this full rollout from being submitted until the five-row smoke has passed.
`watch_rollout_capture_materialize.sh` and `refresh_eagle3_operator_state.py`
now refresh that gate automatically.
The watcher remains no-submit unless explicitly launched with
`AUTO_SUBMIT_FULL_ROLLOUT=true` and `ALLOW_FULL_ROLLOUT_HEAVY_GPU=true`.
With those flags, the full-rollout gate starts a dedicated full rollout watcher
after successful submit; without `--allow-background`, watcher startup is
rejected before submission. The smoke materialization watcher now also reruns
the operator refresh immediately after an auto full-rollout gate execution, so a
submitted full rollout is reflected in planner/goal/completion reports instead
of leaving the previous no-submit plan stale.

## Backend Decision Snapshot

The immediate primary path remains **ModelOpt Eagle3** because the local and
remote pipeline already carries the required RL provenance: rollout capture,
conversation normalization, hidden-state dump, offline train, export, and
trained-draft NeMo-RL/vLLM smoke/sweep validation. This is the path that can
answer whether a Qwen3-235B Thinking draft helps the current SWE/RL generation
loop.

`vllm-project/speculators` is now recorded as a second-track backend, not an
immediate replacement. Its public docs describe EAGLE3 training with
`speculators>=0.5.0` and a separate serving environment with `vllm>=0.18`; the
latest PyPI vLLM check shows `vllm 0.21.0` released on 2026-05-15. That is
newer than the current source-built `vllm 0.10.2` runtime used to unblock
NeMo-RL rollout capture. Use Speculators only after a separate environment probe
and a converter from our normalized rollout conversations into its
`prepare_data.py` input path are added.

SpecForge remains an SGLang-side reference. Its data format is useful for
comparison (`id` + `conversations` or preformatted text), and the current
normalizer can emit `--output-schema specforge`; however, the main Qwen3-235B
RL path is still ModelOpt plus vLLM/NeMo-RL validation.

The longer backend comparison and data-flow notes are tracked in:

```text
experiments/eagle3_qwen3_235b/SPECULATIVE_TRAINING_BACKENDS.md
```

## Current State

Qwen3-235B Eagle3 draft training has not started yet. Earlier rollout retries
moved the active gate from vLLM native ABI to the source-built rollout smoke:

```text
source-build job: 2855535
job name: q235b-vllm-build
state: FAILED, then salvaged from tmp site
started: 2026-05-22 08:34:58 PDT
target site: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/python_site/vllm_0_10_2_cu129_torch28nv_source_py312
finalize job: 2856310 COMPLETED, installed pybase64
ABI probe job: 2856339 PASS
rollout smoke job: 2856410 FAILED, missing cpuinfo
dependency patch job: 2856499 COMPLETED, installed py-cpuinfo and passed AsyncLLM import
rollout smoke job: 2856536 FAILED, missing frozendict
dependency patch job: 2856588 COMPLETED, installed frozendict and passed Qwen3MoeForCausalLM import
runtime probe job: 2856645 PASS, AsyncEngineArgs.create_engine_config for Qwen3-235B
strict pip-check probe: 2856680 FAILED on package metadata/missing optional deps after runtime PASS
missing pure/runtime deps patch job: 2856741 FAILED probe after installing deps; pycountry was missing
pycountry patch job: 2856752 COMPLETED, verified AsyncLLM and Qwen3MoeForCausalLM imports
post-patch runtime probe: 2856767 functional PASS, strict pip check still FAILS on deferred NeMo-stack mismatches
rollout smoke job: 2856596 FAILED/CANCELLED, vLLM compile path hit missing torch._inductor.standalone_compile
compile-off 32-node retry: 2857291 FAILED, Hydra append syntax missing for compilation_config.level
compile-off compact retry: 2857334 FAILED, same Hydra append syntax issue
compile-off compact retry: 2857503 FAILED, OpenAIServingChat model_config API drift
compile-off 32-node retry: 2857581 FAILED, same OpenAI serving model_config API drift
model-config compatibility retry: 2858232 superseded by later Megatron/datafix retries
Megatron bridge compatibility probe: 2860778 PASS
rollout retry: 2860803 FAILED after Ray readiness because the configured full SWE/R2E-Gym JSONL was missing
raw-example smoke rollout: 2861293 CANCELLED before allocation because the visible example lacks metadata.instance_dict
current smoke rollouts: 2861605 CANCELLED after 2863716 started; 2863716 FAILED on Ray runtime-env agent timeout at nvl72089-T06; 2864216 FAILED on Ray head CoreWorker port 10002 bind at nvl72007-T01; 2866525 FAILED on missing Qwen3MoE provider finalize(); probe 2866588 PASSED finalize compatibility; 2866601 FAILED on expert tensor parallelism product mismatch; 2866688 FAILED after Bridge finalize recoupled ETP despite Hydra ETP=1; probe 2866747 PASSED after-finalize ETP preservation; 2866765 CANCELLED after source inspection showed ETP must be passed as initialize_model_parallel kwarg; 2866789 FAILED after passing ETP/world-size and vLLM startup, blocked by Bridge provide_distributed_model/provide_models API drift; 2866871 FAILED after vLLM/HF shard fetch on NCCL duplicate GPU during HF-to-Megatron scatter; 2867033 FAILED after CUDA device binding fixed duplicate GPU but exposed fused layernorm mapping scatter; probe 2867247 PASSED fallback AutoMapping compatibility; 2867262 FAILED on old ReplicatedMapping target/device broadcast behavior; 2867356 FAILED/CANCELLED on CPU target tensor NCCL broadcast behavior; probe 2867422 PASSED the CUDA-broadcast fallback patch; 2867545 FAILED on grouped expert linear module type detection; probe 2867656 PASSED grouped expert linear registration; 2867662 FAILED on grouped expert modules missing .weight; probe 2867766 is submitted for the temporary .weight patch, fixed 5-row SWE-Gym example data remains non-canonical
vLLM 0.13.0 source-build fallback: 2857812 RUNNING, ABI-only, no automatic rollout submit
```

The previous Qwen3-235B SWE rollout smoke reached Ray/vLLM worker startup and
then failed on the vLLM native extension ABI:

```text
undefined symbol: _ZN3c104cuda9SetDeviceEab
```

The source build produced a 1.04GB `vllm-0.10.2+cu129` aarch64 wheel against
the target NeMo container's Torch/CUDA, then failed only because `pybase64` was
missing in the target site. Finalize job `2856310` installed that dependency
into the tmp site, moved it to the canonical source-built site, and wrote
`vllm_native_source_build.json` PASS. ABI probe job `2856339` then passed on
that source-built site. The first source-built rollout smoke `2856410` moved
past native ABI but failed in `AsyncLLM` import because `cpuinfo` was missing.
Patch job `2856499` installed `py-cpuinfo` and verified `AsyncLLM` import. The
next smoke `2856536` then reached Qwen3-MoE model inspection and failed because
`compressed_tensors` needed `frozendict`. Patch job `2856588` installed
`frozendict` and verified `Qwen3MoeForCausalLM` import. A lightweight runtime probe
`2856645` also passed `AsyncEngineArgs.create_engine_config()` for
`Qwen/Qwen3-235B-A22B-Thinking-2507`, so the remaining uncertainty is the full
distributed vLLM engine startup and rollout capture path. A stricter runtime
probe `2856680` reran the same import/model-config path successfully, then
failed `pip check`. The useful signal from that failure is missing pure/runtime
packages, not a mandate to replace the NeMo container's Torch/Ray/TorchVision
stack. Patch job `2856741` successfully installed those packages but its probe
found one more dependency, `pycountry`. Follow-up patch `2856752` installed
`pycountry` and verified the source-built site imports. Probe `2856767` then
passed the runtime path again and failed only at strict `pip check`, where the
remaining issues are deferred stack-level mismatches such as `torchaudio`,
`triton`, `numba`, Torch, Ray, TorchVision, and setuptools. These should not be
changed unless the rollout smoke produces a concrete runtime failure requiring
one of them. The source-built rollout smoke `2856596` then reached Qwen3 model
load and vLLM worker startup, but failed during vLLM profile/KV-cache setup
because vLLM attempted the torch.compile/Inductor path and imported
`torch._inductor.standalone_compile`, which is absent from the target NVIDIA
Torch build. Retries `2857291` and `2857334` failed before runtime because the
new compile-config keys were passed without Hydra append syntax. Retries
`2857503` and `2857581` fixed that syntax and reached vLLM OpenAI serving setup,
then failed because the loaded vLLM API requires `model_config` for
`OpenAIServingChat` and `OpenAIServingTokenization`. The current SpecDec-RL patch
passes `model_config` based on constructor signatures, and later retries moved
past this vLLM API issue with:
`policy.generation.vllm_cfg.enforce_eager=True`,
`+policy.generation.vllm_kwargs.compilation_config.level=0`, and
`+policy.generation.vllm_kwargs.compilation_config.use_inductor=False`.
The next blocking issue was Megatron-Bridge Qwen3MoE registration; a narrow
sitecustomize plugin now passes compatibility probe `2860778`. Retry `2860803`
then failed on the missing full SWE/R2E-Gym data path. The direct visible
SWE-Gym example was then found to be incomplete for the current SWE-agent
runner because it lacks `metadata.instance_dict`. The next retry
uses a repaired five-row example to prove capture and normalization before any
full training rollout; earlier repaired-input retries `2861605`, `2863716`,
`2864216`, `2866525`, `2866601`, `2866688`, `2866765`, `2866789`, `2866871`, `2867033`, `2867262`, `2867356`, `2867545`, and `2867662` did not produce rollout JSONL.
The parallel `0.13.0` source-build fallback is only an ABI candidate at this
point; it should not replace the active `0.10.2` rollout path unless its
source-build and ABI reports pass and there is a concrete runtime/API or
speed/acceptance reason to promote it.

## Primary Data Decision

For the Qwen3-235B SWE/RL target, the primary Eagle3 training data should be
captured from actual Qwen3-235B Thinking rollout responses in the NeMo-RL SWE
loop.

Canonical target corpus:

```text
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations.jsonl
```

The expected source before normalization is the rollout log family:

```text
train_data_step*.jsonl
```

Those logs are normalized to ModelOpt-style conversation rows with assistant
tokens masked for Eagle3 loss.

DAPO and OpenMathInstruct-style math data are not the primary corpus for this
run. They are useful as references because Hayate's visible workflow used a
math-oriented response-generation -> Eagle3-training -> export flow, but they
match a math target distribution rather than the current SWE/RL rollout
distribution.

## Staged Training Schedule

Training should be staged rather than starting directly at production scale.

| stage | data | train steps | expected runtime | purpose |
| --- | --- | ---: | --- | --- |
| smoke | repaired SWE-Gym example rows | none or tiny dry-run | minutes once runtime works | prove vLLM runtime, capture, and normalization |
| pilot | 8 rollout conversations | 20 | dump 2h, train 2h, export 1h limits | prove ModelOpt Eagle3 train/export path |
| calibration | 5k-10k rollout conversations | up to 2k | roughly half-day to one day including rollout/queue/debug | measure acceptance, speedup, and reward risk |
| production candidate | 50k-100k+ rollout conversations | about one epoch, around 25k steps at effective batch 4 for 100k | roughly 1-3 days excluding queue/debug | produce a serious draft candidate |

The first useful training submission is the pilot. It should not be submitted
until the rollout corpus exists and passes the corpus analyzer.

Public EAGLE-3/sample-size evidence supports this staged plan:

- vLLM Speculators' Qwen3-8B online tutorial uses 5k samples as a getting-started
  path, not a production-quality draft target.
- The original EAGLE paper used 68,000 ShareGPT dialogue iterations and reports
  1-2 days on 4x A100 40G for LLaMA2-Chat 70B. That is a useful low-cost
  baseline, but EAGLE-3 was introduced partly because it keeps improving with
  larger data.
- Baseten's EAGLE-3 training guide recommends about 100k samples for specialized
  task/format drafters and about 500k samples for large generic conversation
  drafters.
- The EAGLE-3 paper reports that the new architecture benefits from scaling
  training data and uses ShareGPT 68k plus UltraChat 464k, while also adding
  OpenThoughts-114k-math for the DeepSeek-R1-Distill reasoning model.
- NVIDIA's public Qwen3-235B-A22B Eagle3 model card reports 503.3k total data
  points / roughly 500k samples, using UltraChat-200k and
  Magpie-Llama-3.1-Pro-300K-Filtered prompts with Qwen3-235B-A22B-generated
  synthetic responses.
- SpecForge's data-prep docs list 200k UltraChat, 120k ShareGPT, and 1.4M
  PerfectBlend as supported sources and recommend regenerating responses with
  the target model for production acceptance rate.
- TorchSpec's Kimi K2.5 EAGLE-3 scale report shows a high-end path at 600k
  samples / 6B tokens / 1500 H200 GPU hours, but also highlights the hidden-state
  storage/memory burden for long-context samples.
- TAPS (Task Aware Proposal Distributions, 2026) is a useful warning for this
  workstream: in its controlled EAGLE-2/HASS study, 70k MathInstruct examples
  beat 70k ShareGPT on math benchmarks while ShareGPT was stronger on MT-Bench.
  Mixed data improved robustness but did not uniformly dominate. The practical
  implication is that DAPO/OpenMathInstruct-style data is right for a math
  draft, while SWE/RL acceptance should be trained and measured on Qwen3-235B
  SWE/RL rollout conversations.
- NVIDIA's April 2026 NeMo-RL speculative-decoding report reinforces the same
  point in the RL setting: switching from generic chat data to in-domain
  post-training data improved the reported RL-Zero rollout speedup from about
  1.5x to 1.8x. This is not a request to train only on DAPO for SWE; it is
  evidence that the draft corpus should match the rollout distribution being
  accelerated.

For Qwen3-235B SWE/RL, use the specialized path first. The current full
SWE-Gym materialization gives 2,438 prompts, enough for a first calibration
rollout after the smoke. A serious SWE/RL draft should then use 10k-50k
target-domain rollout conversations before spending on a 50k-100k candidate.
Only consider 300k-500k if the objective changes from SWE/RL acceleration to a
general-purpose Qwen3-235B draft.

Practical readout: the first non-toy training should use the 2,438 SWE-Gym
rollout rows only as calibration, with one pass or <=1k train steps. If that
improves acceptance without hurting RL behavior, collect more target-model
rollout conversations and move to 10k-50k. The first serious checkpoint should
target 50k-100k SWE/RL conversations at about 1k-2k tokens/sample. A 500k run is
appropriate only when we want a broad reusable Qwen3-235B draft and can afford
the rollout generation, hidden-state extraction, and storage.

## Execution Order

1. Source-build vLLM against the target NeMo container Torch/CUDA. DONE.
2. Run native ABI probe on only the source-built vLLM site. DONE.
3. Run Qwen3-235B SWE-Gym one-step rollout smoke. Job `2856596` proved model load but failed on vLLM Inductor compile incompatibility. Retries `2857291` and compact `2857334` failed on Hydra append syntax; `2857503` and `2857581` then failed on vLLM OpenAI serving `model_config` API drift. Later retries moved past that path; Megatron-Bridge Qwen3MoE registration is now patched by a narrow plugin. Job `2860803` reached Ray readiness and failed on a missing full SWE/R2E-Gym data path. Job `2861293` was cancelled before allocation because the raw visible example misses `instance_dict`. Retry `2861605` was cancelled after the balanced fallback started. Balanced fallback `2863716` then failed on a Ray runtime-env agent timeout at `nvl72089-T06`; retry `2864216` excluded that node but failed on Ray head CoreWorker port `10002` binding at `nvl72007-T01`; retry `2866525` moved into Megatron/vLLM worker setup and failed on missing Qwen3MoE provider `finalize()`. Probe `2866588` passed the finalize shim. Retry `2866601` then failed because the provider saw `expert_tensor_parallel_size=4`, producing an expert/tensor/pipeline product of 256 for decoder world size 64. Retry `2866688` proved Hydra carried `expert_tensor_parallel_size=1`, but Bridge provider `finalize()` recoupled it before Megatron-Core initialization. Probe `2866747` passed the after-finalize ETP preservation patch. Retry `2866765` was cancelled after source inspection `2866786` showed the Bridge mixin does not forward ETP unless supplied as an initialize kwarg. Retry `2866789` passed the prior ETP/world-size failure and initialized vLLM generation, then failed on Bridge provider API drift: `provide_distributed_model()` is absent and `provide_models()` is the available entrypoint. Retry `2866871` passed that path and exposed NCCL duplicate GPU detection during HF-to-Megatron scatter. Retry `2867033` passed the CUDA binding fix and reached HF-to-Megatron import, then failed because old `TPAwareMapping` treated fused layernorm weights as column-parallel. Probe `2867247` passed the fallback `AutoMapping` shim. Retry `2867262` selected replicated mapping but exposed old `ReplicatedMapping` target/device broadcast behavior. Retry `2867356` selected the actual target tensor but exposed CPU target tensor NCCL broadcast behavior. Probe `2867422` passed the CUDA-broadcast fallback patch. Retry `2867545` passed that path and exposed missing grouped expert linear module registration. Probe `2867656` passed the grouped linear registration patch. Retry `2867662` then exposed old column/row mappings expecting `.weight` on grouped expert modules. Probe `2867766` is submitted for the temporary `.weight` helper patch before the next 24-node rollout retry.
4. Materialize the smoke output as non-canonical data and inspect it.
5. Run a real target-domain Qwen3-235B SWE rollout capture.
6. Normalize `train_data_step*.jsonl` into the canonical conversation corpus.
7. Run Eagle3 pilot hidden-state dump, train, export, and artifact validators.
8. Run fixed-draft vLLM/NeMo-RL smoke with the exported draft.
9. Only then scale to calibration and production-candidate training.

## Do Not Start Yet Unless These Are True

- `vllm_native_source_build.json` reports PASS for the source-built site. DONE.
- `vllm_native_abi_probe.json` reports PASS for the source-built site. DONE.
- The Qwen3-235B rollout smoke produces parseable role-aware rollout logs.
- The canonical rollout conversation corpus exists.
- The corpus strategy report selects `actual_rl_rollout` as the primary source.
- The pipeline submit preflight reports that dump/train/export submission is
  ready.
