# Qwen3-235B Eagle3 Backend And Rollout Data Report

Last updated: 2026-05-23 01:14 PDT

This report is the rolling context anchor for the Qwen3-235B Eagle3 draft-model
work. Update it whenever the rollout data source, training backend, or cluster
gate changes so the effort can continue after context compaction.

## Current State

We are not training the draft model yet, and we do not have rollout data yet.
The active work is a Qwen3-235B SWE rollout-capture smoke retry:

```text
job: 2861605
name: qwen3-235b-swe-rollout-vllm0102src-swegym-fixed-instancedict-smoke1step
state at 2026-05-22 19:01 PDT: CANCELLED by 150081 before allocation
squeue start estimate: n/a
shape: 16 nodes, 4 GPUs/node, NUM_GEN_NODES=4
input: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/swegym_example_for_sweagent_with_instance_dict.jsonl
output target: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations_vllm0102src_swegym_fixed_instancedict_smoke.jsonl

job: 2863716
name: qwen3-235b-swe-rollout-capture-balanced24n4g
state: FAILED, ExitCode=1:0, elapsed=00:04:07
start time: 2026-05-22T19:01:00
end time: 2026-05-22T19:05:07
allocated shape: 24 nodes, 96 GPUs
shape: 24 nodes, 4 GPUs/node, NUM_GEN_NODES=8
input: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/swegym_example_for_sweagent_with_instance_dict.jsonl
output target: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations_balanced24n4g.jsonl
failure: Ray runtime-env agent timeout on nvl72089-T06 / 10.109.17.223; no rollout JSONL produced

job: 2864216
name: qwen3-235b-swe-rollout-capture-balanced24n4g-excl-t06
state: FAILED, ExitCode=1:0, elapsed=00:02:47
start time: 2026-05-22T19:11:28
end time: 2026-05-22T19:14:15
allocated shape: 24 nodes, 96 GPUs
shape: 24 nodes, 4 GPUs/node, NUM_GEN_NODES=8
exclude: nvl72089-T06
input: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/swegym_example_for_sweagent_with_instance_dict.jsonl
output target: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations_balanced24n4g_excl_t06.jsonl
failure: Ray CoreWorker gRPC bind failed on head nvl72007-T01, port 10002 already in use; no rollout JSONL produced

job: 2866525
name: qwen3-235b-swe-rollout-capture-balanced24n4g-headport-excl-t06-t01
state: FAILED, ExitCode=1:0, elapsed=00:06:14
start time: 2026-05-22T22:50:23
end time: 2026-05-22T22:56:37
allocated shape: 24 nodes, 96 GPUs
shape: 24 nodes, 4 GPUs/node, NUM_GEN_NODES=8
exclude: nvl72089-T06,nvl72007-T01
input: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/swegym_example_for_sweagent_with_instance_dict.jsonl
output target: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations_balanced24n4g_headport_excl_t06_t01.jsonl
failure: Megatron Qwen3MoE provider missing finalize(); no rollout JSONL produced

job: 2866588
name: q235b-megatron-compat-probe
state: COMPLETED, ExitCode=0:0, elapsed=00:01:44
result: PASS, qwen3_moe_provider_smoke.has_finalize=True, errors=[]
report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/megatron_compat_probe_finalize.md

job: 2866601
name: qwen3-235b-swe-rollout-capture-balanced24n4g-headport-finalize-excl-t06-t01
state: FAILED, ExitCode=1:0, elapsed=00:06:16
start time: 2026-05-22T23:00:19
end time: 2026-05-22T23:06:35
allocated shape: 24 nodes, 96 GPUs
shape: 24 nodes, 4 GPUs/node, NUM_GEN_NODES=8
exclude: nvl72089-T06,nvl72007-T01
input: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/swegym_example_for_sweagent_with_instance_dict.jsonl
output target: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations_balanced24n4g_headport_finalize_excl_t06_t01.jsonl
watcher PID: 2755376
watcher log: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/watch_rollout_capture_2866601_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_excl_t06_t01_swegym_smoke.log
state report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/rollout_capture_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_excl_t06_t01_2866601_swegym_state_advance.json
failure: Megatron saw expert_tensor_parallel_size=4, so decoder world_size 64 was not divisible by expert_tensor_model_pipeline_parallel size 256; no rollout JSONL produced

job: 2866688
name: qwen3-235b-swe-rollout-capture-balanced24n4g-headport-finalize-etp1-excl-t06-t01
state: FAILED, ExitCode=1:0, elapsed=00:06:21
start time: 2026-05-22T23:09:59
end time: 2026-05-22T23:16:20
allocated shape: 24 nodes, 96 GPUs
shape: 24 nodes, 4 GPUs/node, NUM_GEN_NODES=8, TP=4, ETP=1, EP=16, CP=1, PP=4
exclude: nvl72089-T06,nvl72007-T01
input: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/swegym_example_for_sweagent_with_instance_dict.jsonl
output target: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations_balanced24n4g_headport_finalize_etp1_excl_t06_t01.jsonl
watcher PID: 2834941
watcher log: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/watch_rollout_capture_2866688_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etp1_excl_t06_t01_swegym_smoke.log
state report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/rollout_capture_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etp1_excl_t06_t01_2866688_swegym_state_advance.json
failure: Hydra carried expert_tensor_parallel_size=1, but Bridge finalize recoupled provider ETP to TP before Megatron-Core initialize; no rollout JSONL produced

job: 2866747
name: q235b-megatron-compat-probe
state: COMPLETED, ExitCode=0:0, elapsed=00:01:57
result: PASS, qwen3_moe_provider_smoke.expert_tensor_parallel_size_after_finalize=1, errors=[]
report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/megatron_compat_probe_etp_finalize.md

job: 2866765
name: qwen3-235b-swe-rollout-capture-balanced24n4g-headport-finalize-etp1-preserve-excl-t06-t01
state: CANCELLED by 150081, elapsed=00:05:50
start time: 2026-05-22T23:20:45
end time: 2026-05-22T23:26:35
allocated/requested shape: 24 nodes, 96 GPUs
shape: 24 nodes, 4 GPUs/node, NUM_GEN_NODES=8, TP=4, ETP=1, EP=16, CP=1, PP=4
exclude: nvl72089-T06,nvl72007-T01
input: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/swegym_example_for_sweagent_with_instance_dict.jsonl
output target: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations_balanced24n4g_headport_finalize_etp1_preserve_excl_t06_t01.jsonl
watcher PID: 2937984
watcher log: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/watch_rollout_capture_2866765_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etp1_preserve_excl_t06_t01_swegym_smoke.log
state report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/rollout_capture_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etp1_preserve_excl_t06_t01_2866765_swegym_state_advance.json
failure: source inspection showed Bridge initialize_model_parallel ignores provider.expert_tensor_parallel_size unless passed through kwargs; job was cancelled after same product-256 error appeared

job: 2866786
name: q235b-bridge-src-inspect
state: COMPLETED, ExitCode=0:0, elapsed=00:01:12
result: Megatron-Bridge model_provider_mixin.initialize_model_parallel accepts **model_parallel_kwargs and does not pass provider.expert_tensor_parallel_size by default
report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/bridge_src_inspect_2866786.out

job: 2866789
name: qwen3-235b-swe-rollout-capture-balanced24n4g-headport-finalize-etpkwarg-excl-t06-t01
state: FAILED, ExitCode=1:0, elapsed=00:06:02
start time: 2026-05-22T23:27:44
allocated/requested shape: 24 nodes, 96 GPUs
shape: 24 nodes, 4 GPUs/node, NUM_GEN_NODES=8, TP=4, ETP=1, EP=16, CP=1, PP=4
exclude: nvl72089-T06,nvl72007-T01
input: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/swegym_example_for_sweagent_with_instance_dict.jsonl
output target: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations_balanced24n4g_headport_finalize_etpkwarg_excl_t06_t01.jsonl
watcher PID: 2994448
watcher log: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/watch_rollout_capture_2866789_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etpkwarg_excl_t06_t01_swegym_smoke.log
state report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/rollout_capture_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etpkwarg_excl_t06_t01_2866789_swegym_state_advance.json
failure: passed the prior ETP/world-size failure and vLLM startup, then failed because this Bridge provider exposes provide_models(), not provide_distributed_model()

job: 2866871
name: qwen3-235b-swe-rollout-capture-balanced24n4g-headport-finalize-etpkwarg-providemodels-excl-t06-t01
state at 2026-05-22 23:50 PDT: FAILED
allocated/requested shape: 24 nodes, 96 GPUs
shape: 24 nodes, 4 GPUs/node, NUM_GEN_NODES=8, TP=4, ETP=1, EP=16, CP=1, PP=4
exclude: nvl72089-T06,nvl72007-T01
input: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/swegym_example_for_sweagent_with_instance_dict.jsonl
output target: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations_balanced24n4g_headport_finalize_etpkwarg_providemodels_excl_t06_t01.jsonl
watcher PID: 3119496
watcher log: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/watch_rollout_capture_2866871_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etpkwarg_providemodels_excl_t06_t01_swegym_smoke.log
state report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/rollout_capture_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etpkwarg_providemodels_excl_t06_t01_2866871_swegym_state_advance.json
failure: vLLM and HF shard fetch completed, then Megatron-Bridge HF-to-TP scatter failed with NCCL duplicate GPU detection because same-node ranks used cuda:0

job: 2867033
name: qwen3-235b-swe-rollout-capture-balanced24n4g-headport-finalize-etpkwarg-providemodels-cudadev-excl-t06-t01
state: FAILED, ExitCode=1:0, elapsed=00:06:18
start time: 2026-05-22T23:59:02
end time: 2026-05-23T00:05:20
allocated/requested shape: 24 nodes, 96 GPUs
shape: 24 nodes, 4 GPUs/node, NUM_GEN_NODES=8, TP=4, ETP=1, EP=16, CP=1, PP=4
exclude: nvl72089-T06,nvl72007-T01
input: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/swegym_example_for_sweagent_with_instance_dict.jsonl
output target: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations_balanced24n4g_headport_finalize_etpkwarg_providemodels_cudadev_excl_t06_t01.jsonl
watcher PID: 3325121
watcher log: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/watch_rollout_capture_2867033_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etpkwarg_providemodels_cudadev_excl_t06_t01_swegym_smoke.log
state report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/rollout_capture_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etpkwarg_providemodels_cudadev_excl_t06_t01_2867033_swegym_state_advance.json
failure: CUDA device binding fixed the duplicate GPU scatter issue, but old Megatron-Bridge TPAwareMapping treated fused TELayerNormColumnParallelLinear.layer_norm_weight as column-parallel instead of replicated

job: 2867247
name: q235b-megatron-compat-probe
state: COMPLETED, ExitCode=0:0, elapsed=00:01:42
result: PASS, qwen3_moe_mapping_registry.layernorm_entries[0].type=AutoMapping, errors=[]
report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/megatron_compat_probe_automapping_fallback.md

job: 2867262
name: qwen3-235b-swe-rollout-capture-balanced24n4g-headport-finalize-etpkwarg-providemodels-cudadev-automapping-excl-t06-t01
state: FAILED, ExitCode=1:0, elapsed=00:06:18
start time: 2026-05-23T00:21:30
end time: 2026-05-23T00:27:48
allocated/requested shape: 24 nodes, 96 GPUs
shape: 24 nodes, 4 GPUs/node, NUM_GEN_NODES=8, TP=4, ETP=1, EP=16, CP=1, PP=4
exclude: nvl72089-T06,nvl72007-T01
input: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/swegym_example_for_sweagent_with_instance_dict.jsonl
output target: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations_balanced24n4g_headport_finalize_etpkwarg_providemodels_cudadev_automapping_excl_t06_t01.jsonl
watcher PID: 3598143
watcher log: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/watch_rollout_capture_2867262_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etpkwarg_providemodels_cudadev_automapping_excl_t06_t01_swegym_smoke.log
state report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/rollout_capture_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etpkwarg_providemodels_cudadev_automapping_excl_t06_t01_2867262_swegym_state_advance.json
failure: fallback AutoMapping selected replicated path, but old ReplicatedMapping still built non-source TP tensors from megatron_module.weight and attempted CPU tensor broadcast

job: 2867356
name: qwen3-235b-swe-rollout-capture-balanced24n4g-headport-finalize-etpkwarg-providemodels-cudadev-automaprepbcast-excl-t06-t01
state: CANCELLED by 150081, elapsed=00:05:37
allocated/requested shape: 24 nodes, 96 GPUs
shape: 24 nodes, 4 GPUs/node, NUM_GEN_NODES=8, TP=4, ETP=1, EP=16, CP=1, PP=4
exclude: nvl72089-T06,nvl72007-T01
input: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/swegym_example_for_sweagent_with_instance_dict.jsonl
output target: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations_balanced24n4g_headport_finalize_etpkwarg_providemodels_cudadev_automaprepbcast_excl_t06_t01.jsonl
watcher PID: 3745453
watcher log: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/watch_rollout_capture_2867356_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etpkwarg_providemodels_cudadev_automaprepbcast_excl_t06_t01_swegym_smoke.log
state report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/rollout_capture_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etpkwarg_providemodels_cudadev_automaprepbcast_excl_t06_t01_2867356_swegym_state_advance.json
failure: fallback AutoMapping selected the actual target tensor shape, but some target parameters were still CPU tensors during NCCL broadcast

job: 2867422
name: q235b-megatron-compat-probe
state: COMPLETED, ExitCode=0:0, elapsed=00:01:50
result: PASS, qwen3_moe_bridge_registered=True, qwen3_moe_provider_smoke.provider=Qwen3MoEModelProvider, errors=[]
report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/megatron_compat_probe.json

job: 2867545
name: qwen3-235b-swe-rollout-capture-balanced24n4g-headport-finalize-etpkwarg-providemodels-cudadev-automaprepbcastcuda-excl-t06-t01
state: FAILED, ExitCode=1:0, elapsed=00:06:09
start time: 2026-05-23T00:44:36
end time: 2026-05-23T00:50:45
allocated/requested shape: 24 nodes, 96 GPUs
shape: 24 nodes, 4 GPUs/node, NUM_GEN_NODES=8, TP=4, ETP=1, EP=16, CP=1, PP=4
exclude: nvl72089-T06,nvl72007-T01
input: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/swegym_example_for_sweagent_with_instance_dict.jsonl
output target: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations_balanced24n4g_headport_finalize_etpkwarg_providemodels_cudadev_automaprepbcastcuda_excl_t06_t01.jsonl
watcher PID: 3855439
watcher log: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/watch_rollout_capture_2867545_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etpkwarg_providemodels_cudadev_automaprepbcastcuda_excl_t06_t01_swegym_smoke.log
state report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/rollout_capture_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etpkwarg_providemodels_cudadev_automaprepbcastcuda_excl_t06_t01_2867545_swegym_state_advance.json
failure: passed the previous CPU/NCCL fallback path and vLLM loaded the 235B model; then old TPAwareMapping could not infer TEColumnParallelGroupedLinear for decoder.layers.0.mlp.experts.linear_fc1.weight0

job: 2867656
name: q235b-megatron-compat-probe
state: COMPLETED, ExitCode=0:0, elapsed=00:01:36
result: PASS, tpaware_grouped_linear_detection={TEColumnParallelGroupedLinear: column, TERowParallelGroupedLinear: row}, errors=[]
report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/megatron_compat_probe.json

job: 2867662
name: qwen3-235b-swe-rollout-capture-balanced24n4g-headport-finalize-etpkwarg-providemodels-cudadev-automapgroupedlin-excl-t06-t01
state: FAILED, ExitCode=1:0, elapsed=00:05:58
start time: 2026-05-23T01:07:06
end time: 2026-05-23T01:13:04
allocated/requested shape: 24 nodes, 96 GPUs
shape: 24 nodes, 4 GPUs/node, NUM_GEN_NODES=8, TP=4, ETP=1, EP=16, CP=1, PP=4
exclude: nvl72089-T06,nvl72007-T01
input: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/swegym_example_for_sweagent_with_instance_dict.jsonl
output target: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations_balanced24n4g_headport_finalize_etpkwarg_providemodels_cudadev_automapgroupedlin_excl_t06_t01.jsonl
watcher PID: 4017490
watcher log: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/watch_rollout_capture_2867662_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etpkwarg_providemodels_cudadev_automapgroupedlin_excl_t06_t01_swegym_smoke.log
state report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/rollout_capture_vllm0102src_megatroncompat_resourcefix_balanced_24n4g_headport_finalize_etpkwarg_providemodels_cudadev_automapgroupedlin_excl_t06_t01_2867662_swegym_state_advance.json
failure: grouped module type detection worked, but old ColumnParallelMapping expected .weight while TEColumnParallelGroupedLinear exposes weight0/weight1

job: 2867766
name: q235b-megatron-compat-probe
state at 2026-05-23 01:14 PDT: submitted, result pending because local SSH/DNS lookup for oci-hsg-cs-001-vscode-02 is temporarily failing
expected result: grouped_linear_temporary_weight_attr.has_weight=true and weight_is_weight0=true
```

These smokes use five repaired SWE-Gym rows and are runtime/capture proofs only.
They should show whether the patched NeMo-RL/vLLM path produces parseable
`train_data_step*.jsonl` records and whether those records normalize into
assistant-final training conversations.

The next retry will be submitted after thirteen infrastructure/API fixes. First,
the top-level rollout launcher supports `SBATCH_EXCLUDE`, which removed the
runtime-env timeout node `nvl72089-T06`. Second, remote `SpecDec-RL/ray.sub`
now passes `--min-worker-port=54001` and `--max-worker-port=54513` to the Ray
head command as well as worker commands, avoiding the default head CoreWorker
bind to `10002`. The `2866525` ray-head log confirmed the patched head command
includes that worker port range. Third, the Qwen3MoE bridge shim now exposes a
backward-compatible `finalize()` method, and probe `2866588` passed that provider
smoke. Fourth, the rollout launcher now carries `ETP="${ETP:-1}"` into
`policy.megatron_cfg.expert_tensor_parallel_size=${ETP}`. Fifth, after `2866688`
proved the override reached Hydra but was lost during Bridge provider
`finalize()`, the Qwen3MoE shim and SpecDec-RL `community_import.py` now
restore `expert_tensor_parallel_size` immediately after finalize; probe
`2866747` passed with after-finalize ETP still equal to 1. Sixth, source
inspection `2866786` showed the Bridge mixin only forwards ETP through
`**model_parallel_kwargs`, so `community_import.py` now calls
`initialize_model_parallel(seed=0, expert_tensor_parallel_size=...)`. Seventh,
`2866789` proved the current Bridge provider API is `provide_models()`, not
`provide_distributed_model()`, so `community_import.py` now falls back to
`model_provider.provide_models(wrap_with_ddp=False)`. Eighth, `2866871` reached
HF-to-Megatron weight scatter and failed with NCCL duplicate GPU detection, so
SpecDec-RL `setup_distributed()` now calls `torch.cuda.set_device()` from
`LOCAL_RANK` before NCCL process-group initialization. Ninth, `2867033` proved
the CUDA binding fix and exposed a Megatron-Bridge mapping gap; the Qwen3MoE
plugin now provides a fallback `AutoMapping` that routes fused layernorm/router
and norm weights through replicated mapping, and probe `2867247` passed this
mapping-registry check. Retry `2867262` then showed that old `ReplicatedMapping`
still built non-source TP tensors from `megatron_module.weight` and tried to
broadcast CPU tensors. The fallback `AutoMapping` now creates replicated
broadcast tensors from the actual target parameter. Retry `2867356` then showed
that the target parameter can still be CPU while the TP group is NCCL, so the
fallback now broadcasts on the current CUDA device and moves back to the target
device before returning. Probe `2867422` passed after this patch. The current
retry `2867545` then passed the previous CPU/NCCL failure and exposed that old
`TPAwareMapping` does not know grouped expert linear module names. The Qwen3MoE
plugin now registers `TEColumnParallelGroupedLinear` as column-parallel and
`TERowParallelGroupedLinear` as row-parallel. Probe `2867656` passed this check.
`2867662` then showed that old column/row mappings still expect `.weight`, while
Transformer Engine grouped expert modules expose numbered weights such as
`weight0`. The plugin now wraps grouped mapping calls with a temporary `.weight`
attribute pointing at the resolved target tensor. Probe `2867766` was submitted
to validate that helper before the next rollout retry. The current next action
is to inspect `2867766`, resubmit rollout if it passes, and wait for
`train_data_step*.jsonl`.
`full_swegym_after_smoke_gate` is still `waiting` / `poll_smoke`, and
`pipeline_submit_preflight` is still `incomplete` because no rollout corpus
exists yet.

The full SWE-Gym rollout submit preflight was refreshed at
`2026-05-22 17:50 PDT` after a stale dry-run-named artifact was found. It now
uses `WANDB_NAME=qwen3-235b-swe-rollout-vllm0102src-swegym-full`, contains no
`dryrun` marker, and carries the source-built vLLM runtime passthrough env:
`SHARED_VLLM_SITE`, `VLLM_PIP_SPEC`, `VLLM_ENFORCE_EAGER=True`,
`VLLM_COMPILATION_LEVEL=0`, and `VLLM_USE_INDUCTOR=False`. The preflight now
auto-requires those env vars for `vllm0102src` rollout names. The operator
refresh also runs `validate_rollout_submit_preflight_contract.py`, which proves
that missing or invalid source-built vLLM env fails while the refreshed full
preflight passes. The full-rollout gate still rejects dry-run-named full
preflights, so stale submit packets cannot become the full rollout handoff
after smoke PASS.

The larger rollout input is now available and preflighted:

```text
full SWE-Gym input: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/swegym_train_nemogym_hf_full.jsonl
rows: 2438
materialization report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/swegym_hf_materialize_full.md
full rollout submit preflight: PASS, submit_ready=true
preflight report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/rollout_capture_swegym_full_submit_preflight.md
gated full-submit report: /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/full_swegym_after_smoke_gate.md
```

The gated full-submit report currently says `WAITING` / `poll_smoke`: full
SWE-Gym rollout is ready to submit, but it is intentionally blocked until the
five-row smoke rollout proves runtime/capture and produces a valid normalized
conversation corpus. The rollout watcher and operator refresh path now refresh
this report automatically, so the next action should flip from `poll_smoke` to
`submit_full_rollout` when the smoke state becomes PASS.
The watcher is still no-submit by default. It can only submit the full rollout
after all gates pass if started with both `AUTO_SUBMIT_FULL_ROLLOUT=true` and
`ALLOW_FULL_ROLLOUT_HEAVY_GPU=true`.
In that explicit auto-submit mode, the gate also starts a full-rollout
materialization watcher by default (`START_FULL_ROLLOUT_WATCHER=true`,
`ALLOW_FULL_ROLLOUT_BACKGROUND=true`) so the 2,438-row rollout is not submitted
without terminal artifact handling. The smoke watcher now follows any auto
full-rollout gate execution with a second operator refresh, preventing planner
and completion reports from staying on the pre-submit gate state.

The old expected full path
`swe_all_datasets_train_w_agent_ref_r2e_gym_subset.jsonl` is still missing on
both checked Lustre roots. A direct `R2E-Gym/R2E-Gym-Subset` materialization
probe timed out after 300 seconds and left an empty output file, so it is not a
ready data path. For the current target, `SWE-Gym/SWE-Gym` is the usable full
dataset.

## How Rollout Data Will Be Collected

The training corpus should come from actual Qwen3-235B Thinking responses
inside the NeMo-RL SWE loop, not from static math data. The data flow is:

1. Run the active smoke rollout job `2867545` and use it only if it produces a
   valid terminal capture.
2. Inspect its `train_data_step*.jsonl` files under the rollout log directory.
3. Normalize the captured rows:

```bash
python3 experiments/eagle3_qwen3_235b/normalize_rl_rollouts_to_conversations.py \
  --input /path/to/rl_rollout_capture_logs \
  --output /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations.jsonl \
  --model Qwen/Qwen3-235B-A22B-Thinking-2507 \
  --include-reasoning-content \
  --compact-current-turn
```

4. Validate the normalized corpus:

```bash
python3 experiments/eagle3_qwen3_235b/validate_training_conversations.py \
  --input /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/qwen3_235b_swe_rollout_conversations.jsonl \
  --markdown-out /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/training_conversations_validation.md \
  --json-out /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/training_conversations_validation.json
```

5. After the smoke proves runtime and capture, submit the full SWE-Gym rollout
   using `swegym_train_nemogym_hf_full.jsonl` as both train and validation input.
6. Feed the validated conversation corpus into the selected draft trainer.

The no-submit gate for step 5 is:

```bash
python3 experiments/eagle3_qwen3_235b/submit_full_rollout_after_smoke_if_ready.py \
  --artifact-root /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3 \
  --json-out /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/full_swegym_after_smoke_gate.json \
  --markdown-out /lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/full_swegym_after_smoke_gate.md
```

Only when that report says `READY` should the same command be rerun with
`--execute --allow-heavy-gpu`.
For an automatic handoff from smoke watcher to full rollout submission, launch
the watcher with `AUTO_SUBMIT_FULL_ROLLOUT=true` and
`ALLOW_FULL_ROLLOUT_HEAVY_GPU=true`; without the allow flag the gate records
`rerun_with_allow_heavy_gpu` and does not submit.
The gate uses `--start-watcher --allow-background` in that handoff, so successful
full rollout submission immediately gets a materialization watcher.

The canonical ModelOpt schema remains:

```json
{"conversation_id": "...", "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

For SpecForge or vLLM Speculators comparison runs, the same normalizer can write:

```bash
python3 experiments/eagle3_qwen3_235b/normalize_rl_rollouts_to_conversations.py \
  --input /path/to/train_data_step_files \
  --output /path/to/speculators_conversations.jsonl \
  --output-schema speculators
```

which produces:

```json
{"id": "...", "conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

The helper `convert_conversations_to_speculators_jsonl.py` can also convert an
already materialized ModelOpt corpus into this `id/conversations` schema and
emits the corresponding `scripts/prepare_data.py` command.

## Sample Size Recommendation

Public evidence suggests three distinct scales:

- **Smoke / framework tutorial:** 5k samples. vLLM Speculators' online EAGLE3
  tutorial uses `--max-samples 5000` for Qwen3-8B and reports about 17 minutes
  on 4x H100 for the demo path. Treat this as a framework smoke or
  getting-started scale, not a production target.
- **Original EAGLE baseline:** about 68k samples. The EAGLE paper trained on
  ShareGPT with 68,000 dialogue iterations; for LLaMA2-Chat 70B it reports
  1-2 days on 4x A100 40G. This is a useful "works at modest scale" reference,
  but EAGLE-3 was explicitly designed to benefit more from scaling data.
- **Specialized/task-specific draft:** about 100k samples. Baseten's EAGLE-3
  training guide recommends about 100k samples for specialized formats/tasks,
  including large models, and emphasizes regenerating outputs with the target
  model. It also recommends about 1k-2k total tokens per sample.
- **General-purpose draft:** about 500k samples. The EAGLE-3 paper uses
  ShareGPT 68k plus UltraChat 464k, calls the target model to regenerate
  responses, and adds OpenThoughts-114k-math for the DeepSeek-R1-Distill
  reasoning model. NVIDIA's public Qwen3-235B-A22B Eagle3 model card reports
  503.3k total data points / roughly 500k samples from UltraChat-200k and
  Magpie-Llama-3.1-Pro-300K-Filtered prompts, with synthetic responses from
  Qwen3-235B-A22B. SpecForge also treats 200k UltraChat, 120k ShareGPT, and
  1.4M PerfectBlend as standard supported sources.
- **Long-context frontier scale:** 600k samples can be used, but the system
  cost changes materially. The TorchSpec Kimi K2.5 EAGLE-3 report describes
  600k samples / 6B tokens and 1500 H200 GPU hours; it also notes that one
  128k-token sample can require about 7GB of hidden states. This is not the
  first target for our SWE/RL path.
- **Domain match matters:** TAPS (2026) trained task-aware drafters with 70k
  MathInstruct examples versus 70k ShareGPT examples and found clear
  specialization: math-trained drafters were stronger on GSM8K/MATH-style
  workloads, while ShareGPT was stronger on MT-Bench. Mixed-domain checkpoints
  helped robustness but did not uniformly dominate. For this Qwen3-235B
  workstream, this supports collecting target Qwen3 SWE/RL rollout samples
  instead of treating DAPO/OpenMathInstruct as the primary corpus.
- **RL rollout alignment matters:** NVIDIA's April 2026 NeMo-RL speculative
  decoding report says in-domain post-training draft initialization improves
  the 8B RL-Zero rollout speedup from about 1.5x to 1.8x versus generic chat
  initialization. For this workstream, that strengthens the case for collecting
  Qwen3-235B SWE/RL rollout conversations first, then using math corpora only
  when the target workload is math reasoning.

For this Qwen3-235B SWE/RL workstream, treat SWE/RL as a specialized target,
not a general chat target. The recommended progression is:

```text
smoke: 5 repaired SWE-Gym rows, no training, runtime/capture proof only
pilot: 8 rollout conversations, 20 steps, proves hidden dump/train/export
calibration-1: 2.4k SWE-Gym rollout rows, one pass or <=1k train steps
calibration-2: 10k-50k target-domain rollout conversations if acceptance improves
production candidate: 50k-100k+ target-domain rollout conversations
general-purpose optional: 300k-500k only if we want a reusable broad Qwen3-235B draft outside SWE/RL
```

Do not train a serious draft directly on the five-row smoke or static human
answers. The useful samples are target-model generated responses from the same
distribution and prompt/template path as the RL rollout.

External source notes checked on 2026-05-23:

- EAGLE: https://arxiv.org/abs/2401.15077
  - 68,000 ShareGPT dialogue iterations; 70B-class draft head training reported
    as 1-2 days on 4x A100 40G.
- EAGLE-3: https://arxiv.org/abs/2503.01840
  - ShareGPT about 68k plus UltraChat-200K about 464k entries; responses are
    regenerated by the target model, and OpenThoughts-114k-math is added for
    the DeepSeek-R1-Distill reasoning draft.
- NVIDIA Qwen3-235B-A22B Eagle3 model card:
  https://huggingface.co/nvidia/Qwen3-235B-A22B-Eagle3
  - 503.3k data points, 100% training partition, synthetic responses generated
    from Qwen3-235B-A22B using UltraChat-200k and Magpie-Llama-3.1-Pro-300K
    prompts.
- vLLM Speculators online EAGLE3 tutorial:
  https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train_eagle3_online/
  - Qwen3-8B getting-started example uses `--max-samples 5000` and 5 epochs.
- Baseten EAGLE-3 training guide:
  https://www.baseten.co/blog/how-to-train-custom-eagle-3-heads-for-speculative-decoding/
  - Recommends about 100k samples for specialized tasks, about 500k for large
    generic heads, and about 1k-2k total tokens per sample.
- SpecForge data preparation:
  https://sgl-project.github.io/SpecForge/basic_usage/data_preparation.html
  - Pre-supported datasets include UltraChat 200k, ShareGPT 120k, and
    PerfectBlend 1.4M; it recommends target-model response regeneration for
    production performance.
- TorchSpec scale report:
  https://pytorch.org/blog/torchspec-speculative-decoding-training-at-scale/
  - Kimi K2.5 EAGLE-3 was trained at 600k samples / 6B tokens / 1500 H200 GPU
    hours, with about 7GB hidden states for one 128k-token sample.
- TAPS task-aware speculative sampling:
  https://arxiv.org/abs/2603.27027
  - Shows domain alignment matters; specialized drafters should be trained on
    data matching the downstream task distribution.
- NVIDIA NeMo-RL speculative decoding report:
  https://research.nvidia.com/labs/nemotron/rl-speculative-decoding/
  - Reports 1.5x-1.8x faster rollout generation on 8B reasoning workloads,
    projected gains at 235B scale, and a concrete in-domain initialization
    effect for RL-Zero.

## Backend Comparison

| backend | what it gives us | current fit | decision |
| --- | --- | --- | --- |
| ModelOpt Eagle3 | Existing local/remote wrappers for hidden-state dump, offline train, export, validators, and RL smoke/sweep. Matches the Hayate/ModelOpt path already analyzed. | Best immediate fit for the current NeMo-RL Qwen3-235B workstream. | Primary path. Do not switch before rollout smoke proves capture. |
| vLLM Speculators | vLLM-native library for training speculators and serving them directly in vLLM. Supports online and offline hidden-state training through vLLM hidden extraction. Public docs show `speculators>=0.5.0` and `vllm>=0.18` for the online tutorial. | Strong future target if we want vLLM-native draft training/export. It is a separate training stack from the current NeMo 25.07.01 + source-built vLLM 0.10.2 runtime. | Track as second backend. First build a data adapter and environment probe; do not replace the active RL smoke path. |
| SpecForge | SGLang ecosystem trainer with EAGLE3 support, Qwen-family examples, `--target-model-backend sglang`, and simple `id/conversations` or preformatted-text input. | Useful reference and possible path if serving moves to SGLang. Less direct for current NeMo-RL/vLLM rollout validation. | Keep as comparison/reference, not the main path. |

## Why ModelOpt Remains Primary

The immediate blocker is not "which trainer is nicest"; it is proving that the
Qwen3-235B NeMo-RL SWE loop can run with the patched vLLM runtime and emit
trainable logs. The ModelOpt path is already wired to that evidence chain:

- `run_rollout_capture_smoke.sh` submits the NeMo-RL rollout capture.
- `normalize_rl_rollouts_to_conversations.py` converts `train_data_step*.jsonl`
  into ModelOpt conversations.
- `modelopt_qwen3_235b_dump_hidden_states.sh`,
  `modelopt_qwen3_235b_offline_train.sh`, and
  `modelopt_qwen3_235b_export_vllm.sh` form the draft pipeline.
- `preflight_eagle3_pipeline_submit.py`, checkpoint validators, export
  validators, and trained-draft sweep wrappers keep the RL context attached.

Switching to vLLM Speculators or SpecForge before the patched smoke produces a valid
capture would not solve the current uncertainty, because both still need
high-quality target responses in the actual SWE/RL distribution.

## vLLM Speculators Notes

External references checked on 2026-05-22:

- https://github.com/vllm-project/speculators
- https://docs.vllm.ai/projects/speculators/en/latest/user_guide/getting_started/
- https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train_eagle3_online/
- https://docs.vllm.ai/projects/speculators/en/latest/cli/prepare_data/
- https://docs.vllm.ai/projects/speculators/en/latest/cli/train/

Key points:

- The project describes itself as a library for training speculative decoding
  draft models that deploy directly to inference engines like vLLM.
- Its supported-model table lists Qwen3 MoE 235B variants with EAGLE3 training
  and vLLM deployment support.
- The online EAGLE3 tutorial installs `speculators>=0.5.0` in one environment
  and `vllm>=0.18` in a separate serving environment.
- `prepare_data.py` applies the chat template, tokenizes samples, builds the
  assistant/loss mask, and writes processed data for online or offline hidden
  state generation.
- `train.py` supports online, offline, and hybrid hidden-state modes. Online
  training uses a vLLM endpoint and `--on-missing generate`; offline training
  uses pre-generated hidden states with `--on-missing raise`.

This is promising, but it is not the same runtime as our current source-built
`vllm 0.10.2` site. To use it responsibly here, we need:

1. A clean `speculators` checkout or package environment on `oci-hsg`.
2. A vLLM version/environment compatible with Qwen3-235B MoE hidden extraction
   on the available H100/GB200 hardware.
3. A converter from our normalized rollout conversations to the exact
   `prepare_data.py` accepted local JSONL schema. The first local helper now
   exists as `convert_conversations_to_speculators_jsonl.py`; it still needs a
   real rollout corpus and a cluster-side `prepare_data.py` smoke.
4. A small `Qwen3-235B` or smaller Qwen3-MoE smoke that proves hidden extraction
   and FSDP training in this cluster.
5. A serving smoke that proves the exported speculator works inside the same
   RL generation stack, not only standalone vLLM.

## SpecForge Notes

External references checked on 2026-05-22:

- https://sgl-project.github.io/SpecForge/basic_usage/data_preparation.html
- https://github.com/sgl-project/SpecForge/blob/main/examples/run_qwen3_235b_a22b_eagle3.sh

Key points:

- SpecForge supports custom JSONL in conversation format:
  `{"id": "...", "conversations": [{"role": "user|assistant", "content": "..."}]}`.
- It also accepts preformatted text with `--is-preformatted`; the matching chat
  template is still needed to identify assistant spans and build the loss mask.
- The referenced example filename says Qwen3-235B-A22B, but the current visible
  command targets `Qwen3-Next-80B-A3B-Instruct-FP8`, uses
  `configs/qwen3-next-80b-a3b-eagle3.json`, and sets
  `--target-model-backend sglang`.

SpecForge is therefore a good SGLang-side reference, but it should not replace
the ModelOpt/NeMo-RL path unless the serving target also moves to SGLang.

## Next Actions

1. Inspect probe `2867766` once SSH/DNS recovers. The next-action planner now
   treats `megatron_compat_probe.json` as a rollout gate, so a missing/pending
   probe promotes a probe poll or submit action before any heavy rollout.
   The guarded shortcut is
   `bash experiments/eagle3_qwen3_235b/followup_megatron_probe_to_rollout.sh`;
   it prints the rollout retry after PASS and only submits with
   `SUBMIT_ROLLOUT=true ALLOW_HEAVY_GPU=true`.
2. If it passes and materializes conversations, inspect the normalized corpus
   and rerun `submit_full_rollout_after_smoke_if_ready.py`. If it returns
   `READY`, submit the full SWE-Gym rollout capture using the 2,438-row input.
3. Run the ModelOpt pilot only after the canonical rollout corpus validates:
   8 conversations, 20 steps, 2h/2h/1h dump/train/export limits.
4. Add a vLLM Speculators probe as a separate branch:
   environment install, data adapter, `prepare_data.py` dry-run, hidden
   extraction smoke, tiny train smoke.
5. Keep SpecForge as SGLang comparison; use `--output-schema specforge` from
   the normalizer when a comparison dataset is needed.
