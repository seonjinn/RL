# Qwen3-235B CuTeDSL Performance Design

- **Approved approach:** 2026-07-13
- **Primary cluster:** Pre-Tyche
- **Source branch:** `sna/nemo-2606-cutedsl-a2a-factorial-20260712`
- **Model:** `Qwen/Qwen3-235B-A22B`
- **Primary feature:** CuTeDSL fused grouped MLP

## Purpose

Measure the policy-training impact of the NeMo 26.06 CuTeDSL fused grouped-MLP
path on the repository's official Qwen3-235B GB200 performance topology. The
experiment must preserve the official model, workload, parallelism, and rollout
precision while changing only the CuTeDSL selector between accepted timing
arms.

This is the large-model scalability follow-up to Qwen3-30B-A3B. It does not
claim full-iteration CUDA Graph or expert-parallel A2A-overlap support.

## Official Base Configuration

The base recipe is:

```text
examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g.yaml
```

The resolved fixed workload is:

| Item | Value |
|---|---|
| Nodes and GPUs | 16 nodes, 4 GB200 GPUs per node, 64 GPUs total |
| Segment size | 16 nodes |
| Prompts and generations | 16 prompts × 32 generations |
| Global and micro batch | GBS512, MBS1 |
| Logprob batch | 1 |
| Maximum sequence length | 8192 |
| Policy topology | TP2, PP4, CP2, EP16, ETP1 |
| Policy schedule | sequence parallel and activation checkpointing enabled |
| MoE dispatcher | flex dispatcher with HybridEP, 32 SMs |
| Generation | vLLM TP8, asynchronous engine, BF16, 0.4 GPU-memory utilization |
| Dataset | `nvidia/OpenMathInstruct-2`, `train_1M` |

The performance harness disables validation and checkpoint saving identically
in every timing arm. These operations are not policy-training work, occur inside
the measured step timer, and can add large CPU-memory and storage side effects.
The model, data, batch, sequence length, and parallel topology remain unchanged.

## Policy MXFP8 Overlay

CuTeDSL requires the policy expert path to use MXFP8. A new 235B overlay enables:

```text
policy.megatron_cfg.moe_router_dtype=fp32
policy.megatron_cfg.use_transformer_engine_op_fuser=true
policy.megatron_cfg.moe_mlp_glu_interleave_size=32
policy.megatron_cfg.fp8_cfg.enabled=true
policy.megatron_cfg.fp8_cfg.fp8=e4m3
policy.megatron_cfg.fp8_cfg.fp8_recipe=mxfp8
policy.megatron_cfg.fp8_cfg.fp8_param=false
```

The existing `grpo-qwen3-235b-16n4g-mxfp8-rollout.yaml` is not the base for
this comparison because it also changes rollout precision and vLLM's MoE
backend. Rollout remains BF16 so generation does not become a second feature
variable.

The two accepted arms differ only in:

```text
policy.megatron_cfg.env_vars.NVTE_CUTEDSL_FUSED_GROUPED_MLP="0"  # OFF
policy.megatron_cfg.env_vars.NVTE_CUTEDSL_FUSED_GROUPED_MLP="1"  # ON
```

Every other resolved config field and environment selector must match after
normalizing result paths, logger names, and the one feature selector.

## Feature Boundary

The first 235B experiment keeps these settings disabled in both arms:

```text
cuda_graph_impl=none
overlap_moe_expert_parallel_comm=false
high_priority_a2a_comm_stream=false
delay_wgrad_compute=false
```

The official 235B topology uses PP4. The current approved NeMo-RL A2A-overlap
boundary does not support enabling the combined schedule on this PP topology,
and full-iteration CUDA Graph does not yet support the complete RL
generation/refit/logprob lifecycle. Neither feature may be silently enabled or
reported as measured.

## Harness Architecture

The existing 30B script contains the required fail-closed timing, workload, and
attribution machinery but hard-codes model and topology assertions. The 235B
path reuses that engine through an explicit model-profile interface instead of
copying the approximately two-thousand-line matrix payload.

The common harness gains typed, required profile inputs for:

- experiment and result root;
- model and tokenizer name;
- recipe path;
- node, GPU, and segment counts;
- prompts, generations, GBS, MBS, and logprob batch;
- sequence length;
- TP, PP, CP, EP, and ETP;
- rollout precision and generation TP;
- dispatcher type and HybridEP settings;
- expected policy precision and CuTeDSL prerequisites.

A new model-specific experiment directory supplies the 235B values, owns its
results and HTML report, and exposes a thin Pre-Tyche submission wrapper. Missing
or inconsistent profile inputs fail before scheduler submission. The 30B
profile remains the default only for its existing wrapper; the common engine
does not infer a model from a recipe filename.

## Cache and Initialization

The 235B path depends on the approved node-local Triton cache isolation design.
No 16-node job is submitted while policy ranks share one Lustre Triton cache.

One locked shared Hugging Face cache warm-up resolves and records the exact
40-character revisions for:

- `Qwen/Qwen3-235B-A22B`;
- `nvidia/OpenMathInstruct-2`.

It materializes and verifies the exact one-million-row `train_1M` dataset in a
fresh offline process. The benchmark then sets Hub, Transformers, and Datasets
offline before creating Ray actors. The warm-up records file count and bytes so
the 235B snapshot cannot exhaust shared storage without a bounded error.

Megatron checkpoint conversion uses a job-scoped shared path visible to all
sixteen nodes. Concurrent ranks never convert independently, and a completed
conversion is verified before policy initialization.

## Staged Execution

### Stage 1: scheduler and cache preflight

- verify source, recursive submodules, upstream SHA, and pinned image SHA;
- run the exact 16-node `sbatch --test-only` request;
- verify cache capacity before model download;
- warm and offline-verify model and dataset snapshots.

### Stage 2: functional gate

Run one 16n4g CuTeDSL-ON job for three optimizer updates. It must cross
generation, refit, policy/reference logprob, PolicyTraining, and the next mature
offload boundary. This run is never performance evidence.

### Stage 3: timing-duration pilot

Run one unprofiled OFF/ON pilot with five warmups and five measured updates per
arm. Use it only to verify that a full paired job has at least 30 minutes of
margin under Pre-Tyche's five-hour limit. Pilot samples are excluded from the
final aggregate.

If the projected full paired job is at most 4.5 hours, retain paired timing jobs.
Otherwise, freeze an arm-separated six-job design before collecting accepted
data. Arm-separated data uses balanced submission order and independent
two-sample bootstrap; it is never mixed with paired statistics.

### Stage 4: accepted timing matrix

The preferred paired contract uses three replicas:

1. ON then OFF;
2. OFF then ON;
3. ON then OFF.

Each arm runs five warmup and twenty measured updates. A timing job contains no
Nsight capture. All three replicas must use one source SHA, image SHA, model and
dataset revision, base-config hash, and metric schema.

### Stage 5: profile-only attribution

Run CuTeDSL ON and OFF as separate two-update profile jobs after timing succeeds.
Profile jobs cannot change a completed timing job's Slurm status. The retained
Nsight evidence must show grouped expert GEMMs in both arms, the CuTeDSL fused
GLU/dGLU path only when enabled, and the exact feature selector in the manifest.

## Workload and Metric Contract

Measured ON/OFF samples must match exactly for:

- measured step sequence;
- mean prompt length;
- valid sample count;
- total turns;
- normalized resolved config and all non-feature selectors.

On-policy response lengths may diverge after the first optimizer update. Total
and valid-token aggregates therefore use the existing symmetric limits: at most
1% arm-total delta and 2% maximum paired-step delta. Every throughput sample is
recomputed from that arm's actual token count.

The primary endpoint is PolicyTraining tokens/s/GPU. Secondary endpoints are:

- PolicyTraining latency;
- E2E step latency and tokens/s/GPU;
- policy/reference Logprob latency and tokens/s/GPU;
- Generation latency and tokens/s/GPU;
- generation finalization;
- refit transfer/update latency;
- peak GPU and host-cgroup memory.

Report paired per-replica effects, geometric-mean speedup, bootstrap confidence
intervals, raw medians, workload-equivalence evidence, and order sensitivity.
Refit or generation is reported as neutral/inconclusive when replicate direction
is inconsistent.

## TDD and Review Contract

Implementation begins with failing tests that prove:

1. the 235B profile resolves the exact official workload and topology;
2. MXFP8 policy prerequisites are identical across arms;
3. the only accepted ON/OFF difference is the CuTeDSL selector;
4. rollout remains BF16 and the MXFP8-rollout recipe is not selected;
5. full-CG and A2A selectors are false and fail closed if requested;
6. HF cache preparation uses the 235B model revision and offline verification;
7. node-local Triton cache scope is required before 16-node submission;
8. timing and profile jobs are separate and cannot share acceptance status;
9. collectors reject mixed 30B/235B, topology, source, image, or revision data;
10. public reports contain no internal paths, hostnames, IPs, credentials, or
    raw worker logs.

Focused tests, the existing 30B regression suite, Ruff, Bash syntax, config
resolution, and recursive source-cleanliness checks must pass before cluster
submission. An independent code review is required before push.

## Acceptance and Reporting

No 235B speedup is claimed until:

- the functional gate passes;
- three accepted timing replicas pass workload equivalence;
- the aggregate collector accepts every required component series;
- separate ON/OFF profile jobs pass kernel attribution;
- the HTML report records source, image, revisions, config, job IDs, root-cause
  incidents, and reproducible collection commands.

Any failed stage produces bounded incident evidence and one root-cause fix before
retry. A smaller model, shorter sequence, lower batch, or different topology is
not substituted for the official 16n4g performance result.
