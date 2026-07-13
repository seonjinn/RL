# Qwen3-235B Latest-Main Full-GRPO PARD Summary

Status timestamp: 2026-06-07 15:35 PDT, updated 17:35 PDT

Update as of 17:00 PDT: the current stable PARD/Full-GRPO probes are still
short fixed-decode runs. Qwen3-235B jobs `3208028` and `3208066` both use
`policy.generation.max_new_tokens=256`, `policy.max_total_sequence_length=8192`,
`policy.generation.vllm_cfg.max_model_len=8192`, and
`max_num_batched_tokens=8192`. They are not 8K/16K generation-length tests.

## Current Fixed-Runner Submission

The older Qwen3-235B public PARD jobs failed before useful NeMo-RL metrics
because the launcher used `uv run` and hit a workspace parse error around
`nemo-gym`. A first fixed-runner retry used a `16n8g` base recipe with 32n4g
overrides; those pending jobs were cancelled before starting so that the active
retry can use the official `32n4g` recipe base directly.

The active retry uses the nightly container venv python directly:

`/opt/nemo_rl_venv/bin/python examples/run_grpo.py ...`

The intent is to keep the recipe/config path simple and only change the runner
needed to make latest-main nightly execute inside the container.

The first official-32n4g submissions, `3207450`, `3207451`, and `3207452`,
were cancelled because they had the same `/home/sna` workdir/Ray marker issue
as the first Qwen3-30B-A3B retry. A `SEGMENT=32` retry was rejected by Slurm
because the requested topology was unavailable. The active `SEGMENT=4`
resubmissions are:

| Job | Variant | Status | Notes |
|---:|---|---|---|
| 3207511 | Baseline | failed after 15 emitted result blocks, reached Step 16 | official `grpo-qwen3-235b-32n4g.yaml` base, full GRPO, 20 steps |
| 3207512 | PARD K=3 | FAILED | vLLM 0.20 rejected `draft_tensor_parallel_size=1` with target generation TP=4 |
| 3207513 | PARD K=5 | FAILED | vLLM 0.20 rejected `draft_tensor_parallel_size=1` with target generation TP=4 |

The PARD failures are configuration failures, not performance results:

```text
ValueError: Currently, 'draft_tensor_parallel_size' and 'tensor_parallel_size' must be the same.
Got 1 and 4.
```

vLLM 0.20 requires the drafter TP to match the target model TP in this path.
The PARD retries with `draft_tensor_parallel_size=4` are:

| Job | Variant | Status | Notes |
|---:|---|---|---|
| 3207654 | PARD K=3 | failed after Step 1 result | generation TP=4, draft TP=4 |
| 3207655 | PARD K=5 | failed after Step 1 result | generation TP=4, draft TP=4 |

Superseded pending jobs `3207415`, `3207416`, and `3207417` were cancelled at
elapsed `00:00:00`.

Key run shape:

| Field | Value |
|---|---|
| Base config | `examples/configs/recipes/llm/performance/grpo-qwen3-235b-32n4g.yaml` |
| Target model | `Qwen/Qwen3-235B-A22B` |
| Drafter | public PARD `amd/PARD-Qwen3-0.6B` local snapshot |
| Nodes / GPUs | 32 nodes x 4 GPUs |
| Generation TP | 4 |
| Training TP / PP / CP / EP | 2 / 8 / 2 / 16 |
| `policy.generation.max_new_tokens` | 256 |
| Fixed decode | `min_tokens=256`, ignore EOS/stop |
| Sampling | temperature 1.0, top_p 1.0, top_k -1 |
| `policy.generation.vllm_cfg.async_engine` | `false` |
| `grpo.async_grpo.enabled` | `false` |
| `policy.draft.enabled` | `false` |
| `grpo.max_num_steps` | 20 |

Latest baseline timing evidence: `3207511` emitted 15 training-result blocks
before failing around Step 16 with NCCL timeout / actor-death signatures. On
the Step 2+ window, the average total step time is `76.59s`, generation time is
`34.77s`, and baseline generation fraction is about `45.4%`. This baseline
fraction is the denominator for any future Qwen3-235B PARD E2E claim;
generation-only improvement will not map 1:1 to total step speedup.

The TP4 PARD retries passed the earlier vLLM config error and reached vLLM
engine initialization with `speculative_config=SpeculativeConfig(...)`,
`tensor_parallel_size=4`, and GPU KV cache size `240,240` tokens per engine.
They each emitted one Step 1 result and then failed at Step 2 generation start
because Ray killed policy workers due node host-memory pressure.

| Job | Variant | Step 1 total | Step 1 generation | Step 1 E2E tok/s/GPU | Step 1 gen tok/s/GPU | Acceptance | Failure |
|---:|---|---:|---:|---:|---:|---:|---|
| `3207654` | PARD K=3 | `289.26s` | `26.64s` | `2.56` | `27.79` | mean `66.65%` from emitted buckets | Ray node memory OOM at Step 2 |
| `3207655` | PARD K=5 | `292.85s` | `26.30s` | `2.53` | `28.15` | mean `50.8%` from emitted buckets | Ray node memory OOM at Step 2 |

Compared with the baseline Step 2+ average generation time (`34.77s`), the
Step 1 PARD generation slice is faster (`1.31x` K3, `1.32x` K5). However, Step
1 policy training is a warmup-heavy `242-248s`, so this is not a valid E2E
speedup claim. A memory-safer retry must survive multiple steps before
averaging E2E metrics.

The latest-main worktree includes NeMo-RL PR #2658:

```text
e94d33c88 revert: Discard weight when finish generation in the main loop (#2495) (#2658)
```

So the current failures are not due the old #2495 weight-discard sleep level.
They are host-memory pressure in Ray/Megatron policy workers for Qwen3-235B
PARD TP4, and vLLM CuMem wake-up headroom for the Qwen3-30B-A3B long-OSL
diagnostic.

A short K3 retry was submitted to test whether the Ray host-memory kill is the
immediate blocker:

| Job | Variant | Shape | Purpose |
|---:|---|---|---|
| `3207770` | public PARD K=3 | official 32n4g recipe, GBS256, generation TP4, draft TP4, `MAX_STEPS=5`, `RAY_memory_usage_threshold=0.99` | Cancelled/replaced before useful metrics; actor startup hit `ModuleNotFoundError: No module named 'ray'` because this submit still used the actor-venv path (`NEMO_RL_PY_EXECUTABLES_SYSTEM=0`) |
| `3207856` | public PARD K=3 | same official 32n4g recipe and TP shape, `NEMO_RL_PY_EXECUTABLES_SYSTEM=1`, fixed decode 256, `RAY_memory_usage_threshold=0.99` | Failed before useful metrics; plain system Python actor path did not have `vllm` installed |
| `3207963` | public PARD K=3 | same official 32n4g recipe and TP shape, `NEMO_RL_PY_EXECUTABLES_SYSTEM=0`, `NEMO_RL_VENV_DIR=/opt/ray_venvs`, recursive submodules initialized | Cancelled after K5 showed the same remote-HF target weight path risk |
| `3207964` | public PARD K=5 | same as `3207963`, K=5 | Failed during vLLM target weight loading on some workers |
| `3208028` | public PARD K=3 | same shape, but `policy.model_name` / `TARGET_MODEL_ID` use the local Qwen3-235B snapshot path | Failed after Step 1; fixed the local-HF weight lookup class but hit Ray host-memory OOM at Step 2 |
| `3208029` | public PARD K=5 | same as `3208028`, K=5 | Failed before driver launch due one Ray worker startup timeout, not due model loading or PARD quality |
| `3208066` | public PARD K=5 | same local-HF shape as `3208029`, with `RAY_raylet_start_wait_time_s=180` passed through `sbatch --export` | Failed after Step 1; the Ray wait fix worked, but the same TP4 host-memory OOM appeared at Step 2 |

This retry is not meant as a final E2E claim unless it survives multiple steps.
It is a stability probe for the observed host-memory failure mode.

The failed `3207770` and `3207856` evidence is important because it separates
three failure classes. `3207770` did not test the Ray host-memory threshold
because it never reached vLLM engine setup; Ray workers crashed during actor
initialization with:

```text
ModuleNotFoundError: No module named 'ray'
```

`3207856` used `NEMO_RL_PY_EXECUTABLES_SYSTEM=1`, which avoided the actor venv
builder but mapped VLLM actors to plain system Python. That also did not reach
vLLM setup because the actor process then failed with:

```text
ModuleNotFoundError: No module named 'vllm'
```

The underlying actor-venv failure was a missing recursive submodule checkout:
`3rdparty/Gym-workspace/Gym` and `3rdparty/Automodel-workspace/Automodel` were
present as empty submodule directories. Because `pyproject.toml` declares
`nemo_gym = { workspace = true }`, bare actor `uv sync` could not resolve the
workspace. The corrective action was:

```text
git submodule update --init --recursive \
  3rdparty/Gym-workspace/Gym \
  3rdparty/Automodel-workspace/Automodel
```

The corrected active job record is:

```text
experiments/eagle3_online/latest_qwen235b_public_pard_k3k5_submodulefix_step5_jobs.txt
QWEN235_public_pard_k3=3207963
QWEN235_public_pard_k5=3207964
```

The `3207964` failure was not a PARD quality or acceptance failure. It reached
vLLM actor execution and then failed because remote id resolution was not
reliable on all workers:

```text
RuntimeError: Cannot find any model weights with `Qwen/Qwen3-235B-A22B`
```

The corrective retry switches the target model id from the remote HF id to the
local snapshot already available on Lustre:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--Qwen--Qwen3-235B-A22B/snapshots/8efa61729e24bd65b1d152b5ab5409052aa80e65
```

New local-HF target jobs:

```text
QWEN235_PUBLIC_PARD_LOCALHF_K3=3208028
QWEN235_PUBLIC_PARD_LOCALHF_K5=3208029
QWEN235_PUBLIC_PARD_LOCALHF_K5_RAYWAIT=3208066
```

The `3208029` failure class is separate from the earlier remote-HF weight path
failure. It never launched the GRPO driver. `ray-worker-20` retried
`ray start --address ...` three times and failed with:

```text
Exception: The current node timed out during startup.
GCS cannot find the node with node ID ...
Try increase the RAY_raylet_start_wait_time_s config.
```

The corrected `3208066` kept the same K5 local-HF target/drafter settings and
only increased Ray worker startup wait time. It reached `128/128` Ray workers
and emitted Step 1 metrics, proving the original `ray-worker-20` timeout was an
operability issue rather than a K5/PARD failure. It then failed at Step 2 with
the same Ray host-memory OOM pattern as K3 TP4.

Latest TP4 local-HF PARD evidence:

| Job | Variant | Step 1 total | Step 1 generation | Step 1 E2E tok/s/GPU | Step 1 gen tok/s/GPU | Acceptance | Failure |
|---:|---|---:|---:|---:|---:|---:|---|
| `3208028` | PARD K=3, TP4 | `293.60s` | `27.99s` | `2.52` | `26.45` | mean `60.60%` from emitted buckets | Ray host-memory OOM at Step 2 |
| `3208066` | PARD K=5, TP4 | `283.95s` | `25.91s` | `2.61` | `28.57` | mean `53.35%` from emitted buckets | Ray host-memory OOM at Step 2 |

This makes the TP4 conclusion stronger: PARD K3/K5 can initialize and produce
generation metrics, but colocated Qwen3-235B TP4 does not survive multi-step
Full-GRPO because node host memory is too tight. The Ray memory killer reports
node RAM around `879.9GB / 920GB`; top memory users are the vLLM
`RayWorkerWrapper` processes plus Megatron policy/offload workers, not the
small drafter model alone.

The next active retry increases generation TP from 4 to 8 and also submits a
matched TP8 baseline so that the PARD result is not compared against a different
generation-TP baseline:

| Job | Variant | Shape | Current status |
|---:|---|---|---|
| `3208890` | baseline, no SpecDec | 32n4g, GBS256, generation TP8, fixed decode 256 | Completed 5/5 metrics. Step 2-5 avg: total `77.84s`, generation `35.44s`, E2E `9.16 tok/s/GPU`, generation `20.06 tok/s/GPU`; generation fraction `45.53%`. Shutdown emitted `recvBytes` messages after metrics, but no step metric was missing. |
| `3208891` | public PARD K=3 | 32n4g, GBS256, generation TP8, draft TP8, fixed decode 256 | Failed after Step 4 total / Step 5 acceptance buckets with `VllmGenerationWorker` ActorDied and NCCL timeout signatures. Step 2-4 avg vs TP8 baseline Step 2-5 avg: total step `69.56s` vs `77.84s`, or `1.119x`; generation `22.33s` vs `35.44s`, or `1.587x`; E2E `10.38` vs `9.16 tok/s/GPU`, or `1.133x`; generation throughput `31.38` vs `20.06 tok/s/GPU`, or `1.564x`. Acceptance avg `58.20%`. This is a positive but partial cross-node TP8 signal, not the clean final result. |

Success gate for this branch: both TP8 baseline and TP8 PARD need to survive at
least Step 2, then compare Step 2+ windows. Step 1 alone is not a valid E2E
performance claim because Qwen3-235B policy/training warmup dominates it.

TP8 has an important performance caveat on this cluster: each node exposes 4
GPUs, so vLLM warns that `tensor_parallel_size=8` is spread across nodes. This
is useful for testing whether more TP ranks reduce the TP4 host-memory OOM, but
it is not the cleanest final performance measurement because cross-node TP can
add communication overhead.

To directly target the host-memory failure while keeping generation TP=4, a
non-colocated TP4 pair was submitted. This keeps 32 nodes for training and adds
4 generation-only nodes, so the vLLM target/draft workers and Megatron policy
workers no longer share host RAM on the same nodes:

| Job | Variant | Shape | Current status |
|---:|---|---|---|
| `3209047` | baseline, no SpecDec | 36n4g total: 32 train nodes + 4 generation nodes, generation TP4, fixed decode 256, GBS256 | Completed 5/5 timing metrics. Driver confirms `policy.generation.colocated.enabled=false`, `resources.num_nodes=4`, and parallel worker initialization in non-colocated mode. Step 2-5 avg: total `92.75s`, generation `57.77s`, E2E `6.88 tok/s/GPU`, generation `98.33 tok/s/GPU`; baseline generation fraction `62.29%`. |
| `3209048` | public PARD K=3 | same non-colocated shape, generation TP4, draft TP4, always-on PARD | Completed 5/5 timing metrics. Driver confirms non-colocated resources and `speculative_config.method=draft_model`, `num_speculative_tokens=3`, `draft_tensor_parallel_size=4`, `parallel_drafting=true`. Step 2-5 avg: total `65.25s`, generation `30.61s`, E2E `9.77 tok/s/GPU`, generation `185.62 tok/s/GPU`; acceptance avg `57.58%` from 20 emitted buckets. |

This non-colocated pair is the cleaner performance branch and is now the best
Qwen3-235B NeMo-RL Full-GRPO evidence. On the matched Step 2-5 window, public
PARD K3 gives total-step speedup `1.421x`, generation-time speedup `1.887x`,
E2E throughput speedup `1.420x`, and generation-throughput speedup `1.888x`.
The key reason this finally propagates to E2E is that the non-colocated TP4
baseline spends about `62.3%` of the steady-state step in generation, unlike
the shorter/colocated settings where generation was a much smaller fraction.

The 20-step extension of the same non-colocated TP4 fixed256 shape is now the
active stability gate. The first baseline/PARD attempts both reached Step 16 and
then failed with 600s Megatron NCCL watchdog collective timeouts, not OOM. This
means the Step16 stability issue is not PARD-specific.

| Job | Variant | Current status |
|---:|---|---|
| `3210070` | public PARD K3, 20-step | Failed after Step 16 with 600s NCCL watchdog timeouts. Matched against baseline retry `3210159` on Step 2-16: total-step `1.454x`, generation-time `1.836x`, E2E throughput `1.461x`, generation-throughput `1.838x`; acceptance `57.70%`. |
| `3210159` | baseline, 20-step retry | Failed after Step 16 with the same 600s NCCL watchdog timeout class. Step 2-16 avg: total `88.00s`, generation `56.78s`, E2E `7.70 tok/s/GPU`, generation `107.01 tok/s/GPU`. |
| `3210513` | public PARD K3, timeout=1800 retry | Running. The patched `nemo_rl/models/megatron/setup.py` path is active; driver log confirms `Initializing Megatron NCCL process group with timeout=1800s`. Step 2-12 early window: total `61.02s`, generation `30.77s`, E2E `10.69 tok/s/GPU`, generation `188.68 tok/s/GPU`; acceptance avg `57.84%` over 53 buckets. This is not a final claim until the 20-step gate passes. |
| `3210580` | baseline, timeout=1800 retry | Running as the fair timeout-patched baseline counterpart to `3210513`; driver log confirms `timeout=1800s`. Step 1 warmup emitted total `306.22s`, generation `60.26s`, E2E `2.15 tok/s/GPU`, generation `98.27 tok/s/GPU`; use Step 2+ for matched comparison. |

The timeout patch adds an explicit `timeout=timedelta(seconds=$NRL_MEGATRON_NCCL_TIMEOUT_SECONDS)`
argument to `torch.distributed.init_process_group("nccl")`; the submit scripts
already set `NRL_MEGATRON_NCCL_TIMEOUT_SECONDS=1800`, but the unpatched setup
code left PyTorch at the 600s default.

The active K3 local-HF job `3208028` has now passed the previous local-path
failure class. The driver log confirms vLLM `v0.20.0` and:

```text
speculative_config=SpeculativeConfig(... num_spec_tokens=3)
model=local Qwen3-235B-A22B snapshot
draft model=local amd/PARD-Qwen3-0.6B snapshot
tensor_parallel_size=4
enforce_eager=True
max_seq_len=8192
```

It is currently loading target model shards. vLLM also emits a performance
warning for this shape:

```text
max_num_scheduled_tokens is set to 8096 based on the speculative decoding
settings. This may lead to suboptimal performance. Consider increasing
max_num_batched_tokens to accommodate the additional draft token slots, or
decrease num_speculative_tokens or max_num_seqs.
```

Therefore, if K3/K5 completes but underperforms, the first config-level follow
up should be a higher `max_num_batched_tokens` probe, balanced against the
already tight Qwen3-235B memory envelope.

Update as of 16:40 PDT: K3 `3208028` has progressed past vLLM target load on
multiple engines. The driver log reports:

```text
Model loading took 109.84 GiB memory
GPU KV cache size: 240,240 tokens
Using vLLM backend for generation with local Qwen3-235B-A22B
128 policy workers initialized in 71.81s
```

It has not emitted a GRPO `Step 1/5` marker yet. K5 `3208066` has passed the
Ray startup issue and reached driver/vLLM worker initialization, but likewise
has not emitted a step metric yet.

Update as of 17:00 PDT: K3 `3208028` emitted a Step 1 result but failed at
Step 2 with Ray host-memory OOM, not CUDA OOM:

| Job | Variant | Step 1 total | Step 1 generation | Acceptance | Failure |
|---:|---|---:|---:|---:|---|
| `3208028` | public PARD K=3, local-HF target, TP4 | `293.60s` | `27.99s` | `60.6%` avg over emitted buckets | Ray killed a policy worker when node RAM reached `879.87GB / 920.00GB` |

The Ray OOM top-memory list explains the root cause. On the killed node, four
vLLM `RayWorkerWrapper` processes held about `150GB` each, and four
`MegatronPolicyWorker` processes held about `63GB` each. This is a colocated
TP4 host-DRAM problem: vLLM keeps very large Qwen3-235B CPU-side target weight
shards, then Megatron policy workers are also resident during the next step.
The small PARD drafter is not the dominant memory consumer here.

K5 `3208066` has now reached `Step 1/5` with the same TP4 colocated shape. It
may still provide a Step 1 generation/acceptance datapoint, but it should be
expected to have the same Step 2 host-memory risk unless the worker placement or
generation TP changes.

The next corrective probe should avoid repeating TP4 colocated. Two viable
directions are:

1. Increase Qwen3-235B generation TP from 4 to 8 or 16 so each vLLM rank holds a
   smaller CPU weight shard while keeping 32 training nodes.
2. Use non-colocated inference resources so vLLM and Megatron policy workers do
   not share node DRAM. This is cleaner but requires extra inference nodes or a
   smaller train allocation.

The immediate follow-up is TP8 colocated fixed-decode first. If that survives
multiple GRPO steps, then longer OSL and non-colocated throughput probes can be
made without confounding the current host-memory failure.

Submitted TP8 fixed-decode follow-up jobs:

| Job | Variant | Purpose |
|---:|---|---|
| `3208890` | no-spec baseline, generation TP8 | same TP as the PARD probe, needed for a fair comparison after changing generation TP |
| `3208891` | public PARD K=3, generation TP8, draft TP8 | host-memory mitigation probe; first gate is surviving Step 2 without Ray node-memory kill |

## Previous No-Stop Jobs

| Label | Job | Status | Steps | Gen TPS speedup | Gen time speedup | E2E TPS speedup | E2E step speedup | Acceptance | Latest error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 3207165 | missing_log |  |  |  |  |  |  |  |
| public_pard_k3_temp1_step5 | 3207001 | missing_log |  |  |  |  |  |  |  |
| public_pard_k3_temp1_step20 | 3207103 | missing_log |  |  |  |  |  |  |  |
| public_pard_k5_temp1_step20 | 3207104 | missing_log |  |  |  |  |  |  |  |
