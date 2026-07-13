# Qwen3-30B-A3B Full-GRPO 20-Step Stability Run

Date: 2026-06-06 PDT

## Purpose

The immediate Qwen3-30B-A3B goal was to get an OOM-free NeMo-RL Full-GRPO run
for about 20 steps before expanding the performance sweep. Earlier
latest-main/vLLM0.20 Qwen3-30B-A3B baseline and PARD jobs emitted Step 1-2
metrics, then hit Step 3 vLLM CuMem allocator OOM during the colocated
sleep/wake/refit lifecycle.

The completed retry keeps the same functional path but uses the conservative
memory envelope that stabilized Qwen3-32B:

```text
gpu_memory_utilization=0.80
max_num_batched_tokens=16384
max_num_seqs=32
max_model_len=4096
```

## Completed Jobs

| Model | Job | Mode | K | Status | Shape |
|---|---:|---|---:|---|---|
| Qwen3-30B-A3B | `3198446` | baseline | 0 | `COMPLETED 0:0`, 20/20 steps | 4n4g, GBS 512, generation TP 1, train TP 1, EP 16 |
| Qwen3-30B-A3B | `3198447` | public PARD | 3 | `COMPLETED 0:0`, 20/20 steps | 4n4g, GBS 512, generation TP 1, train TP 1, EP 16 |

Generation uses `max_new_tokens=256`, `temperature=1.0`, `top_p=1.0`, and
`top_k=-1`. With `NUM_PROMPTS=16` and `NUM_GENERATIONS=32`, the global batch is
`512`; across 16 vLLM engines this is the worker-batch-32 setting that better
matches the standalone batch-size-32 comparison. The `3198447` driver log
confirms `generation_batch_size=32` and the `amd/PARD-Qwen3-0.6B` K3
`speculative_config`. The baseline `3198446` driver log confirms the matched
vLLM0.20 baseline path with `speculative_config=None`.

Machine-readable per-step metrics are in:

```text
docs/qwen3_30ba3b_fullgrpo20_early_metrics_20260606.csv
```

## Final Matched Metrics

Step 2-20 excludes the first warmup-heavy step. On those common matched steps:

| Metric | Baseline `3198446` | PARD K3 `3198447` | Speedup |
|---|---:|---:|---:|
| Total step time | `84.34s` | `81.58s` | `1.03x` |
| E2E throughput | `141.17 tok/s/GPU` | `146.21 tok/s/GPU` | `1.04x` |
| Generation time | `15.77s` | `10.82s` | `1.46x` |
| Policy training time | `33.36s` | `34.22s` | `0.97x` |
| Policy/ref logprobs time | `15.62s` | `15.29s` | `1.02x` |

The PARD K3 acceptance average across 40 emitted metric buckets was:

```text
Mean acceptance length: 3.08
Avg draft acceptance rate: 69.45%
Last emitted bucket: mean length 3.33, avg draft acceptance 77.6%
```

## Interpretation

This run answers the stability question positively for Qwen3-30B-A3B: the
worker-batch-32, mem80/bt16k shape completed 20 Full-GRPO steps and passed the
previous Step-3 CuMem OOM point.

SpecDec also clearly accelerates the generation slice: generation time drops
from `15.77s` to `10.82s`, a `1.46x` speedup. The E2E speedup remains small
because generation is only `18.7%` of the baseline Step 2-20 total step time,
and PARD adds or shifts some non-generation overhead. The non-generation part
goes from about `68.57s` in baseline to about `70.76s` with PARD K3, so the
observed total step-time speedup is `1.03x` rather than the idealized
generation-only Amdahl estimate.

There was no fatal OOM, actor-death, or traceback signature in either completed
driver log.

## GBS2048 Follow-Up

On 2026-06-07 PDT we also checked the larger GBS2048 shape because the original
performance recipe uses a larger rollout batch than the completed GBS512
stability pair.

The failed GBS2048 baseline `3207093` is important: it was not a SpecDec run.
The driver config confirms `policy.draft.enabled=false` and
`speculative_config=None`. It still failed around Step 3 in
`VllmGenerationWorker.wake_up()` while remapping the vLLM CuMem pool:

```text
gpu_memory_utilization=0.90
max_num_batched_tokens=32768
max_num_seqs=32
GPU KV cache size: 1,152,896 tokens
CuMemAllocator sleep freed: 162.45 GiB
Sleep mode still in use before failure: about 15.82 GiB
failure: wake_up(tags=["kv_cache"]) -> CUDA OOM at cumem_allocator.cpp:139
```

The matched PARD K3 run `3207094` failed on the same Step-3 sleep/wake path.
PARD adds only about `1.1 GiB` to the model load footprint in this setup, so
the current evidence points to the GBS2048 vLLM reservation envelope, not the
small drafter, as the primary OOM cause.

Direct GBS2048 retries were submitted with the same workload but a smaller
vLLM reservation. The first pair, `3207472` and `3207473`, was cancelled because
it was submitted from `/home/sna`; with `--no-container-mount-home`, the Ray
startup marker path was not visible inside the container. The Lustre-resubmitted
validation jobs are:

| Job | Mode | K | Status at submission | Reservation envelope |
|---:|---|---:|---|---|
| `3207492` | baseline | 0 | running, passed Step 3 and reached Step 17 as of 2026-06-07 15:46 PDT | `gpu_memory_utilization=0.80`, `max_num_batched_tokens=16384`, `max_num_seqs=32` |
| `3207493` | public PARD | 3 | failed during policy init | `gpu_memory_utilization=0.80`, `max_num_batched_tokens=16384`, `max_num_seqs=32` |
| `3207737` | public PARD | 3 | failed before model setup due actor-venv `uv sync` workspace parse error | `gpu_memory_utilization=0.80`, `max_num_batched_tokens=16384`, `max_num_seqs=32` |
| `3207940` | public PARD | 3 | submitted after submodule fix | `gpu_memory_utilization=0.80`, `max_num_batched_tokens=16384`, `max_num_seqs=32` |
| `3207978` | public PARD | 3 | running after forced actor-venv rebuild; emitted Step 1 timing | `gpu_memory_utilization=0.80`, `max_num_batched_tokens=16384`, `max_num_seqs=32` |

These retries are the current validation gate for whether Qwen3-30B-A3B can run
GBS2048 Full-GRPO for 20 steps without the vLLM wake-up OOM.

`3207493` did not fail due vLLM OOM. It failed because it raced the baseline
Megatron checkpoint conversion directory: the directory existed, so import was
skipped, but `iter_0000000/run_config.yaml` was not yet present on rank 1.
Use separate `NRL_MEGATRON_CHECKPOINT_DIR` values for concurrent baseline/PARD
Qwen3-30B-A3B submissions.

The replacement PARD K3 job `3207737` was submitted with:

```text
QWEN30_MEGATRON_CHECKPOINT_DIR=.../nrl_megatron_ckpts_qwen30ba3b_gbs2048_pard_k3_unique_20260607
MAX_STEPS=20
NUM_PROMPTS=64
NUM_GENERATIONS=32
TRAIN_GLOBAL_BATCH_SIZE=2048
SPEC_NUM_TOKENS=3
```

It did not reach vLLM or GRPO timing metrics. The failure was:

```text
uv sync --directory ... returned non-zero
nemo-gym references a workspace in tool.uv.sources, but is not a workspace member
```

This is the same actor/runtime environment class of failure as the later
Qwen3-235B `3207770` retry, not a PARD performance result.

Root cause for the actor-venv failure was a missing recursive submodule
checkout. `git submodule status --recursive` showed:

```text
-92635e74... 3rdparty/Automodel-workspace/Automodel
-50af84a5... 3rdparty/Gym-workspace/Gym
```

Because `pyproject.toml` declares `nemo_gym = { workspace = true }` and lists
`3rdparty/Gym-workspace/Gym` as a workspace member, `uv sync` could not build
the actor venv when that submodule directory was empty. Running
`git submodule update --init --recursive 3rdparty/Gym-workspace/Gym
3rdparty/Automodel-workspace/Automodel` populated the missing workspaces.

After the submodule fix, PARD K3 was resubmitted as:

```text
experiments/eagle3_online/latest_qwen30ba3b_gbs2048_pard_k3_submodulefix_step20_jobs.txt
QWEN30_public_pard_k3=3207940
```

The purpose of `3207940` is to validate the same Step-20 memory envelope as the
baseline `3207492`, but with public PARD K3 enabled.

`3207940` still reused a stale partial actor venv and was cancelled. The
corrected retry is `3207978`, submitted after recursive submodule init with
`NRL_FORCE_REBUILD_VENVS=true` and
`NRL_FORCE_REBUILD_ACTOR_VENVS=true`.

Early Step 1 evidence from `3207978` is positive but not final:

| Window | Baseline `3207492` | PARD K3 `3207978` | Speedup |
|---|---:|---:|---:|
| Step 1 total step time | `316.35s` | `299.31s` | `1.06x` |
| Step 1 generation time | `57.55s` | `40.12s` | `1.43x` |
| Step 1 generation throughput | not applicable here | `1144.03 tok/s/GPU` | pending matched average |

The emitted PARD K3 vLLM buckets show mean acceptance length roughly
`2.76-3.10` and average draft acceptance roughly `58.7-69.9%`, with
per-position acceptance examples around `0.84, 0.69, 0.57`. Step 1 is still a
warmup-heavy window; the useful comparison is the Step 2+ average once
`3207978` emits more training-result blocks.

Update as of 2026-06-07 16:44 PDT: `3207978` is still running and has emitted
metrics through Step 7, with Step 8 in progress. The Step 2-7 matched window is
already past the original Step-3 OOM point:

| Window | Baseline `3207492` | PARD K3 `3207978` | Speedup |
|---|---:|---:|---:|
| Step 2 generation time | `53.91s` | `31.15s` | `1.73x` |
| Step 2 generation throughput | `948.70 tok/s/GPU` | `1641.78 tok/s/GPU` | `1.73x` |
| Step 2 total step time | `254.89s` | `239.95s` | `1.06x` |
| Step 2 E2E throughput | `200.66 tok/s/GPU` | `213.15 tok/s/GPU` | `1.06x` |
| Step 3 generation time | `52.89s` | `30.62s` | `1.73x` |
| Step 3 generation throughput | `868.43 tok/s/GPU` | `1499.83 tok/s/GPU` | `1.73x` |
| Step 3 total step time | `246.07s` | `233.46s` | `1.05x` |
| Step 3 E2E throughput | `186.66 tok/s/GPU` | `196.74 tok/s/GPU` | `1.05x` |
| Step 4 generation time | `54.22s` | `31.20s` | `1.74x` |
| Step 4 generation throughput | `912.71 tok/s/GPU` | `1586.14 tok/s/GPU` | `1.74x` |
| Step 4 total step time | `253.67s` | `231.48s` | `1.10x` |
| Step 4 E2E throughput | `195.07 tok/s/GPU` | `213.77 tok/s/GPU` | `1.10x` |
| Step 5 generation time | `53.42s` | `32.23s` | `1.66x` |
| Step 5 generation throughput | `846.50 tok/s/GPU` | `1402.86 tok/s/GPU` | `1.66x` |
| Step 5 total step time | `248.99s` | `231.53s` | `1.08x` |
| Step 5 E2E throughput | `181.60 tok/s/GPU` | `195.30 tok/s/GPU` | `1.08x` |
| Step 6 generation time | `53.43s` | `31.18s` | `1.71x` |
| Step 6 generation throughput | `874.96 tok/s/GPU` | `1499.26 tok/s/GPU` | `1.71x` |
| Step 6 total step time | `254.83s` | `230.64s` | `1.10x` |
| Step 6 E2E throughput | `183.45 tok/s/GPU` | `202.69 tok/s/GPU` | `1.10x` |
| Step 7 generation time | `54.42s` | `31.48s` | `1.73x` |
| Step 7 generation throughput | `850.44 tok/s/GPU` | `1470.07 tok/s/GPU` | `1.73x` |
| Step 7 total step time | `243.55s` | `225.13s` | `1.08x` |
| Step 7 E2E throughput | `190.01 tok/s/GPU` | `205.56 tok/s/GPU` | `1.08x` |

The current Step 2-7 average is:

| Metric | Baseline `3207492` | PARD K3 `3207978` | Speedup |
|---|---:|---:|---:|
| Total step time | `250.33s` | `232.03s` | `1.079x` |
| E2E throughput | `189.58 tok/s/GPU` | `204.54 tok/s/GPU` | `1.079x` |
| Generation time | `53.72s` | `31.31s` | `1.716x` |
| Generation throughput | `883.62 tok/s/GPU` | `1516.66 tok/s/GPU` | `1.716x` |

Update as of 2026-06-07 17:25 PDT: `3207978` emitted all 20 timing blocks and
was in Slurm `COMPLETING` at the last poll. The final warmup-excluded Step 2-20
window is:

| Metric | Baseline `3207492` | PARD K3 `3207978` | Speedup |
|---|---:|---:|---:|
| Total step time | `248.85s` | `229.80s` | `1.083x` |
| E2E throughput | `188.61 tok/s/GPU` | `204.23 tok/s/GPU` | `1.083x` |
| Generation time | `54.11s` | `31.49s` | `1.719x` |
| Generation throughput | `867.70 tok/s/GPU` | `1492.51 tok/s/GPU` | `1.720x` |

The PARD K3 acceptance average was `69.10%` across 79 emitted metric buckets.
This is consistent with the GBS512 result: PARD gives a clear generation
speedup, while E2E speedup is limited because baseline generation is only about
`21.7%` of total step time in the GBS2048 fixed-256 workload.

## GBS512 Long-OSL Decode-Heavy Follow-Up

The Qwen3-30B-A3B runs use the official latest-main base recipe:

`examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml`

To test whether PARD/SpecDec benefit becomes visible when rollout generation is
the dominant cost, a GBS512 long-output diagnostic was submitted on
2026-06-07 PDT. This keeps full GRPO enabled (`NRL_STOP_AFTER_GENERATION=false`)
and forces long decode by setting `max_new_tokens=16384` and `min_tokens=16384`.

| Job | Mode | K | Shape |
|---:|---|---:|---|
| `3207578` | baseline | 0 | 4n4g, GBS512, prompts 16 x generations 32, max steps 3 |
| `3207579` | public PARD | 3 | 4n4g, GBS512, prompts 16 x generations 32, max steps 3 |

Key overrides:

```text
policy.generation.max_new_tokens=16384
NRL_VLLM_GENERATION_MIN_TOKENS=16384
policy.max_total_sequence_length=20480
policy.generation.vllm_cfg.max_model_len=20480
max_num_batched_tokens=32768
max_num_seqs=32
gpu_memory_utilization=0.80
```

Important caveat: although the launcher attempted to set
`NRL_VLLM_GENERATION_MIN_TOKENS=16384`, the failure dump shows vLLM
`SamplingParams(... max_tokens=16384, min_tokens=0, ignore_eos=False ...)`.
So this diagnostic is best described as a `max_new_tokens=16K` decode-heavy
run, not a strict fixed-OSL-16K run where every sample must emit 16K tokens.

This is the first decode-heavy NeMo-RL test before moving to the 28K-32K output
range. For an exact 32K output target, the model length must exceed prompt
length plus output length; otherwise any non-empty prompt can violate
`max_model_len`.

Initial status: both jobs reached vLLM initialization. Baseline model load was
`56.88 GiB` with `951,584` KV-cache tokens. PARD K3 model load was `57.99 GiB`
with `433,488` KV-cache tokens. This confirms the long-OSL config and PARD
config are accepted by vLLM.

Final diagnostic status on 2026-06-07 PDT: both jobs emitted matched Step 1
timing blocks, then entered Step 2 generation and failed. Slurm reports both
jobs as `FAILED` with exit code `1:0`, so this diagnostic should be treated as a
matched Step 1 result rather than a completed 3-step run.
Completed vLLM worker progress bars showed the expected
generation-throughput signal:

| Job | Mode | Completed worker samples | Mean output tok/s | Median output tok/s |
|---:|---|---:|---:|---:|
| `3207578` | baseline | 33 | `473.95` | `490.67` |
| `3207579` | public PARD K3 | 48 | `824.62` | `852.23` |

This is still a progress-bar signal rather than a completed step metric, but it
implies about `1.74x` generation throughput uplift, and PARD reaching logprobs
before the baseline leaves generation is a positive decode-heavy signal. The
E2E result depends on the baseline generation fraction:

```text
E2E speedup = 1 / ((1 - f_gen) + f_gen / s_gen)

where:
  f_gen = baseline_generation_time / baseline_total_step_time
  s_gen = baseline_generation_time / pard_generation_time
```

With the current partial `s_gen ~= 1.74x`, the idealized E2E projection is:

| Baseline generation fraction | Projected E2E speedup |
|---:|---:|
| `50%` | `1.27x` |
| `60%` | `1.34x` |
| `70%` | `1.42x` |
| `80%` | `1.52x` |
| `90%` | `1.62x` |

The actual E2E speedup should be computed from completed Step 1+ metrics:
`baseline_total_step_time / pard_total_step_time`. The projection above does
not include any extra non-generation overhead from PARD.

Matched Step 1 result:

| Metric | Baseline `3207578` | PARD K3 `3207579` | Speedup |
|---|---:|---:|---:|
| Total step time | `981.19s` | `737.72s` | `1.330x` |
| E2E tokens/sec/GPU | `234.11` | `308.08` | `1.316x` |
| Generation time | `809.56s` | `574.42s` | `1.409x` |
| Generation tokens/sec/GPU | `283.74` | `395.67` | `1.394x` |
| Policy/ref logprobs | `84.76s` | `78.92s` | `1.074x` |
| Policy training | `77.48s` | `77.20s` | `1.004x` |
| Prepare for generation | `7.53s` | `5.94s` | `1.268x` |

This confirms the long-OSL job is decode-heavy even after logprobs/training are
included. Baseline Step 1 spends `809.56 / 981.19 = 82.5%` of total time in
generation. PARD reduces generation time by `29.0%`, and total step time by
`24.8%`.

Using the measured baseline generation fraction and measured generation speedup:

```text
f_gen = 0.825
s_gen = 1.409
ideal E2E speedup = 1 / ((1 - f_gen) + f_gen / s_gen) = 1.315x
observed E2E tokens/sec/GPU speedup = 1.316x
observed total step-time speedup = 1.330x
```

So the long-OSL result matches the expected Amdahl-style model: when generation
is about `82.5%` of the baseline step, a `1.41x` generation-time improvement
does show up as about `1.32x` E2E improvement.

The failure is the same vLLM sleep/wake memory lifecycle pattern seen in other
colocated GRPO runs, not a failure to produce the Step 1 long-OSL metric. At
Step 2 generation start, `VllmGenerationWorker.wake_up()` calls into vLLM
`CuMemAllocator.wake_up()`, which fails while remapping the KV-cache allocation:

```text
RuntimeError: CUDA Error: out of memory at /workspace/csrc/cumem_allocator.cpp:139
```

A lower-reservation retry was then attempted with:

```text
gpu_memory_utilization=0.70
max_num_batched_tokens=16384
max_num_seqs=32
NEMO_RL_PY_EXECUTABLES_SYSTEM=0
```

Those first lower-reservation jobs, `3207762` baseline and `3207763` public
PARD K3, were cancelled because they did not reach vLLM setup. Ray workers
failed during actor initialization with:

```text
ModuleNotFoundError: No module named 'ray'
```

The same long-OSL shape was resubmitted with system Python actors to bypass the
broken actor-venv path:

| Job | Mode | K | Shape | Status |
|---:|---|---:|---|---|
| `3207808` | baseline | 0 | 4n4g, GBS512, max steps 3, `max_new_tokens=16384`, `gpu_memory_utilization=0.70`, `max_num_batched_tokens=16384` | running as of 2026-06-07 15:36 PDT |
| `3207809` | public PARD | 3 | same shape, `NEMO_RL_PY_EXECUTABLES_SYSTEM=1`, public PARD K3 | running as of 2026-06-07 15:36 PDT |

Record file:

```text
experiments/eagle3_online/latest_qwen30ba3b_longosl16k_gbs512_mem70_bt16k_systempy_fullgrpo3_jobs.txt
```

For a multi-step long-OSL run, the next retry should lower the vLLM reservation
further, for example `gpu_memory_utilization=0.70` or a smaller
`max_num_batched_tokens`, while keeping the Step 1 shape available as the
decode-heavy performance datapoint.

A memory-safer retry was submitted:

| Job | Mode | K | Shape | Purpose |
|---:|---|---:|---|---|
| `3207762` | baseline | 0 | GBS512, max OSL 16K, `gpu_memory_utilization=0.70`, `max_num_batched_tokens=16384`, `max_num_seqs=32` | Check whether Step 2 vLLM wake-up survives with lower CuMem reservation |
| `3207763` | public PARD | 3 | same as baseline | Same stability check plus PARD speed signal |

The latest-main worktree used for this diagnostic includes NeMo-RL PR #2658:

```text
e94d33c88 revert: Discard weight when finish generation in the main loop (#2495) (#2658)
```

Therefore this failure is not because the old `finish_generation` weight-discard
path from #2495 is still active. It is a remaining vLLM CuMem wake-up headroom
problem under colocated training/generation.
