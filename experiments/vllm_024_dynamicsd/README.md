# vLLM 0.24 DynamicSD on GB200

This experiment stages the official ARM64 vLLM 0.24 image and compares three
Qwen3-32B generation paths on Lyris or Pre-Tyche:

- `baseline`: target-only decoding.
- `static`: Eagle-3 with fixed `K=5`.
- `dynamic`: Eagle-3 with K selected from the active scheduler batch size.

All three variants use Model Runner V1 and `PIECEWISE` CUDA graphs. This keeps
the graph mode matched because DynamicSD does not support full CUDA graphs in
vLLM 0.24. Do not compare these rows directly with a full-CUDA-graph baseline.

## Software

- Image: `vllm/vllm-openai:v0.24.0-aarch64-ubuntu2404`
- vLLM release: <https://github.com/vllm-project/vllm/releases/tag/v0.24.0>
- DynamicSD documentation:
  <https://docs.vllm.ai/en/v0.24.0/features/speculative_decoding/dynamic_speculative_decoding/>
- Target: `Qwen/Qwen3-32B`
- Drafter: `RedHatAI/Qwen3-32B-speculator.eagle3`

The runner sets only `VLLM_USE_V2_MODEL_RUNNER=0`. The old `VLLM_USE_V1`
variable is intentionally not used in vLLM 0.24.

## Dynamic Schedule

The default schedule is:

```text
BS 1-16:   K=5
BS 17-32:  K=4
BS 33-64:  K=3
BS 65-128: K=1
BS 129-512: K=0
```

The global `num_speculative_tokens` is set to the largest K in the schedule.
This matters because vLLM clamps every scheduled K to that global value. Gaps
between ranges are valid and inherit the previous K; overlapping ranges are
rejected by `benchmark.py` before model initialization.

## Stage the Image

Run on a cluster login node from this checkout:

```bash
CLUSTER=lyris TEST_ONLY=true ./stage_image.sh
CLUSTER=lyris ./stage_image.sh
```

For Pre-Tyche, replace `CLUSTER=lyris` with `CLUSTER=ptyche`. Staging uses
Pyxis `--container-save`, validates `aarch64` and `vllm==0.24.0` inside the
image, then atomically installs:

```text
/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/
  vllm-openai-v0.24.0-aarch64-ubuntu2404.sqsh
```

The adjacent `.sha256` and `.metadata` files identify the exact artifact.

## Smoke Matrix

The default smoke uses ISL 1024, OSL 128, BS 1/2, temperature 0 and 1, and one
warmup plus one measured repeat:

```bash
CLUSTER=lyris TEST_ONLY=true ./submit_matrix.sh
CLUSTER=lyris ./submit_matrix.sh
```

If image staging is still running:

```bash
CLUSTER=lyris DEPENDENCY=afterok:<image-job-id> ./submit_matrix.sh
```

The full matrix uses OSL 512, BS 1/2/4/8/16/32/64, and three measured repeats:

```bash
CLUSTER=lyris SMOKE=false ./submit_matrix.sh
```

Each `result.json` records the image/runtime version, exact engine settings,
sampling settings, latency repeats, output tok/s/GPU, acceptance rate, mean
acceptance length, and per-position acceptance rate.

The fixed-length matrix validates individual DynamicSD batch-size tiers. It is
not the primary model of an RL rollout because all requests have the same OSL.

## Synchronous RL Rollout

`submit_sync_rollout.sh` models the barrier used by synchronous GRPO-style
rollouts:

1. Select `num_prompts` prompts.
2. Expand each prompt into `samples_per_prompt` requests with unique seeds.
3. Sample with temperature 1.0 and top-p 0.9 while allowing EOS.
4. Submit the entire rollout batch in one `LLM.generate()` call.
5. Start the next rollout batch only after every request in the current call
   returns.

The default smoke is `4 prompts x 2 samples x 2 rollout batches` with a
256-token cap. It uses target TP=2 and draft TP=1, matching the topology used
by the current Qwen3-32B NeMo-RL diagnosis more closely than the TP=1 fixed
batch microbenchmark:

```bash
CLUSTER=lyris TEST_ONLY=true ./submit_sync_rollout.sh
CLUSTER=lyris ./submit_sync_rollout.sh
```

Before a full run, materialize one or both pinned RL math prompt sets. The
materializer streams rows, removes duplicate prompts, excludes reference
solutions from model input, and writes source revision and SHA256 metadata:

```bash
CLUSTER=lyris TEST_ONLY=true ./stage_math_datasets.sh
CLUSTER=lyris ./stage_math_datasets.sh
```

The pinned sources are:

- `BytedTsinghua-SIA/DAPO-Math-17k` at
  `65877096c24ffa7abc4e4fa5edb95cf3413a5674`.
- `nvidia/OpenMathInstruct-2` at
  `469216e3f46f4dacf476b382e192485ea51a143e`, split `train_1M`.

The full setup is `16 prompts x 16 samples x 3 rollout batches`, a 4096-token
cap, and an engine concurrency cap of 64. `SMOKE=false` requires a materialized
JSONL so built-in prompts cannot accidentally become a reported result:

```bash
CLUSTER=lyris \
PROMPT_JSONL=/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm024-dynamicsd/datasets/<file>.jsonl \
SMOKE=false \
./submit_sync_rollout.sh
```

Pass `PROMPT_JSONL=/lustre/.../prompts.jsonl` to replace the built-in math
smoke prompts with the same prompt set used by an RL recipe. The JSONL reader
accepts `messages`, `prompt`, `question`, `problem`, or `input` fields.

Each result records rollout time, actual output tok/s/GPU, requests/s,
completion-length p50/p90/p99/max, finish reasons, output-token hashes, and
SpecDec acceptance metrics. After all three variants finish:

```bash
python3 summarize_sync_rollout.py <matrix-root>
```

The generated `summary.csv` and `summary.json` report throughput speedup and
rollout-time reduction versus both target-only baseline and fixed-K Eagle-3.
The target-only row answers whether SpecDec helps end to end; the dynamic vs
static row isolates the value of adaptive K.

This is an offline generation makespan benchmark, not NeMo-RL E2E step time.
It includes prompt processing, sampled generation, chosen-token logprobs, and
the synchronous generation barrier. It excludes Ray dispatch, reward model or
verifier work, policy training, and weight synchronization. A 256-request
global batch is queued through one engine capped at 64 active sequences; use a
multi-replica NeMo-RL run to validate the final rollout-step reduction.

### Qwen SWE 32K/64K Sync-RL Matrix

`submit_swe_sync_rollout_matrix.sh` expands the long-tail Sync-RL request-plan
profiles from `model_method_matrix.json` and emits supported, integration-only,
and unsupported cells without guessing missing compatibility:

| Model | 32K profile | 64K profile | Supported methods | Integration only | Unsupported |
|---|---|---|---|---|---|
| Qwen3-30B-A3B | native, OSL 32768 | YaRN-4, OSL 65536 | baseline, Eagle-3 static, DynamicSD | PARD | PARD-2, DFlash, DFlare |
| Qwen3-32B | native, OSL 32768 | YaRN-4, OSL 65536 | baseline, Eagle-3 static, DynamicSD | PARD | PARD-2, DFlash, DFlare |
| Qwen3-235B-A22B | native, OSL 32768 | YaRN-4, OSL 65536 | baseline, Eagle-3 static, DynamicSD | PARD | PARD-2, DFlash, DFlare |

Run only the local manifest and scheduler checks here; remote submission,
monitoring, and artifact pull are handled elsewhere:

```bash
CLUSTER=lyris TEST_ONLY=true ./submit_swe_sync_rollout_matrix.sh
```

Current completed local evidence for this area remains the earlier Qwen3-32B
Math Sync-RL DynamicSD summaries in `report/results/dapo_sync_full/summary.csv`
and `report/results/openmath_sync_full/summary.csv`, both sampled at
temperature 1.0 and top-p 0.9. Do not reuse those values as SWE 32K/64K
placeholders; every SWE row stays pending until its own `result.json` lands.

### SPEED-Bench Official vs Overlay

`stage_speedbench.sh` pins both upstream inputs:

- SPEED-Bench dataset revision:
  `487aa718444e816458d1a0a52bfce7a454285cf4`
- NVIDIA Model Optimizer revision:
  `43fee0cd70fa9e5f85782d52a4bd8ad9c8b88446`

The two SPEED-Bench cohorts are intentionally separate:

| Cohort | Protocol | Sampling | Comparison rule |
|---|---|---|---|
| Official SPEED-Bench | instrumented upstream ModelOpt `run.py` | keep the official resolved config; no overlay default override | compare only to baselines with matching official provenance |
| Sync-RL overlay | manifest-backed prepared parquet plus barriered AsyncLLM overlay | default to NeMo-RL-matched `temperature=1.0`, `top_p=1.0` | compare only within the overlay cohort and exact matched provenance |

`summarize_speedbench_sync_rollout.py` rejects official/overlay baseline
matching across cohorts. No completed official or overlay SPEED-Bench
`result.json` artifacts are stored in this checkout yet, so Task 6
documentation stays at launch-support status only.

### NeMo-RL Performance Recipe Shapes

`submit_nemorl_perfcfg_sync_matrix.sh` maps the synchronous NeMo-RL
performance recipes to one representative vLLM generation replica:

| Model | Source recipe | Global rollout | Generation replicas | Per-engine rollout | Target TP | Max sequence |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B | `grpo-qwen3-30ba3b-4n4g.yaml` | 64 x 32 | 16 | 4 x 32 | 1 | 4096 |
| Qwen3-32B | `grpo-qwen3-32b-4n4g.yaml` | 64 x 32 | 8 | 8 x 32 | 2 | 4096 |
| Qwen3-235B-A22B | `grpo-qwen3-235b-32n4g.yaml` | 16 x 32 | 16 | 1 x 32 | 8 | 8192 |

The launcher uses OpenMathInstruct-2 prompts, temperature 1.0, top-p 1.0,
recipe GPU-memory utilization, and the recipe Triton MoE backend where
specified. Qwen3-235B uses two GB200 nodes with `--segment=2` and a Ray-backed
TP8 engine. The other models use one representative generation engine.

```bash
CLUSTER=lyris TEST_ONLY=true ./submit_nemorl_perfcfg_sync_matrix.sh
CLUSTER=lyris ./submit_nemorl_perfcfg_sync_matrix.sh

CLUSTER=lyris SMOKE=false TEST_ONLY=true \
  ./submit_nemorl_perfcfg_sync_matrix.sh
CLUSTER=lyris SMOKE=false ./submit_nemorl_perfcfg_sync_matrix.sh
```

Smoke runs preserve recipe concurrency but cap output at 256 tokens for
runtime validation. Full runs use the recipe output cap and three rollout
batches. The recipe does not set `max_num_batched_tokens`, so the launcher
leaves it at the vLLM 0.24 default and records that choice in each result.

DynamicSD requires Model Runner V1 and PIECEWISE CUDA graphs in vLLM 0.24.
Baseline, static K5, and DynamicSD all use PIECEWISE graphs here for a matched
comparison. This differs from an untouched NeMo-RL runtime and must remain a
separate setup label in reports.

Qwen3-235B TP8 spans two four-GPU GB200 nodes. The official image does not
ship Ray, so stage the pinned relocatable Ray site once before that run:

```bash
CLUSTER=lyris TEST_ONLY=true ./stage_ray_site.sh
CLUSTER=lyris ./stage_ray_site.sh
```

### June 19 Standalone Replay

`submit_legacy_0619_replay_matrix.sh` reproduces the contract used by the
June 19 standalone report: Math500 and SWE-verified prompts, ISL 4096, OSL
32768, batch sizes 1/2/4/8/16/32, temperatures 0/1, model-specific TP, four
allocated GPUs, Triton attention, and Triton MoE. Qwen3-235B also keeps the
original FP8 KV cache setting. The historical replay corpus used eager mode;
new CUDA Graph cohorts use matched PIECEWISE graphs for every compared method.

The unmodified vLLM 0.24 image supports baseline and Eagle-3, so this replay
compares baseline, static Eagle-3 K3, and DynamicSD.

```bash
CLUSTER=lyris TEST_ONLY=true ./submit_legacy_0619_replay_matrix.sh
CLUSTER=lyris ./submit_legacy_0619_replay_matrix.sh

CLUSTER=lyris SMOKE=false TEST_ONLY=true \
  ./submit_legacy_0619_replay_matrix.sh
CLUSTER=lyris SMOKE=false ./submit_legacy_0619_replay_matrix.sh
```

Use this explicit profile for the CUDA Graph cohort:

```bash
CLUSTER=lyris ENFORCE_EAGER=false CUDAGRAPH_MODE=PIECEWISE \
  SMOKE=false ./submit_legacy_0619_replay_matrix.sh
```

At temperature 1, exact token hashes can differ because of numerical and batch
invariance effects. `summarize_sync_rollout.py` marks direct time comparison as
valid only when total generated-token counts agree within 1%; otherwise use
throughput and repeated-run confidence intervals rather than raw makespan.

### Qwen3-8B Extended Methods

The extended matrix adds methods that have an exact Qwen3-8B checkpoint or do
not require one. It preserves the June 19 ISL 4096, OSL 32768, BS
1/2/4/8/16/32, Math/SWE, and temperature 0/1 contract:

| Method | Runtime path | K | Checkpoint |
|---|---|---:|---|
| Suffix | vLLM 0.24 native | 32 | none |
| PARD | vLLM 0.24 `draft_model` parallel drafting | 12 | `amd/PARD-Qwen3-0.6B` |
| PARD-2 | minimal vLLM 0.24 target-feature overlay | 15 | `amd/PARD2-Qwen3-8B` |
| DFlash | vLLM 0.24 native | 15 | `z-lab/Qwen3-8B-DFlash-b16` |

Stage the pinned repositories, checkpoints, Arctic Inference dependency, and
PARD-2 overlay before submission:

```bash
CLUSTER=lyris TEST_ONLY=true ./stage_extended_method_assets.sh
CLUSTER=lyris ./stage_extended_method_assets.sh

CLUSTER=lyris TEST_ONLY=true ./submit_qwen8_extended_methods_matrix.sh
CLUSTER=lyris ./submit_qwen8_extended_methods_matrix.sh
```

The default is an OSL 256, BS4 smoke. Set `SMOKE=false` for the full legacy
shape. Baseline and every method use the same target model, prompt files,
sampling parameters, CUDA Graph mode, attention backend, and throughput GPU
denominator.

The existing Native 32K corpus was measured with eager mode and
`CUDAGRAPH_MODE=NONE`. Use the following matched PIECEWISE profile to collect
CUDA Graph enabled rows without mixing the cohorts:

```bash
CLUSTER=lyris ENFORCE_EAGER=false CUDAGRAPH_MODE=PIECEWISE \
  TEST_ONLY=true ./submit_qwen8_extended_methods_matrix.sh
CLUSTER=lyris ENFORCE_EAGER=false CUDAGRAPH_MODE=PIECEWISE \
  ./submit_qwen8_extended_methods_matrix.sh

CLUSTER=lyris ENFORCE_EAGER=false CUDAGRAPH_MODE=PIECEWISE \
  SMOKE=false ./submit_qwen8_extended_methods_matrix.sh
```

DynamicSD in vLLM 0.24 requires PIECEWISE rather than full CUDA graphs, so use
PIECEWISE when one report must compare baseline, fixed SpecDec, and DynamicSD.

DFlare is not a vLLM 0.24 method. `submit_angelslim_matrix.sh` therefore runs
the pinned AngelSlim reference benchmark and keeps those Transformer-native
results in a separate result tree. Each DFlash/DFlare job runs its own matched
autoregressive baseline and records decode throughput speedup, acceptance
rate, mean acceptance length, and the full acceptance-length histogram:

```bash
CLUSTER=lyris TEST_ONLY=true ./submit_angelslim_matrix.sh
CLUSTER=lyris ./submit_angelslim_matrix.sh
```

### Long-Context Profiles

`submit_qwen8_long_context_matrix.sh` extends the Qwen3-8B matrix with two
YaRN factor-4 profiles. The 128K label is the supported total sequence length,
not a 128K output added after the 4K input:

| Profile | ISL | OSL | Total sequence | Initial batch size |
|---|---:|---:|---:|---:|
| 64K | 4,096 | 65,536 | 69,632 | 1 |
| total 128K | 4,096 | 126,976 | 131,072 | 1 |

The wrapper creates symlink-backed model views under
`${LUSTRE_ROOT}/vllm024-dynamicsd/long-context-models/yarn4`. Only
`config.json` and provenance metadata are new files; weights remain in the
pinned Hugging Face snapshots. Target and draft checkpoints all receive the
same YaRN parameters so baseline and SpecDec use matched position encoding.

Run scheduling validation before submission:

```bash
CLUSTER=lyris TEST_ONLY=true \
  ./submit_qwen8_long_context_matrix.sh

CLUSTER=lyris \
  ./submit_qwen8_long_context_matrix.sh
```

Each profile starts with BS1, zero benchmark-level warmup repeats, and one
measured exact-length generation. To add BS2 after the initial jobs establish
wall-time and KV-cache headroom, set `BATCH_SIZES_64K=2` or
`BATCH_SIZES_128K=2`. Every batch size is submitted as a separate job group so
a timeout cannot discard another batch-size result.

The AngelSlim DFlare reference runner is excluded from the first 64K/128K
launch. It executes an autoregressive baseline and SpecDec serially; the
measured decode rate cannot complete that pair within Lyris's five-hour wall
limit. DFlash remains covered by the native vLLM 0.24 matrix.

For long-context DFlare itself, the staged AngelSlim runner supports
`RUN_MODE=both|baseline|spec`. The long-context wrapper submits only the DFlare
path so 64K and total-128K measurements run in parallel instead of waiting for
an autoregressive Transformers baseline in the same job:

```bash
CLUSTER=lyris TEST_ONLY=true \
  ./submit_angelslim_long_context_dflare.sh

CLUSTER=lyris \
  ./submit_angelslim_long_context_dflare.sh
```

This creates eight jobs: Math/SWE, temperature 0/1, and the 64K/total-128K
profiles. The results provide AngelSlim-native DFlare throughput, acceptance,
and mean accepted length. Do not label a ratio against the vLLM baseline as an
apples-to-apples speedup because the runtime and attention backend differ. The
32K `RUN_MODE=both` jobs remain the matched AngelSlim baseline comparison.

## NSys Profiling

Use NSys after the smoke confirms that the image has an `nsys` binary:

```bash
CLUSTER=lyris TEST_ONLY=true ./submit_nsys.sh
CLUSTER=lyris PROFILE_BATCH_SIZE=16 ./submit_nsys.sh
```

The profiler runs two warmups, then captures exactly one measured generation
range with `cudaProfilerApi`. Profile baseline, static, and dynamic in separate
processes so model initialization and CUDA graph capture are outside the trace.

The first analysis should split GPU and CPU time into:

1. Target verification forward pass.
2. Eagle-3 draft forward pass and draft LM head.
3. Rejection sampling and recovery-token softmax.
4. CUDA graph replay and launch gaps.
5. NCCL collectives when repeating the profile with TP greater than one.

Use the `llm-analyzer` workflow on each `.nsys-rep` after checking the NVTX
tree. The benchmark emits one outer range named
`vllm024.<variant>.bs<batch>.repeat0`; deeper draft/verify attribution requires
the NVTX ranges exported by vLLM itself or a small source instrumentation patch.

If the official image does not contain `nsys`, `submit_nsys.sh` exits with a
clear `nsys_unavailable.txt` artifact. Do not install packages into the staged
benchmark image in place; create and checksum a derived profiling image so the
throughput and profiling environments remain identifiable.

## Interpretation

NSys is the correct next tool when acceptance length is healthy but throughput
is unexpectedly below baseline. Acceptance metrics alone cannot distinguish a
slow drafter, verifier shape inflation, rejection-sampler cost, graph fallback,
or TP communication. Use the wall-clock matrix first, then profile one
representative batch size where the regression is reproducible.
