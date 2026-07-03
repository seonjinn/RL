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

At temperature 1, exact token hashes can differ because of numerical and batch
invariance effects. `summarize_sync_rollout.py` marks direct time comparison as
valid only when total generated-token counts agree within 1%; otherwise use
throughput and repeated-run confidence intervals rather than raw makespan.

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
