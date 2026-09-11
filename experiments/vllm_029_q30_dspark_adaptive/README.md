# vLLM 0.29 Q30 DSpark adaptive-verification study

This standalone experiment estimates the generation-only headroom for the
Qwen3-30B-A3B NeMo-RL GBS-2048 performance shape without modifying the stable
vLLM 0.25.1 integration.

## Matched workload

- Target: Qwen3-30B-A3B Base.
- Drafter: PTV3-SWE + replay + SWA DSpark step 44000.
- 64 prompts x 32 generations = 2,048 samples at one synchronous barrier.
- 16 independent TP1 engines, 128 samples per engine, on 4 x 4 GB200 GPUs.
- Natural EOS with a 4,096-output-token cap, temperature 1.0, top-p 1.0.
- `max_num_seqs=128`, `max_num_batched_tokens=32768`.
- BF16, FlexAttention target attention, FlashInfer TRTLLM MoE, FAP CUDA Graph
  with a 1,024-token maximum capture size.
- Identical prompt partitions and per-request seeds in every arm.

## Arms

| Arm | Meaning |
|---|---|
| `baseline` | No speculative decoding |
| `dspark_k5` | Fixed K5, probabilistic drafting |
| `dspark_k7` | Fixed K7, probabilistic drafting |
| `dspark_adaptive_k7` | DSpark confidence-scheduled adaptive verification, maximum K7 |

The adaptive arm is not the batch-size `num_speculative_tokens_per_batch_size`
DynamicSD controller. It is vLLM 0.29 DSpark adaptive verification. All DSpark
arms load the trained confidence head so fixed/adaptive startup and memory are
matched; only adaptive budget selection differs.

## Attention-backend cohorts

Keep the following cohorts separate:

| Cohort | Target attention | Valid arms | Status |
|---|---|---|---|
| FlashInfer fixed-only | FlashInfer | Baseline, fixed K5, fixed K7 | Complete diagnostic cohort |
| FA4 adaptive probe | FlashAttention 4 | None | Runtime-blocked on GB200 |
| Fully matched adaptive | FlexAttention | Baseline, fixed K5, fixed K7, adaptive max-K7 | Requires adaptive canary before matrix submission |

Adaptive verification uses device-selected variable query lengths. vLLM 0.29
rejects the target FlashInfer backend because it reports
`AttentionCGSupport.UNIFORM_BATCH`; adaptive FAP graphs require
`AttentionCGSupport.ALWAYS`. On GB200 the image selects FlashAttention 4, which
also reports `UNIFORM_BATCH`; only FlashAttention 3 reports `ALWAYS` in
`vllm/v1/attention/backends/flash_attn.py:356`. The exact v0.29 source declares
FlexAttention as `ALWAYS` in
`vllm/v1/attention/backends/flex_attention.py:848`, so the next validated
candidate is a fully matched FlexAttention cohort. Do not compare the
FlashInfer fixed-only numbers directly with a FlexAttention result.

The completed FlashInfer fixed-only diagnostic cohort is under
`matrix-20260911T1600-cacheisolated`: baseline 121,286.14 output tok/s, fixed K5
233,956.86 output tok/s (1.929x), and fixed K7 224,188.52 output tok/s (1.848x).
All three summaries report 2,048 samples and `tokens_ok=true`.

Each worker JSON validates its 128 completions. The barrier aggregator requires
all 16 workers, recomputes output tokens from per-request rows, compares that
against the independent length summary, and publishes `tokens_ok=true` only
after both accounts and the 2,048-sample count agree.

## Reproducibility sequence

1. Commit and push this experiment branch.
2. Pull the exact commit into a `/home` repository on OCI-HSG.
3. Submit `stage_vllm029_container.sbatch` with the official
   `registry-1.docker.io/vllm/vllm-openai:v0.29.0-ubuntu2404` image and vLLM commit
   `98dff2a81d747d1dba01a47f939f48c3526d4206`.
4. Verify the immutable sqsh, metadata, SHA256, and stable symlink.
5. Run `smoke_vllm029_container.sbatch`; it verifies the GPU, package versions,
   and `/usr/local/cuda-13.0/bin/ptxas` before writing its JSON receipt.
6. Render and pass the one-worker adaptive canary before using 16 workers.
7. Render the four independent jobs with `render.py`, run `sbatch --test-only`
   for every job, then submit all four without inter-arm dependencies.
