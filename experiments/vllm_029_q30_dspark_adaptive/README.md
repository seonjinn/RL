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
- BF16, FlashInfer TRTLLM MoE, FAP CUDA Graph with a 1,024-token maximum
  capture size. The publication comparison uses Triton target attention so all
  four arms share an adaptive-compatible backend.
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
| FlexAttention adaptive probe | FlexAttention | Adaptive max-K7 canary | Complete, but too slow for the matrix |
| Fully matched adaptive | Triton | Baseline, fixed K5, fixed K7, adaptive max-K7 | Complete |

Adaptive verification uses device-selected variable query lengths. vLLM 0.29
rejects the target FlashInfer backend because it reports
`AttentionCGSupport.UNIFORM_BATCH`; adaptive FAP graphs require
`AttentionCGSupport.ALWAYS`. On GB200 the image selects FlashAttention 4, which
also reports `UNIFORM_BATCH`; only FlashAttention 3 reports `ALWAYS` in
`vllm/v1/attention/backends/flash_attn.py:356`. The exact v0.29 source declares
FlexAttention as `ALWAYS` in
`vllm/v1/attention/backends/flex_attention.py:848` and Triton attention as
`ALWAYS` in `vllm/v1/attention/backends/triton_attn.py:100`. Both adaptive
canaries passed, but the Triton canary was substantially faster, so the fully
matched matrix uses Triton target attention. Do not compare the FlashInfer
fixed-only numbers directly with Triton or FlexAttention results.

## Fully matched Triton results

All four jobs used source commit
`9e5181d62aaf9a7ecefdfbb1c147fd5da5c80b8e`, vLLM commit
`98dff2a81d747d1dba01a47f939f48c3526d4206`, the same prompt partitions and
seeds, and the pinned workload above. Every summary reports 2,048 samples and
`tokens_ok=true`.

| Arm | Job | Output tokens | Barrier | Output tok/s | Versus baseline |
|---|---:|---:|---:|---:|---:|
| No SpecDec | 7091333 | 6,249,754 | 62.344 s | 100,245.59 | 1.000x |
| DSpark fixed K5 | 7091334 | 6,261,903 | 39.870 s | 157,058.61 | 1.567x (+56.7%) |
| DSpark fixed K7 | 7091335 | 6,262,745 | 44.650 s | 140,264.58 | 1.399x (+39.9%) |
| DSpark adaptive max-K7 | 7091336 | 6,286,968 | 39.233 s | 160,246.09 | 1.599x (+59.9%) |

Adaptive max-K7 is 1.020x (+2.03%) faster than fixed K5 and 1.142x (+14.2%)
faster than fixed K7 in the matched Triton cohort. Its aggregate acceptance
rate is 0.3709 and mean acceptance length is 3.5965. Fixed K5 records 0.4713
and 3.3566; fixed K7 records 0.3846 and 3.6925. The controller improves the
barrier by varying the verified budget, so acceptance rate alone does not rank
the arms.

The one-worker backend canaries retain separate receipts. FlexAttention job
7090546 completed with 338,187 tokens in 307.513 seconds (1,099.75 tok/s),
while Triton job 7091184 completed with 331,919 tokens in 26.079 seconds
(12,727.64 tok/s). Both report 128 samples and `tokens_ok=true`; Triton was
11.57x faster for the generation region, which motivated the backend choice.

The completed FlashInfer fixed-only diagnostic cohort is under
`matrix-20260911T1600-cacheisolated`: baseline 121,286.14 output tok/s, fixed K5
233,956.86 output tok/s (1.929x), and fixed K7 224,188.52 output tok/s (1.848x).
All three summaries report 2,048 samples and `tokens_ok=true`.

FlashInfer fixed K5 remains the highest absolute fixed-arm result at
233,956.86 tok/s. It is not an adaptive comparison: vLLM 0.29 rejects adaptive
verification with the FlashInfer target attention backend, and its matched
baseline differs from the Triton cohort. The publishable adaptive conclusion
is therefore the within-Triton 1.599x baseline speedup and 2.03% gain over
fixed K5, not a cross-backend comparison against FlashInfer fixed K5.

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
6. Render and pass the one-worker adaptive canary before using 16 workers. Pass
   `--target-attention-backend TRITON_ATTN` for the matched adaptive cohort.
7. Render the four independent jobs with `render.py`, run `sbatch --test-only`
   for every job, then submit all four without inter-arm dependencies.
