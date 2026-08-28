# vLLM 0.28 Nemotron-3 BF16 DynamicMTP report

## Canary configuration

| Field | Super | Ultra |
| --- | --- | --- |
| Checkpoint revision | `d51eab0d1f979ebc26b546e634a04f450d99158e` | `624ba927cfbef0427354998700de3d51173c8c04` |
| Weight / KV precision | BF16 / FP8 | BF16 / FP8 |
| Parallelism | TP2, DP1, 1 node | TP8, DP1, EP, Ray, 2 nodes |
| Workload | ISL 10000, OSL 1000, BS 1 | ISL 10000, OSL 1000, BS 1 |

The runtime was vLLM 0.28.0 at commit `2cf0a69`, using the official ARM64
image digest
`sha256:41b54fb42c66a670a8b27e613ebef05898f24b9ab1bdab28bd00c877bd4935f4`.
Both runners used exact OSL, ignored EOS, disabled prefix caching, enabled
chunked prefill, and set `max_num_seqs=512` and
`max_num_batched_tokens=32768`.

DynamicMTP used max-K 5 and the schedule
`1:4:5,5:16:3,17:64:2,65:128:1,129:512:0`.

## CUDA Graph verification

The completed BS512 baseline and DynamicMTP logs verify real CUDA Graph
capture rather than eager execution. All four runs report
`enforce_eager=False`, `CUDAGraphMode.PIECEWISE`, one graph warmup, 83 capture
sizes through 1024, and successful `83/83` mixed prefill/decode capture.

| Model | Method | Job | Capture | CUDA Graph pool memory |
| --- | --- | ---: | --- | ---: |
| Super | Baseline | `2816828` | `83/83` | 1.44 GiB |
| Super | DynamicMTP | `2816642` | `83/83` | 1.49 GiB |
| Ultra | Baseline | `2816829` | `83/83` | 2.39 GiB |
| Ultra | DynamicMTP | `2816643` | `83/83` | 2.45 GiB |

Super TP2 runs systematically emitted non-fatal `CUDACachingAllocator`
allocation warnings during pre-capture memory profiling. Every audited Super
baseline, DynamicMTP, and Static K5 job then completed `83/83` graph capture,
generation, exact token validation, and canonical publication. Their throughput
is retained with this caveat; representative repeats are required before the
matrix is marked final. The corresponding Ultra TP8 jobs did not emit these
allocator warnings.

## MRV1 correctness canary

| Model | Method | tok/s/GPU | Speedup | Acceptance | Mean accepted length | Generation latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Super | Baseline | 38.9932 | 1.0000x | - | - | 12.8227 s |
| Super | DynamicMTP | 99.9251 | 2.5626x | 61.38% | 4.0691 | 5.0037 s |
| Ultra | Baseline | 7.2625 | 1.0000x | - | - | 17.2116 s |
| Ultra | DynamicMTP | 20.6467 | 2.8429x | 59.52% | 3.9762 | 6.0542 s |

All four jobs completed exactly 1000 output tokens and atomically published a
validated canonical `result.json`. These are single-concurrency canary results,
not the final matrix averages.

| Model | Baseline job | DynamicMTP job |
| --- | ---: | ---: |
| Super | `2815633` | `2815634` |
| Ultra | `2815504` | `2815505` |

The DynamicMTP logs for both BF16 checkpoints report `Detected MTP model` and
share the target embedding and LM-head weights with the built-in drafter. No
external draft checkpoint was used.

## MRV1 10K/1K gate expansion

### Super BF16

| BS | Method | tok/s/GPU | Speedup | Acceptance | Mean accepted length | Job |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Baseline | 38.9932 | 1.0000x | - | - | `2815633` |
| 1 | Static K5 | 102.7185 | 2.6343x | 61.38% | 4.0691 | `2815986` |
| 1 | DynamicMTP | 99.9251 | 2.5626x | 61.38% | 4.0691 | `2815634` |
| 2 | Baseline | 77.1876 | 1.0000x | - | - | `2817283` |
| 2 | Static K5 | 158.7226 | 2.0563x | 49.02% | 3.4509 | `2817361` |
| 2 | DynamicMTP | 143.3739 | 1.8575x | 39.12% | 2.9558 | `2817288` |
| 4 | Baseline | 154.4845 | 1.0000x | - | - | `2817284` |
| 4 | Static K5 | 283.1409 | 1.8328x | 44.20% | 3.2101 | `2817362` |
| 4 | DynamicMTP | 330.7121 | 2.1407x | 64.89% | 4.2444 | `2817289` |
| 8 | Baseline | 287.1353 | 1.0000x | - | - | `2817286` |
| 8 | Static K5 | 534.1882 | 1.8604x | 55.08% | 3.7539 | `2817363` |
| 8 | DynamicMTP | 470.0615 | 1.6371x | 57.08% | 2.7484 | `2817290` |
| 16 | Baseline | 511.5928 | 1.0000x | - | - | `2817282` |
| 16 | Static K5 | 765.7723 | 1.4968x | 54.23% | 3.7117 | `2817360` |
| 16 | DynamicMTP | 794.3084 | 1.5526x | 56.96% | 2.7127 | `2817287` |
| 32 | Baseline | 824.4557 | 1.0000x | - | - | `2815983` |
| 32 | Static K5 | 1155.8046 | 1.4019x | 51.84% | 3.5919 | `2815988` |
| 32 | DynamicMTP | 1124.0057 | 1.3633x | 76.39% | 2.5622 | `2815985` |
| 128 | Baseline | 1562.5967 | 1.0000x | - | - | `2815982` |
| 128 | Static K5 | 1563.6908 | 1.0007x | 51.93% | 3.5967 | `2815987` |
| 128 | DynamicMTP | 1401.2702 | 0.8968x | 75.56% | 2.5345 | `2815984` |

### Ultra BF16

| BS | Method | tok/s/GPU | Speedup | Acceptance | Mean accepted length | Job |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Baseline | 7.2625 | 1.0000x | - | - | `2815504` |
| 1 | Static K5 | 16.7987 | 2.3131x | 47.34% | 3.3670 | `2815993` |
| 1 | DynamicMTP | 20.6467 | 2.8429x | 59.52% | 3.9762 | `2815505` |
| 2 | Baseline | 14.3217 | 1.0000x | - | - | `2817292` |
| 2 | Static K5 | 34.6345 | 2.4183x | 53.36% | 3.6679 | `2817365` |
| 2 | DynamicMTP | 40.6689 | 2.8397x | 68.33% | 4.4163 | `2817296` |
| 4 | Baseline | 28.0637 | 1.0000x | - | - | `2817293` |
| 4 | Static K5 | 65.2620 | 2.3255x | 67.61% | 4.3803 | `2817366` |
| 4 | DynamicMTP | 69.4982 | 2.4764x | 66.07% | 4.3036 | `2817297` |
| 8 | Baseline | 52.5442 | 1.0000x | - | - | `2817294` |
| 8 | Static K5 | 101.3524 | 1.9289x | 64.64% | 4.2319 | `2817367` |
| 8 | DynamicMTP | 104.9324 | 1.9970x | 73.54% | 3.2606 | `2817298` |
| 16 | Baseline | 95.6500 | 1.0000x | - | - | `2817291` |
| 16 | Static K5 | 166.6426 | 1.7422x | 61.81% | 4.0905 | `2817364` |
| 16 | DynamicMTP | 163.2649 | 1.7069x | 76.52% | 3.3102 | `2817295` |
| 32 | Baseline | 163.7139 | 1.0000x | - | - | `2815990` |
| 32 | Static K5 | 241.9931 | 1.4781x | 60.09% | 4.0046 | `2815995` |
| 32 | DynamicMTP | 213.9473 | 1.3068x | 81.44% | 2.6588 | `2815992` |
| 128 | Baseline | 334.6108 | 1.0000x | - | - | `2815989` |
| 128 | Static K5 | 324.6044 | 0.9701x | 60.68% | 4.0342 | `2815994` |
| 128 | DynamicMTP | 289.2306 | 0.8644x | 88.15% | 2.1732 | `2815991` |

The requested-batch lookup maps BS2/4 to K5, BS8/16 to K3, BS32 to K2, and
BS128 to K1, but runtime K is selected from the actively scheduled batch. The
acceptance percentages are therefore not directly comparable to static K5
acceptance. DynamicMTP did not recover its overhead at offered BS128; the BS512
gate was used to inspect the actual scheduler behavior.

## Offered BS512 active-batch finding

| Model | Baseline tok/s/GPU | DynamicMTP tok/s/GPU | Speedup | Acceptance | Mean accepted length | Baseline job | Dynamic job |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Super | 1941.3559 | 1511.0894 | 0.7784x | 74.58% | 2.4976 | `2816828` | `2816642` |
| Ultra | 383.3092 | 271.0352 | 0.7071x | 87.87% | 1.8896 | `2816829` | `2816643` |

Both corrected-harness reruns completed exactly 512000 measured output tokens,
passed `tokens_ok=true`, exited with code zero, and atomically published a
canonical `result.json`. The matched baseline jobs also completed exactly
512000 output tokens with `tokens_ok=true`. DynamicMTP is slower than baseline
at offered BS512 for both models under this schedule.

The original diagnostic jobs (`2816110` for Super and `2816111` for Ultra)
completed generation but withheld canonical publication because the validator
had assumed offered BS512 implied K0. Logs disproved that assumption: Super
admitted about 63 active requests and selected the K2 range, while Ultra
admitted about 73 and selected the K1 range. As each offline batch drained, its
active batch and selected K continued to change.

The corrected harness records dynamic `effective_k` as unknown, records the
requested-batch schedule lookup separately as K0, and identifies
`active_scheduled_batch` as the runtime K-selection basis. A true zero-draft
control requires an explicit all-K0 schedule; ordinary DynamicSD runs must be
interpreted from their observed draft metrics.

## MRV2 compatibility canary

| Model | Baseline tok/s/GPU | Baseline job | DynamicMTP job | DynamicMTP result |
| --- | ---: | ---: | ---: | --- |
| Super | 96.9387 | `2815808` | `2815809` | Failed during full CUDA graph capture |
| Ultra | 13.7748 | `2815810` | `2815811` | Failed during full CUDA graph capture |

Both MRV2 baselines completed exact output-token validation with
`FULL_AND_PIECEWISE`. Both DynamicMTP jobs failed before generation at the same
vLLM 0.28 assertion in `mamba_attn.py`:

```text
assert m.max_query_len == 1 + self.num_spec_tokens  # decode-only
```

The failure is isolated to the combination of MTP/DynamicSD and full CUDA graph
capture. MRV1 `PIECEWISE` is therefore the validated runner for the remaining
BF16 matrix. MRV2 DynamicMTP results must not be included in performance
comparisons.

## Provenance

- Canary harness commit: `76656d1f1159ccdd44b2290e74e85b755f2421ef`
- 10K/1K gate expansion harness commit: `6d4ac2da89abf64b28f4e6baa8df06798c4747dd`
- BS512 canonical rerun harness commit: `f0dd8af3110820c708de2ce7b0720970f4c8ef8c`
- BS512 canonical jobs: Super `2816642`, Ultra `2816643`
- BS512 matched baseline jobs: Super `2816828`, Ultra `2816829`
- Container staging job: `2815382`
- Super BF16 staging job: `2815383`
- Ray: 2.48.0, exact hash-locked ARM64 sidecar
- Result root: `/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-benchmark-results/vllm028-nemotron-bf16-smoke`
