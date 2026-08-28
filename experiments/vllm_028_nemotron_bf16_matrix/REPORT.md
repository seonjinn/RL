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

- Harness commit: `76656d1f1159ccdd44b2290e74e85b755f2421ef`
- Container staging job: `2815382`
- Super BF16 staging job: `2815383`
- Ray: 2.48.0, exact hash-locked ARM64 sidecar
- Result root: `/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-benchmark-results/vllm028-nemotron-bf16-smoke`
