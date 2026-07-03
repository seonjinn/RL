# vLLM 0.24 DynamicSD Results

Results were collected on Lyris GB200 with Qwen3-32B target TP=2 and the
RedHatAI Qwen3-32B Eagle-3 drafter at draft TP=1. All compared variants use
vLLM 0.24.0, `enforce_eager=False`, PIECEWISE CUDA graphs, prefix caching,
chunked prefill, temperature 1.0, and top-p 0.9.

## Reproducibility

- Source commit: `c56dad30b4f285df35c5e9fb40f06ae6bbf289b1`
- Image: `vllm/vllm-openai:v0.24.0-aarch64-ubuntu2404`
- Image SHA256: `48938d31fc7e04835f8a67107f3f94141012976087cb980745eed6339637b6ae`
- DAPO revision: `65877096c24ffa7abc4e4fa5edb95cf3413a5674`
- OpenMathInstruct-2 revision: `469216e3f46f4dacf476b382e192485ea51a143e`
- Workload: 16 prompts x 16 samples x 3 synchronous rollout batches
- Per-engine concurrency cap: 64; max new tokens: 4096
- Dynamic K schedule: 1-16:5, 17-32:4, 33-64:3, 65-128:1, 129-512:0

Each rollout batch is one blocking `LLM.generate()` call. The following batch
does not start until every request in the current batch completes.

## Primary Results

| Dataset | Variant | Generation time (s) | tok/s/GPU | Throughput speedup | Time reduction | Acceptance | Mean accepted length |
|---|---|---:|---:|---:|---:|---:|---:|
| DAPO-Math-17k | Baseline | 644.87 | 2,435.55 | 1.000x | 0.00% | n/a | n/a |
| DAPO-Math-17k | Static Eagle-3 K5 | 482.53 | 3,253.18 | 1.336x | 25.17% | 31.61% | 2.58 |
| DAPO-Math-17k | DynamicSD | 427.10 | 3,674.68 | 1.509x | 33.77% | 45.65% | 2.38 |
| OpenMathInstruct-2 | Baseline | 580.97 | 2,230.60 | 1.000x | 0.00% | n/a | n/a |
| OpenMathInstruct-2 | Static Eagle-3 K5 | 416.14 | 3,126.10 | 1.401x | 28.37% | 33.05% | 2.65 |
| OpenMathInstruct-2 | DynamicSD | 374.43 | 3,463.05 | 1.553x | 35.55% | 46.59% | 2.43 |

DynamicSD improved throughput over static K5 by 12.96% on DAPO and 10.78% on
OpenMathInstruct-2. Its rollout generation time was 11.49% and 10.02% shorter,
respectively.

The generated-token ratios versus baseline were 99.93-100.38%, so all direct
time comparisons passed the 1% work-equivalence gate. Exact token hashes are
not expected to match at temperature 1.0; this benchmark does not replace a
reward or accuracy evaluation.

## Fixed-Length Tier Test

The batch-size 1, 2, 4, 8, 16, 32, and 64 matrix completed for temperature 0
and 1. At temperature 0, DynamicSD delivered 3.91x, 3.87x, 4.05x, 4.03x,
3.83x, 3.52x, and 2.55x baseline throughput, respectively.

Temperature-1 acceptance varied substantially at small sample counts and with
the repeated synthetic prompt. Treat that matrix as a scheduler-tier smoke
test, not as the primary RL performance result. The real-prompt synchronous
rollout tables above contain 768 sampled completions per variant.

## Artifacts

- [DAPO summary CSV](results/dapo_sync_full/summary.csv)
- [DAPO summary JSON](results/dapo_sync_full/summary.json)
- [OpenMathInstruct summary CSV](results/openmath_sync_full/summary.csv)
- [OpenMathInstruct summary JSON](results/openmath_sync_full/summary.json)
- [Fixed-length result JSON files](results/fixed_full/)

Cluster roots:

```text
/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm024-dynamicsd/sync-rollout/20260703_v024_dapo_sync_full
/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm024-dynamicsd/sync-rollout/20260703_v024_openmath_sync_full
/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm024-dynamicsd/runs/20260703_v024_fixed_full
```

The official throughput image does not contain NSys. Use a separately
checksummed profiling image with `submit_nsys.sh`; do not modify this staged
image in place.
