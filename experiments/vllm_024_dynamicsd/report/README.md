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

## NeMo-RL Performance-Recipe Smoke

These rows model one representative generation replica from each NeMo-RL
performance recipe. They preserve recipe request concurrency and topology but
cap output at 256 tokens, so they validate runtime behavior rather than predict
the final long-generation speedup.

| Model | Variant | Rollout time (s) | tok/s/GPU | Speedup | Acceptance | Mean accepted length |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B | Baseline | 3.282 | 9,983.65 | 1.000x | n/a | n/a |
| Qwen3-30B-A3B | Static Eagle-3 K5 | 4.724 | 6,936.46 | 0.695x | 56.5% | 3.83 |
| Qwen3-30B-A3B | DynamicSD | 3.757 | 8,720.80 | 0.873x | 85.8% | 1.87 |
| Qwen3-32B | Baseline | 3.918 | 8,362.59 | 1.000x | n/a | n/a |
| Qwen3-32B | Static Eagle-3 K5 | 6.696 | 4,893.83 | 0.585x | 34.2% | 2.71 |
| Qwen3-32B | DynamicSD | 4.407 | 7,435.18 | 0.889x | 55.0% | 3.75 |
| Qwen3-235B-A22B | Baseline | 5.646 | 181.38 | 1.000x | n/a | n/a |
| Qwen3-235B-A22B | Static Eagle-3 K5 | 2.983 | 343.29 | 1.893x | 39.4% | 2.97 |
| Qwen3-235B-A22B | DynamicSD fixed K5 diagnostic | 4.447 | 230.25 | 1.269x | 40.1% | 3.01 |

The default multi-range DynamicSD schedule hung after CUDA graph capture for
Qwen3-235B TP8, while an otherwise identical single-range K5 schedule completed.
This rules out Ray, TP8, and PIECEWISE CUDA graphs as the general cause. A fixed
K4 diagnostic is queued to distinguish K4 execution from multi-range schedule
selection. The full recipe-length jobs are `2270333-2270338` and
`2270341-2270342`; Qwen3-235B DynamicSD is intentionally excluded until that
diagnostic is resolved.

## June 19 Replay Smoke

This is a short OSL=256 validation of the June 19 report contract: ISL=4096,
BS=4, eager execution, Math500 or SWE-verified prompts, temperature 0 or 1,
and the original model-specific TP and four-GPU throughput denominator. Static
uses Eagle-3 K3. DynamicSD selects K5 at BS=4. These speedups are not the final
OSL=32768 results.

| Domain | Model | Temp | Static speedup (acceptance) | Dynamic speedup (acceptance) |
|---|---|---:|---:|---:|
| Math | Qwen3-30B-A3B | 0 | 2.74x (94.9%) | 3.59x (91.4%) |
| Math | Qwen3-30B-A3B | 1 | 2.88x (90.9%) | 3.66x (90.5%) |
| Math | Qwen3-32B | 0 | 3.00x (88.6%) | 3.70x (83.7%) |
| Math | Qwen3-32B | 1 | 2.26x (77.1%) | 2.92x (70.0%) |
| Math | Qwen3-235B-A22B | 0 | 2.66x (78.0%) | 2.76x (51.7%) |
| Math | Qwen3-235B-A22B | 1 | 2.64x (78.0%) | 2.85x (51.7%) |
| SWE | Qwen3-30B-A3B | 0 | 2.36x (74.7%) | 2.74x (65.4%) |
| SWE | Qwen3-30B-A3B | 1 | 2.42x (71.5%) | 2.92x (65.4%) |
| SWE | Qwen3-32B | 0 | 2.55x (67.1%) | 2.79x (54.2%) |
| SWE | Qwen3-32B | 1 | 2.53x (73.0%) | 2.64x (56.3%) |
| SWE | Qwen3-235B-A22B | 0 | 2.38x (55.5%) | 2.14x (32.7%) |
| SWE | Qwen3-235B-A22B | 1 | 2.26x (56.3%) | 2.23x (33.4%) |

The complete OSL=32768 matrix contains 216 jobs across both domains, three
models, temperatures 0/1, batch sizes 1/2/4/8/16/32, and the three variants.
Jobs `2270108-2270323` were submitted on July 3, 2026. Three jobs hit a Lyris
`spank_sybil` credential retrieval error before the container started and were
resubmitted as `2270324-2270326`; no benchmark-code failure was observed during
the initial five-minute monitoring window.

## Qwen3-8B Long-Context Extension

The long-context follow-up keeps native OSL=32768 results separate from YaRN
context-extension results. Its two new matched profiles are:

| Profile | ISL | OSL | Total | Position encoding | Initial scope |
|---|---:|---:|---:|---|---|
| 64K | 4,096 | 65,536 | 69,632 | YaRN factor 4 | BS1 |
| total 128K | 4,096 | 126,976 | 131,072 | YaRN factor 4 | BS1 |

Both profiles cover Math/SWE, temperature 0/1, baseline, Suffix K32, PARD
K12, PARD-2 K15, and DFlash K15. A speedup is reportable only after the exact
domain, temperature, context profile, and batch-size baseline row completes.
The target and every drafter use symlink-backed views of the pinned snapshots
with identical YaRN parameters. AngelSlim-native DFlare is tracked separately
because its serial baseline-plus-SpecDec runner cannot fit these lengths in the
five-hour Lyris wall limit.

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
