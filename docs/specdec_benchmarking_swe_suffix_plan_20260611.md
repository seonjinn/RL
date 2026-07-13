# SpecDec Benchmarking SWE Suffix Plan - 2026-06-11

Reference PDF: `/Users/sna/Downloads/SpecDec_Benchmarking.pdf`

Paper: "An Empirical Study of Speculative Decoding on Software Engineering Tasks", arXiv:2604.26469v3, May 4 2026.

## Paper Setup To Match

- SWE benchmark: SWE-bench Verified, 500 issues.
- Primary similar model: `Qwen/Qwen3-32B`, thinking disabled.
- Inference engine: vLLM `0.12.0`.
- Generation: greedy decoding.
- Context/generation: max context `32768`, max generation `1024` tokens per agent turn.
- Serving shape: 8 concurrent threads.
- Agent scaffold: `mini-swe-agent`.
- PLD: lookup window `N=4`, draft length `K=5`.
- Suffix Decoding: max draft length `K=32`.
- Metrics: speedup vs autoregressive baseline, mean acceptance length `tau`, per-position acceptance.

Paper SWE-bench Verified Qwen3-32B target numbers:

| Method | Speedup | Mean acceptance length |
| --- | ---: | ---: |
| PLD | `1.05x` | `2.92` |
| Suffix Decoding | `1.66x` | `3.48` |
| Eagle-3 K3 | `1.10x` | `1.87` |
| Eagle-3 K5 | `1.53x` | `2.39` |

The paper's main claim relevant to us is that model-free Suffix Decoding is strong on repository-level repair/editing because of repeated code/context patterns. It also warns that high acceptance does not translate linearly into speedup in long-context SWE-bench because verification becomes more compute-bound; the paper reports average SWE-bench requests around `15,999` tokens.

## Current Evaluation Scope

This is not a full paper reproduction yet. The current NeMo/vLLM jobs use `standalone_vllm_specdec_breakdown.py` with prepared SWE prompts, so they measure generation throughput and acceptance on SWE-style prompts, not full `mini-swe-agent` end-to-end patch/test correctness.

The first direct comparison is on `Qwen/Qwen3-30B-A3B`, because matching PARD/PARD2 baseline jobs are already running:

- Baseline/PARD2 short output: Verified all prompts, `ISL=4096`, `OSL=1024`, batch sizes `8,16`.
- Baseline/PARD2 original SWE-bench slice: Full test offset `0`, prompt count `256`, `ISL=4096`, `OSL=1024`, batch sizes `8,16`.
- Baseline/PARD2 long output: Verified offset `0`, prompt count `8`, `ISL=4096`, `OSL=16384`, batch sizes `1,2`.
- New Suffix Decoding runs use the same prompt/batch setup and `K=32`.

## Submitted Suffix Jobs

CSV: `docs/qwen30ba3b_swebench_suffix_paperlike_20260611_jobs.csv`

| Job | Dataset | Mode | Prompt count | ISL | OSL | Batch sizes | Dependency | State at submit |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `3263988` | SWE-Bench Verified smoke | suffix K32 | `1` | `4096` | `128` | `1` | none | pending priority |
| `3263989` | SWE-Bench Verified | suffix K32 | `500` | `4096` | `1024` | `8,16` | afterok `3263988` | pending dependency |
| `3263990` | SWE-Bench Full original slice | suffix K32 | `256` | `4096` | `1024` | `8,16` | afterok `3263988` | pending dependency |
| `3263991` | SWE-Bench Verified long output | suffix K32 | `8` | `4096` | `16384` | `1,2` | afterok `3263988` | pending dependency |

The smoke job intentionally gates the larger runs. If the container is missing `arctic-inference` or this vLLM build does not expose `method="suffix"`, the larger jobs will not run.

Status update: the first smoke job `3263988` failed because the vLLM container did not include `arctic-inference`. vLLM emitted:

`ImportError: Arctic Inference is required for suffix decoding. Install via pip install arctic-inference==0.1.1.`

Those dependent suffix jobs were cancelled and replaced with an arctic-enabled run. The replacement scripts install `arctic-inference==0.1.1` into `${BENCH_ROOT}/.container_cache/arctic-inference-0.1.1` via `pip --target` and prepend that path to `PYTHONPATH`, avoiding mutation of the container image.

CSV: `docs/qwen30ba3b_swebench_suffix_paperlike_20260611_arctic_jobs.csv`

| Job | Dataset | Mode | Prompt count | ISL | OSL | Batch sizes | Dependency | Latest state |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `3264022` | SWE-Bench Verified smoke | suffix K32 | `1` | `4096` | `128` | `1` | none | completed |
| `3264023` | SWE-Bench Verified | suffix K32 | `500` | `4096` | `1024` | `8,16` | afterok `3264022` | pending priority |
| `3264024` | SWE-Bench Full original slice | suffix K32 | `256` | `4096` | `1024` | `8,16` | afterok `3264022` | pending priority |
| `3264025` | SWE-Bench Verified long output | suffix K32 | `8` | `4096` | `16384` | `1,2` | afterok `3264022` | pending priority |

The replacement smoke passed the original import/config gate and completed generation. vLLM accepted `speculative_config={'method': 'suffix', 'num_speculative_tokens': 32}` and logged that async scheduling is disabled for suffix-based speculative decoding.

Smoke parser sanity CSV: `docs/qwen30ba3b_swebench_suffix_smoke_20260611_arctic.csv`

Smoke result: `bs=1`, `OSL=128`, output `30.87` tok/s/GPU, parser acceptance `55.00%`, parser mean acceptance length `2.43`.

## Submitted Qwen3-32B Pilot

CSV: `docs/qwen3_32b_swebench_suffix_paperlike_20260611_jobs.csv`

This pilot is closer to the paper's target model. It is still a standalone prompt-throughput benchmark, not a full mini-swe-agent run. It uses `TP2`, `ISL=4096`, `OSL=1024`, prompt offset `0`, prompt count `64`, batch sizes `8,16`.

| Job | Mode | Prompt count | ISL | OSL | Batch sizes | Dependency | State at submit |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `3263997` | baseline | `64` | `4096` | `1024` | `8,16` | none | submitted |
| `3263998` | suffix K32 | `64` | `4096` | `1024` | `8,16` | afterok `3263988` | submitted |

The first Qwen3-32B suffix job was tied to failed smoke `3263988`, so it was cancelled. The Qwen3-32B baseline was also resubmitted under the arctic-enabled run naming to avoid duplicate tracking.

CSV: `docs/qwen3_32b_swebench_suffix_paperlike_20260611_arctic_jobs.csv`

| Job | Mode | Prompt count | ISL | OSL | Batch sizes | Dependency | Latest state |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `3264026` | baseline | `64` | `4096` | `1024` | `8,16` | none | pending priority |
| `3264027` | suffix K32 | `64` | `4096` | `1024` | `8,16` | afterok `3264022` | pending priority |

## Submitted PLD Jobs

The paper also reports PLD with n-gram lookup window `N=4` and draft length `K=5`. Submitted matching vLLM n-gram jobs so Suffix can be compared against both PARD/PARD2 and the paper's model-free baseline.

CSV: `docs/swebench_pld_paperlike_20260611_jobs.csv`

| Job | Model | Dataset | Prompt count | ISL | OSL | Batch sizes | TP | Latest state |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| `3264041` | Qwen3-30B-A3B | SWE-Bench Verified | `500` | `4096` | `1024` | `8,16` | `1` | pending priority |
| `3264042` | Qwen3-30B-A3B | SWE-Bench Full original slice | `256` | `4096` | `1024` | `8,16` | `1` | pending priority |
| `3264043` | Qwen3-30B-A3B | SWE-Bench Verified long output | `8` | `4096` | `16384` | `1,2` | `1` | pending priority |
| `3264044` | Qwen3-32B | SWE-Bench Verified pilot | `64` | `4096` | `1024` | `8,16` | `2` | pending priority |

## Next Comparison

When results complete, parse them together with the existing baseline/PARD2 jobs:

- Verified short output: baseline `3263824`, PARD2 `3263825`, suffix `3264023`, PLD `3264041`.
- Full original slice: baseline `3263880`, PARD2 `3263881`, suffix `3264024`, PLD `3264042`.
- Verified long output: baseline `3263894`, PARD2 `3263895`, suffix `3264025`, PLD `3264043`.
- Qwen3-32B paper-like pilot: baseline `3264026`, suffix `3264027`, PLD `3264044`.

The main acceptance/performance question is whether suffix K32 approaches the paper's SWE-bench direction: roughly `1.66x` speedup and `tau` around `3.5` for Qwen3-32B. On Qwen3-30B-A3B the exact magnitude can differ, but suffix should become more competitive than PARD/PARD2 when the output is repetitive and long.
