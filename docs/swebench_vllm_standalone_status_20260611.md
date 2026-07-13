# SWE-Bench vLLM Standalone SpecDec Status - 2026-06-11

## Scope

This is vLLM standalone generation benchmarking on SWE-Bench prompts. It is not
the full SWE-Bench correctness harness.

Primary summary:

- `docs/swebench_specdec_standalone_summary_20260611.csv`
- `docs/swebench_specdec_standalone_comparison_20260611.csv`

## Completed Results

Qwen3-30B-A3B completed:

| Dataset | Prompts | OSL | Batch sizes | Methods |
| --- | ---: | ---: | --- | --- |
| SWE-Bench Lite | 64 | 1024 | 8, 16 | baseline, PARD K5, suffix K32 |
| SWE-Bench Verified | 500 loaded / 496 used | 1024 | 8, 16 | baseline, PARD K5, suffix K32, PLD K5 |
| SWE-Bench full test slice | 256 | 1024 | 8, 16 | baseline, PARD K5, suffix K32, PLD K5 |
| SWE-Bench Verified long OSL | 8 | 16384 | 1, 2 | baseline, PARD K5, suffix K32, PLD K5 |

Qwen3-32B completed:

| Dataset | Prompts | OSL | Batch sizes | Methods |
| --- | ---: | ---: | --- | --- |
| SWE-Bench Verified pilot | 64 | 1024 | 8, 16 | baseline, suffix K32, PLD K5 |

## Key Numbers

| Dataset | Model | BS | PARD | Suffix | PLD |
| --- | --- | ---: | ---: | ---: | ---: |
| Lite n64 | Qwen3-30B-A3B | 8 | 1.64x | 1.72x | n/a |
| Lite n64 | Qwen3-30B-A3B | 16 | 1.23x | 1.76x | n/a |
| Verified n500 | Qwen3-30B-A3B | 8 | 1.81x | 1.73x | 1.26x |
| Verified n500 | Qwen3-30B-A3B | 16 | 1.72x | 1.83x | 1.28x |
| Full test n256 slice | Qwen3-30B-A3B | 8 | 1.80x | 1.68x | 1.23x |
| Full test n256 slice | Qwen3-30B-A3B | 16 | 1.65x | 1.50x | 1.10x |
| Verified long OSL n8 | Qwen3-30B-A3B | 1 | 3.36x | 7.23x | 3.69x |
| Verified long OSL n8 | Qwen3-30B-A3B | 2 | 3.19x | 8.28x | 2.74x |
| Verified n64 pilot | Qwen3-32B | 8 | n/a | 1.44x | 1.11x |
| Verified n64 pilot | Qwen3-32B | 16 | n/a | 1.58x | 1.06x |

## Not Completed / Missing

- SWE-Bench full test all-prompts jobs for Qwen3-30B-A3B were submitted but
  cancelled:
  - `3263826`: baseline full-test all prompts, cancelled.
  - `3263827`: PARD full-test all prompts, cancelled.
- Qwen3-235B SWE-Bench vLLM standalone results are not present in the current
  standalone SWE-Bench summary.
- Qwen3-32B PARD SWE-Bench standalone is not present in the current summary.
- SWE-Bench Lite PLD is not present in the current summary.

## Same-Prompt Comparisons

The summarized rows use the same prompt JSONL, offset, and prompt count within
each dataset slice, so the listed baseline/PARD/suffix/PLD comparisons are
apple-to-apple for those slices:

- Lite n64: `data/swebench_lite_prompts_64.jsonl`, offset 0, count 64.
- Verified n500: `data/swebench_verified_prompts_all.jsonl`, offset 0, 500
  loaded / 496 used.
- Full test n256: `data/swebench_full_test_prompts_all.jsonl`, offset 0, count
  256.
- Verified long OSL n8: `data/swebench_verified_prompts_all.jsonl`, offset 0,
  count 8.
