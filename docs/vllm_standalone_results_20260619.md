# vLLM Standalone SpecDec Results

Updated: 2026-06-19. Batch-size speedup matrix added.

## Key Findings

- Best Math temp 0.0: Qwen3-32B suffix_k32 at batch 1 = 10.97x.
- Best Math temp 1.0: Qwen3-32B suffix_k32 at batch 8 = 3.11x.
- Best SWE temp 0.0: Qwen3-32B suffix_k32 at batch 4 = 6.55x.
- Best SWE temp 1.0: Qwen3-30B-A3B suffix_k32 at batch 4 = 7.30x.

## Outputs

- `docs/vllm_standalone_results_20260619.html`
- `docs/vllm_standalone_batch_speedups_20260619.csv`
- `docs/vllm_standalone_batch_speedup_matrix_20260619.csv`
- `docs/vllm_standalone_results_20260619_summary.csv`
## Qwen3-235B Math note

The 6/19 Math rerun contains Qwen3-235B baseline rows only for batch 1/2. Qwen3-235B Math SpecDec rows were completed in earlier 6/13 OCI and 6/14 Lyris standalone runs and are now exported separately in `docs/vllm_standalone_qwen235b_math_previous_completed_20260619.csv`. Cross-run speedups in the HTML are reference comparisons against the 6/19 baseline, not strict same-run matched baselines.

