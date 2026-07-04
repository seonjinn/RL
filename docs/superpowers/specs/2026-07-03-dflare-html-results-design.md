# DFlare HTML Results Design

## Goal

Add completed DFlare benchmark results to the existing vLLM standalone HTML
report in a compact, reproducible format. Keep vLLM 0.24 native-method rows
and AngelSlim-native DFlare rows clearly separated because they use different
runtimes.

## Scope

- Cover Qwen3-8B DFlare results for Math and SWE at temperature 0 and 1.
- Present native 32K results and YaRN 64K/total-128K results in separate
  context-profile sections.
- Include only jobs with a completed result JSON. Running, pending, failed,
  timed-out, and result-less jobs do not appear in performance tables.
- Preserve the existing vLLM standalone report as the main entry point.
- Do not add charts until enough matched rows exist to make them meaningful.

## Data Flow

1. Pull completed result JSON files from the pinned Lyris result roots into a
   versioned local report-data directory.
2. Normalize each completed row to the canonical SpecDec fields: domain,
   temperature, top-p, ISL, OSL, batch size, method, K, throughput,
   acceptance rate, mean accepted length, job ID, backend, and source path.
3. Match a baseline only when domain, model, temperature, top-p, batch size,
   ISL, OSL, context-extension configuration, and runtime backend are equal.
4. Generate the HTML section from normalized data rather than manually
   embedding metric values.

## Report Layout

Add a `vLLM 0.24 / DFlare` section to the existing standalone results page.
Start with a compact methodology band containing model, runtime, GPU,
sampling, context profiles, and the SDPA fallback caveat.

Render one table per context profile:

| Domain | Temperature | ISL | OSL | Batch | Method / K | tok/s/GPU | Speedup | Acceptance | Mean accept length | Job ID |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|

The sections are:

- Native 32K: ISL 4,096 and OSL 32,768.
- YaRN 64K: ISL 4,096 and OSL 65,536.
- YaRN total-128K: ISL 4,096 and OSL 126,976, total context 131,072.

Rows are ordered by context profile, domain, temperature, and method. Existing
responsive table styling is reused so the report remains readable on narrow
screens.

## Comparison Rules

- Compute throughput speedup only against an exact matched baseline from the
  same runtime and backend.
- Show `waiting matched baseline` when an exact baseline is unavailable.
- Never compare AngelSlim-native DFlare throughput directly with a vLLM-native
  baseline as a formal speedup.
- Show `n/a` rather than zero for absent metrics.
- Label SDPA fallback results because the missing FlashAttention kernel can
  reduce end-to-end performance.

## Generated Artifacts

- A normalized DFlare CSV or JSON artifact under
  `experiments/vllm_024_dynamicsd/report/`.
- Updated dated and `latest` vLLM standalone HTML pages under
  `public/reports/` and their corresponding local `docs/` copies when the
  existing builder requires both.
- A reproducible builder or collector change that can be rerun as additional
  completed jobs arrive.

## Validation

- Unit-test completed-only filtering, exact baseline matching, missing-metric
  rendering, and deterministic row ordering.
- Run the existing report builders and Python compilation checks.
- Parse the generated HTML to verify valid table structure and required
  headings.
- Preview the page locally at desktop and mobile widths and confirm that
  headers, numeric values, and job IDs remain legible.
- Commit and push the generated report only after the underlying result data
  and builder validation pass.
