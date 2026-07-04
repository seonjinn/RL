# vLLM 0.24 Native Speedup Matrix Design

## Goal

Replace the long row-by-row vLLM 0.24 native profile tables with compact
batch-size speedup matrices matching the readable June 19 report style. Keep
the DFlare performance and failure/status sections unchanged.

## Scope

- Change only the `vLLM 0.24 / Native Profile Results` renderer.
- Preserve the existing exact-baseline matching and canonical CSV schema.
- Render one matrix for each context profile, domain, and temperature slice.
- Preserve detailed throughput, acceptance, source, and status data in a
  collapsed native HTML details section.
- Do not change DFlare tables, legacy standalone sections, or NeMo-RL pages.

## Matrix Layout

Each context profile is a subsection. Within it, the four domain/sampling
slices use a responsive two-column grid:

- Math, temperature 0.0
- Math, temperature 1.0
- SWE, temperature 0.0
- SWE, temperature 1.0

Each matrix uses these columns:

`Model | Method | B1 | B2 | B4 | B8 | B16 | B32`

Rows are grouped by `(model, method, K)` so different speculative token counts
cannot collapse into one row. Profiles that currently contain only B1 still
show all six batch columns, with unavailable cells rendered as `n/a`.

## Cell Semantics

- Matched result: show throughput speedup to two decimals.
- Speedup below 1.00x: red background, with stronger red for larger slowdown.
- Speedup exactly 1.00x: neutral gray background.
- Speedup above 1.00x: blue background, with stronger blue for larger gain.
- Missing batch: muted gray `n/a`.
- Unmatched existing result: muted `waiting baseline`.
- Partial result: append `†` and add a thin amber border.
- Preserve source path, completion state, throughput, acceptance, and mean
  accepted length in an escaped `title` tooltip.

Color intensity is bounded so extreme values remain readable. Text color
switches to white only on sufficiently dark backgrounds.

## Data Flow

1. Load the already normalized native profile rows.
2. Keep the existing runtime-family and target-profile filters.
3. Validate uniqueness of
   `(profile, domain, temperature, model, method, K, batch_size)`.
4. Raise a clear error for duplicate display cells instead of silently choosing
   the last row.
5. Build the 12 profile/domain/temperature matrices in deterministic order.
6. Render the existing detailed row table inside a collapsed `<details>` block.

Baseline matching remains upstream and unchanged. The renderer never computes
or rematches speedups.

## Responsive Presentation

- Reuse the June 19 matrix alignment: Model/Method left aligned and batch cells
  centered with tabular numerals.
- Use two columns on wide screens and one column below 1000px.
- Retain horizontal scrolling for narrow screens.
- Scope matrix CSS to the native matrix classes so other report tables do not
  change.

## Validation

Tests will verify:

- all B1/B2/B4/B8/B16/B32 headings;
- deterministic profile/domain/temperature ordering;
- blue, neutral, red, missing, waiting-baseline, and partial cell states;
- partial PARD-2 markers and missing B16/B32 cells;
- escaped source/tooltips;
- duplicate display-cell rejection;
- unchanged exact baseline-matching behavior;
- generated latest HTML contains the matrix and leaves DFlare sections intact;
- tests and builders leave unrelated historical artifacts unchanged.

