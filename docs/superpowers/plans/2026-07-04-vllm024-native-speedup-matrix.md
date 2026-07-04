# vLLM 0.24 Native Speedup Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the vLLM 0.24 native row tables with readable profile/domain/temperature batch-speedup matrices while preserving all detailed metrics in a collapsed table.

**Architecture:** Keep normalization and exact baseline matching unchanged. Refactor only the native HTML renderer into validated matrix-cell helpers plus the existing detailed renderer, then add scoped matrix CSS to the latest standalone page builder. Generated artifacts continue to consume the same canonical CSV.

**Tech Stack:** Python 3.12, pandas, pytest, Pyright, static HTML/CSS.

## Global Constraints

- Matrix columns are exactly `Model | Method | B1 | B2 | B4 | B8 | B16 | B32`.
- Slice matrices by context profile, domain, and temperature.
- Render speedup below 1.00x red, exactly 1.00x neutral gray, and above 1.00x blue with bounded intensity.
- Append `†` and an amber border to partial cells.
- Render unavailable batches as `n/a` and unmatched existing rows as `waiting baseline`.
- Preserve detailed throughput, latency speedup, acceptance, mean accepted length, source, and completion status in a collapsed detail table.
- Keep exact baseline matching, CSV schemas, DFlare sections, legacy sections, and NeMo-RL pages unchanged.
- Reject duplicate display keys instead of silently aggregating rows.

---

### Task 1: Native Matrix Renderer

**Files:**
- Modify: `scripts/vllm024_profile_report.py:395-507`
- Modify: `tests/test_vllm024_profile_report.py:582-625`

**Interfaces:**
- Consumes: exact-matched native rows from `match_profile_baselines(rows)`.
- Produces: `_speedup_cell(row: object | None) -> str`, `_profile_matrix(rows: pd.DataFrame, profile: str) -> str`, and `render_profile_section(rows: pd.DataFrame) -> str`.

- [ ] **Step 1: Add failing matrix tests**

Add tests that build synthetic Native 32K rows for baseline, speedup `2.25x`, slowdown `0.75x`, partial `0.50x`, missing batch, and unmatched baseline. Assert:

```python
assert '<table class="native-speedup-matrix">' in rendered
assert all(f">B{batch}<" in rendered for batch in (1, 2, 4, 8, 16, 32))
assert 'class="speed-cell speedup"' in rendered
assert 'class="speed-cell neutral"' in rendered
assert 'class="speed-cell slowdown"' in rendered
assert 'class="speed-cell slowdown partial"' in rendered
assert "0.50x†" in rendered
assert 'class="speed-cell empty">n/a</td>' in rendered
assert "waiting baseline" in rendered
assert "spec&lt;&amp;&gt;.json" in rendered
```

Add a duplicate-key test where two rows share
`(profile, domain, temperature, model, method, k, batch_size)`:

```python
with pytest.raises(ValueError, match="duplicate native matrix cell"):
    module.render_profile_section(rows)
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run:

```bash
python3 -m pytest -q tests/test_vllm024_profile_report.py -k 'render_profile or duplicate_native_matrix'
```

Expected: FAIL because the matrix classes and duplicate-display validation do not exist.

- [ ] **Step 3: Implement validated pivot helpers**

Add constants and helpers equivalent to:

```python
BATCH_ORDER = (1, 2, 4, 8, 16, 32)
DISPLAY_KEYS = [
    "context_profile", "domain", "temperature", "model", "method", "k", "batch_size"
]

def _validate_matrix_cells(rows: pd.DataFrame) -> None:
    duplicate = rows.duplicated(DISPLAY_KEYS, keep=False)
    if duplicate.any():
        raise ValueError("duplicate native matrix cell")
```

Implement `_speedup_cell` so it:

- formats finite `throughput_speedup` as `N.NNx`;
- emits `waiting baseline` for an existing unmatched row;
- emits `n/a` for a missing batch;
- assigns `speedup`, `neutral`, or `slowdown` classes;
- adds `partial` and `†` for partial rows;
- writes an escaped `title` containing source status, source path, tok/s/GPU,
  acceptance, and mean accepted length.

Build each table from unique `(model, method, k)` rows and lookup each batch
without `pivot_table(..., aggfunc="last")`.

- [ ] **Step 4: Preserve the detailed table in a collapsed block**

Rename the existing `_profile_table` to `_profile_detail_table` and retain all
current columns. Render it after the matrices as:

```html
<details class="native-profile-details">
  <summary>Detailed native metrics and sources</summary>
  ...existing profile detail tables...
</details>
```

- [ ] **Step 5: Run focused tests and type checking**

Run:

```bash
python3 -m pytest -q tests/test_vllm024_profile_report.py
pyright scripts/vllm024_profile_report.py tests/test_vllm024_profile_report.py
```

Expected: all tests pass and Pyright reports `0 errors`.

- [ ] **Step 6: Commit Task 1**

```bash
git add scripts/vllm024_profile_report.py tests/test_vllm024_profile_report.py
git commit -s -m "feat: render native speedup matrices"
```

### Task 2: Scoped Matrix Styling and Page Integration

**Files:**
- Modify: `scripts/build_latest_specdec_html_pages.py:1721-1723`
- Modify: `tests/test_vllm024_report_integration.py:70-103`

**Interfaces:**
- Consumes: HTML classes emitted by Task 1.
- Produces: responsive heatmap presentation in the generated latest standalone report.

- [ ] **Step 1: Add failing integration assertions**

Extend the temp-output integration test with:

```python
assert html_text.count('class="native-profile-grid"') == 3
assert html_text.count('class="native-speedup-matrix"') == 12
assert "speed-cell slowdown" in html_text
assert "speed-cell speedup" in html_text
assert "speed-cell neutral" in html_text
assert "0.50x†" in html_text or "partial" in html_text
assert '<details class="native-profile-details">' in html_text
assert "vLLM 0.24 / DFlare Completed Results" in html_text
```

- [ ] **Step 2: Run the integration test and confirm failure**

Run:

```bash
python3 -m pytest -q tests/test_vllm024_report_integration.py::test_task5_latest_vllm_html_contains_native_and_status_sections
```

Expected: FAIL until the scoped matrix CSS is included.

- [ ] **Step 3: Add scoped responsive CSS**

Extend the existing `css` string with matrix-only rules:

```css
.native-profile-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
.native-matrix-panel{min-width:0}
.native-speedup-matrix{font-size:14px}
.native-speedup-matrix th,.native-speedup-matrix td{text-align:center;vertical-align:middle}
.native-speedup-matrix th:first-child,.native-speedup-matrix th:nth-child(2),
.native-speedup-matrix td:first-child,.native-speedup-matrix td:nth-child(2){text-align:left}
.speed-cell{font-weight:750;font-variant-numeric:tabular-nums;white-space:nowrap}
.speed-cell.slowdown{background:var(--matrix-red);color:#8f1d16}
.speed-cell.neutral{background:#edf1f5;color:#374151}
.speed-cell.speedup{background:var(--matrix-blue);color:var(--matrix-text)}
.speed-cell.partial{outline:2px solid #d89b22;outline-offset:-2px}
.speed-cell.empty{background:#f8fafc;color:#94a3b8}
.native-profile-details{margin-top:14px}
.native-profile-details summary{cursor:pointer;font-weight:750;color:#374151}
@media(max-width:1000px){.native-profile-grid{grid-template-columns:1fr}}
```

The renderer supplies bounded inline custom properties for blue/red intensity;
do not apply these rules to DFlare or legacy tables.

- [ ] **Step 4: Run integration and no-side-effect tests**

Run:

```bash
python3 -m pytest -q tests/test_vllm024_report_integration.py tests/test_build_latest_specdec_html_pages.py
pyright scripts/build_latest_specdec_html_pages.py tests/test_vllm024_report_integration.py
```

Expected: all tests pass, historical reports remain byte-identical, and Pyright reports `0 errors`.

- [ ] **Step 5: Commit Task 2**

```bash
git add scripts/build_latest_specdec_html_pages.py tests/test_vllm024_report_integration.py
git commit -s -m "style: add native matrix heatmap"
```

### Task 3: Rebuild and Verify Published Artifacts

**Files:**
- Modify through builder: `docs/vllm_standalone_results_latest.html`
- Modify through builder: `public/reports/vllm_standalone_results_latest.html`
- Modify through builder when counts or links change: `public/index.html`
- Modify through builder when counts or links change: `docs/specdec_reports_index_latest.html`

**Interfaces:**
- Consumes: Task 1 renderer, Task 2 CSS, existing canonical CSV.
- Produces: local and Pages-ready latest HTML artifacts.

- [ ] **Step 1: Rebuild only latest artifacts**

Run:

```bash
python3 scripts/build_latest_specdec_html_pages.py
python3 scripts/build_pages_index.py
```

Expected: latest native section contains 12 matrices and the DFlare sections remain present.

- [ ] **Step 2: Restore unrelated deterministic-output churn**

Inspect `git status --short`. Restore only builder-generated files unrelated to
the native matrix change, while keeping the latest standalone report and index
artifacts. Never restore user-authored changes.

- [ ] **Step 3: Run full focused verification**

Run:

```bash
python3 -m pytest -q \
  tests/test_vllm024_profile_report.py \
  tests/test_vllm024_report_integration.py \
  tests/test_vllm024_dflare_report.py \
  tests/test_build_latest_specdec_html_pages.py \
  tests/test_build_pages_index.py
pyright \
  scripts/vllm024_profile_report.py \
  scripts/build_latest_specdec_html_pages.py \
  tests/test_vllm024_profile_report.py \
  tests/test_vllm024_report_integration.py
git diff --check
```

Expected: all tests pass, Pyright reports `0 errors`, and `git diff --check` emits no output.

- [ ] **Step 4: Parse and visually inspect HTML**

Parse the latest docs/public files with `html.parser.HTMLParser`, then open
`public/reports/vllm_standalone_results_latest.html`. Verify desktop and narrow
layouts, readable text contrast, red cells below 1.00x, blue cells above 1.00x,
neutral baseline cells, partial amber outlines, collapsed details, and unchanged
DFlare sections.

- [ ] **Step 5: Commit and push artifacts**

```bash
git add \
  docs/vllm_standalone_results_latest.html \
  public/reports/vllm_standalone_results_latest.html \
  public/index.html \
  docs/specdec_reports_index_latest.html
git commit -s -m "docs: publish native speedup matrices"
git push origin codex/vllm024-dynamicsd
```

