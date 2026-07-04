# DFlare HTML Results Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish completed Qwen3-8B DFlare 32K/64K/total-128K results as compact tables in the existing vLLM standalone HTML report and publish the approved design specification as HTML.

**Architecture:** Keep DFlare result parsing and exact-baseline matching in a focused module, then import its rendered section into the existing standalone report builder. Store small completed raw JSON artifacts under the experiment report tree so report generation is reproducible without cluster access. Use a generic Markdown-to-HTML renderer for design specifications and publish the resulting page under `public/specs/`.

**Tech Stack:** Python 3.11+, pandas, Python-Markdown, pytest, static HTML/CSS, GitLab Pages.

## Global Constraints

- Include only result JSON files whose top-level `status` is exactly `complete`.
- Keep AngelSlim-native DFlare separate from vLLM-native methods.
- Compute speedup only for exact matches on runtime, domain, model, temperature, top-p, batch size, ISL, OSL, position encoding, and backend.
- Render `waiting matched baseline` when no exact AngelSlim baseline exists.
- Label runs that use PyTorch SDPA because FlashAttention is unavailable.
- Present 32K, 64K, and total-128K profiles as separate compact tables.
- Preserve Math/SWE and temperature 0/1 as explicit table columns.
- Publish specifications in Markdown and HTML.

---

### Task 1: Completed DFlare Result Normalization

**Files:**
- Create: `scripts/vllm024_dflare_report.py`
- Create: `tests/test_vllm024_dflare_report.py`
- Read: `experiments/vllm_024_dynamicsd/report/**/result.json`

**Interfaces:**
- Consumes: `Iterable[Path]` containing AngelSlim result JSON files.
- Produces: `load_completed_dflare_results(paths: Iterable[Path]) -> pandas.DataFrame` and `match_dflare_baselines(rows: pandas.DataFrame) -> pandas.DataFrame`.

- [ ] **Step 1: Write failing completed-only and normalization tests**

```python
def test_load_completed_dflare_results_excludes_noncomplete(tmp_path: Path) -> None:
    complete = write_result(tmp_path / "complete.json", status="complete")
    running = write_result(tmp_path / "running.json", status="running")
    rows = module.load_completed_dflare_results([complete, running])
    assert rows["status"].tolist() == ["complete"]
    assert rows.iloc[0]["method"] == "dflare_k16"


def test_spec_only_result_has_no_invented_speedup(tmp_path: Path) -> None:
    result = write_result(tmp_path / "result.json", status="complete", run_mode="spec")
    rows = module.match_dflare_baselines(
        module.load_completed_dflare_results([result])
    )
    assert pandas.isna(rows.iloc[0]["speedup"])
    assert rows.iloc[0]["speedup_label"] == "waiting matched baseline"
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `python3 -m pytest -q tests/test_vllm024_dflare_report.py`

Expected: FAIL because `scripts/vllm024_dflare_report.py` does not exist.

- [ ] **Step 3: Implement typed normalization and exact matching**

```python
MATCH_COLUMNS = [
    "runtime", "domain", "model", "temperature", "top_p", "batch_size",
    "isl", "osl", "position_encoding", "backend",
]


def load_completed_dflare_results(paths: Iterable[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            continue
        rows.append(normalize_dflare_result(payload, path))
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


def match_dflare_baselines(rows: pd.DataFrame) -> pd.DataFrame:
    matched = rows.copy()
    matched["speedup"] = pd.NA
    matched["speedup_label"] = "waiting matched baseline"
    # Group by MATCH_COLUMNS and compare only AngelSlim baseline/spec pairs.
    return fill_exact_matched_speedups(matched)
```

- [ ] **Step 4: Run focused tests**

Run: `python3 -m pytest -q tests/test_vllm024_dflare_report.py`

Expected: all tests pass.

- [ ] **Step 5: Commit normalization code**

```bash
git add scripts/vllm024_dflare_report.py tests/test_vllm024_dflare_report.py
git commit -s -m "Normalize completed DFlare results"
```

### Task 2: Compact DFlare HTML Section

**Files:**
- Modify: `scripts/vllm024_dflare_report.py`
- Modify: `scripts/build_latest_specdec_html_pages.py`
- Modify: `tests/test_vllm024_dflare_report.py`

**Interfaces:**
- Consumes: normalized completed DFlare rows from Task 1.
- Produces: `render_dflare_section(rows: pandas.DataFrame) -> str` and a generated CSV snapshot.

- [ ] **Step 1: Write failing table-rendering tests**

```python
def test_render_groups_context_profiles_and_required_columns() -> None:
    html = module.render_dflare_section(example_rows())
    assert "Native 32K" in html
    assert "YaRN 64K" in html
    assert "YaRN total-128K" in html
    for heading in ["Domain", "Temperature", "ISL", "OSL", "Batch", "Method / K", "tok/s/GPU", "Speedup", "Acceptance", "Mean accept length", "Job ID"]:
        assert heading in html


def test_render_does_not_show_incomplete_rows() -> None:
    html = module.render_dflare_section(example_rows(statuses=["complete", "running"]))
    assert "completed-job" in html
    assert "running-job" not in html
```

- [ ] **Step 2: Run tests and verify the renderer is missing**

Run: `python3 -m pytest -q tests/test_vllm024_dflare_report.py`

Expected: FAIL with `AttributeError: render_dflare_section`.

- [ ] **Step 3: Implement compact profile tables**

```python
def render_dflare_section(rows: pd.DataFrame) -> str:
    completed = rows.loc[rows["status"].eq("complete")].copy()
    sections = [render_profile_table(completed, profile) for profile in PROFILE_ORDER]
    return (
        '<section class="section" id="vllm024-dflare">'
        '<h2>vLLM 0.24 / DFlare Completed Results</h2>'
        '<p class="note">AngelSlim-native; SDPA fallback. Speedup requires an exact AngelSlim baseline.</p>'
        + "".join(sections)
        + "</section>"
    )
```

- [ ] **Step 4: Integrate the section into the existing builder**

Import `load_completed_dflare_results`, `match_dflare_baselines`, and
`render_dflare_section` from `vllm024_dflare_report`. Define
`DFLARE_RESULT_ROOT = ROOT / "experiments/vllm_024_dynamicsd/report"`, load
`DFLARE_RESULT_ROOT.glob("**/result.json")`, and insert the rendered fragment
after the methodology section and before historical batch matrices. Write the
normalized frame to `DFLARE_RESULT_ROOT / "dflare_completed_latest.csv"`.

- [ ] **Step 5: Run focused and existing builder tests**

Run: `python3 -m pytest -q tests/test_vllm024_dflare_report.py tests/test_vllm024_dynamicsd.py`

Expected: all tests pass.

- [ ] **Step 6: Commit HTML integration**

```bash
git add scripts/vllm024_dflare_report.py scripts/build_latest_specdec_html_pages.py tests/test_vllm024_dflare_report.py
git commit -s -m "Add DFlare tables to standalone report"
```

### Task 3: HTML Specification Publishing

**Files:**
- Create: `scripts/render_markdown_spec.py`
- Create: `tests/test_render_markdown_spec.py`
- Create: `public/specs/2026-07-03-dflare-html-results-design.html`
- Modify: `scripts/build_pages_index.py`

**Interfaces:**
- Consumes: a Markdown path, output HTML path, and page title.
- Produces: `render_spec(markdown_path: Path, output_path: Path, title: str) -> None`.

- [ ] **Step 1: Write a failing specification-rendering test**

```python
def test_render_spec_preserves_headings_tables_and_code(tmp_path: Path) -> None:
    source = tmp_path / "spec.md"
    source.write_text("# Design\n\n| A | B |\n|---|---|\n| `x` | y |\n", encoding="utf-8")
    output = tmp_path / "spec.html"
    module.render_spec(source, output, "Design")
    html = output.read_text(encoding="utf-8")
    assert "<h1>Design</h1>" in html
    assert "<table>" in html
    assert "<code>x</code>" in html
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `python3 -m pytest -q tests/test_render_markdown_spec.py`

Expected: FAIL because the renderer does not exist.

- [ ] **Step 3: Implement the renderer with the existing Python-Markdown dependency**

```python
def render_spec(markdown_path: Path, output_path: Path, title: str) -> None:
    body = markdown.markdown(
        markdown_path.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code"],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page_template(title, body), encoding="utf-8")
```

- [ ] **Step 4: Render and index the approved design**

Run:

```bash
python3 scripts/render_markdown_spec.py \
  docs/superpowers/specs/2026-07-03-dflare-html-results-design.md \
  public/specs/2026-07-03-dflare-html-results-design.html \
  --title "DFlare HTML Results Design"
```

Add the generated specification page to the report links emitted by `scripts/build_pages_index.py`.

- [ ] **Step 5: Run tests and commit**

Run: `python3 -m pytest -q tests/test_render_markdown_spec.py`

Expected: all tests pass.

```bash
git add scripts/render_markdown_spec.py tests/test_render_markdown_spec.py public/specs/2026-07-03-dflare-html-results-design.html scripts/build_pages_index.py
git commit -s -m "Publish experiment specifications as HTML"
```

### Task 4: Collect, Build, Validate, and Publish

**Files:**
- Create: `experiments/vllm_024_dynamicsd/report/20260703_dflare_completed/**/result.json`
- Modify: `experiments/vllm_024_dynamicsd/report/dflare_completed_latest.csv`
- Modify: `docs/vllm_standalone_results_20260621.html`
- Modify: `docs/vllm_standalone_results_latest.html`
- Modify: `public/reports/vllm_standalone_results_20260621.html`
- Modify: `public/reports/vllm_standalone_results_latest.html`
- Modify: `public/index.html`

**Interfaces:**
- Consumes: completed 32K, 64K, and total-128K DFlare JSON files from pinned Lyris roots.
- Produces: reproducible local artifacts and published Pages HTML.

- [ ] **Step 1: Pull only completed result JSON artifacts**

Use one-shot SSH to list result files, inspect top-level status, and copy only `status=complete` files into `experiments/vllm_024_dynamicsd/report/20260703_dflare_completed/`. Preserve profile/domain/temperature directories and record each SLURM job ID in the local relative path or a sibling manifest.

- [ ] **Step 2: Generate reports**

Run:

```bash
python3 scripts/build_latest_specdec_html_pages.py
python3 scripts/build_pages_index.py
```

Expected: the latest and dated standalone pages contain `vllm024-dflare`, while performance tables contain no incomplete DFlare rows.

- [ ] **Step 3: Validate code and HTML**

Run:

```bash
python3 -m pytest -q tests/test_vllm024_dflare_report.py tests/test_render_markdown_spec.py tests/test_vllm024_dynamicsd.py
python3 -m py_compile scripts/vllm024_dflare_report.py scripts/render_markdown_spec.py scripts/build_latest_specdec_html_pages.py scripts/build_pages_index.py
python3 -c 'from pathlib import Path; from html.parser import HTMLParser; [HTMLParser().feed(path.read_text()) for path in [Path("public/reports/vllm_standalone_results_latest.html"), Path("public/specs/2026-07-03-dflare-html-results-design.html")]]'
```

Expected: tests and compilation pass; HTML parsing exits zero.

- [ ] **Step 4: Preview desktop and mobile layouts**

Serve `public/` locally, capture desktop and mobile screenshots with the available browser tooling, and verify that context headings, table headers, values, and job IDs do not overlap or truncate incoherently.

- [ ] **Step 5: Commit and push generated artifacts**

```bash
git add experiments/vllm_024_dynamicsd/report docs/vllm_standalone_results_20260621.html docs/vllm_standalone_results_latest.html public/reports/vllm_standalone_results_20260621.html public/reports/vllm_standalone_results_latest.html public/index.html
git commit -s -m "Publish completed DFlare benchmark results"
git push origin codex/vllm024-dynamicsd
```
