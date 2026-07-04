# Task 5 Report

## Scope Delivered

- Integrated `experiments/vllm_024_dynamicsd/report/vllm024_profiles_latest.csv`
  into the latest standalone HTML through
  `scripts/vllm024_profile_report.py::render_profile_section`.
- Extended the AngelSlim DFlare normalizer to emit visible baseline rows and
  exact-match spec rows only against AngelSlim baselines with matching runtime,
  model, domain, temp, top-p, batch, ISL, OSL, profile, and normalized setup.
- Kept DFlare failure/status rows separate from performance rows via a dedicated
  status section and dedicated published CSV.
- Published `vllm024_profiles_latest.csv`, `dflare_completed_latest.csv`, and
  `dflare_job_status_latest.csv` under `public/data`.
- Updated the Pages index summary and links for the native profile CSV, DFlare
  completed CSV, and DFlare failure/status CSV.

## Files Changed

- `scripts/build_latest_specdec_html_pages.py`
- `scripts/vllm024_dflare_report.py`
- `scripts/build_pages_index.py`
- `tests/test_vllm024_dflare_report.py`
- `tests/test_vllm024_report_integration.py`

## Verification

### Focused pytest

Command:

```bash
python3 -m pytest -q \
  tests/test_vllm024_dflare_report.py \
  tests/test_vllm024_profile_report.py \
  tests/test_vllm024_report_integration.py \
  tests/test_build_latest_specdec_html_pages.py \
  tests/test_build_pages_index.py
```

Result:

- `28 passed in 2.98s`

### Bytecode compile

Command:

```bash
python3 -m py_compile \
  scripts/build_latest_specdec_html_pages.py \
  scripts/build_pages_index.py \
  scripts/vllm024_dflare_report.py \
  scripts/vllm024_profile_report.py
```

Result:

- exited `0`

### Fresh builder runs

Commands:

```bash
python3 scripts/build_latest_specdec_html_pages.py
python3 scripts/build_pages_index.py
```

Result:

- both exited `0`
- regenerated latest standalone HTML and report index

### HTML parse checks

Parsed successfully with `html.parser.HTMLParser`:

- `docs/vllm_standalone_results_latest.html`
- `public/index.html`
- `docs/specdec_reports_index_latest.html`

### Data/result checks

Observed after rebuild:

- `dflare_completed_latest.csv`: `12` performance rows across `8` completed jobs
- `dflare_job_status_latest.csv`: `12` status rows, including `3` `TIMEOUT`
  rows
- Completed jobs include `2272937`, `2272938`, and `2272941`
- Status rows include `2272942` with
  `gather_object_cuda_oom_after_generation`

## Notes / Concerns

- Future duplicate AngelSlim baseline rows for the same exact setup are
  resolved by taking the latest stable row during baseline lookup rather than
  raising. That keeps the builder usable when baseline-only and paired runs
  coexist, but it does mean baseline selection is policy-driven rather than
  erroring on duplicates.
- `build_pages_index.py` now treats the pinned 6/19 standalone page as
  navigable if either the local docs copy or the published `public/reports`
  copy exists. This was needed to keep the existing navigation test green in
  the current workspace state.
