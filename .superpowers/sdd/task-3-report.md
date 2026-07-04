# Task 3 Report: Normalize vLLM-Native Profile Results

## Scope

- Modified `scripts/vllm024_profile_report.py`
- Modified `tests/test_vllm024_profile_report.py`

## Review-Fix Summary

Implemented the Task 3 review findings in the owned files only:

- exact baseline matching now preserves `runtime_family`, full runtime
  provenance, exact target checkpoint path, and a normalized
  `setup_signature` built from non-speculative config
- baseline matching now prefers `complete` over `partial`, rejects remaining
  ambiguous duplicate exact keys, and merges with `validate="many_to_one"`
- loading, matching, and rendering are explicitly limited to
  `runtime_family == "vllm_native"` and ignore AngelSlim payloads
- synthetic coverage now exercises 64K/128K profiles, setup omission
  detection, runtime provenance mismatches, duplicate baselines, AngelSlim
  exclusion, variable-length job IDs, and HTML escaping
- malformed or missing `K` now renders as `n/a`
- pyright is clean for both owned files
- top-level `batch_sizes` and `prompt_count_loaded` are excluded from
  `setup_signature`, so targeted retry payloads can match full-sweep baselines
  on per-row batch size while nested setup differences still break the match
- setup normalization exclusions now apply only to the top-level config, so
  nested fields with the same names are preserved in the normalized signature
- real-input integration tests now skip cleanly when the 60-file corpus is not
  present, while synthetic fixtures fully cover the matching behavior

## Red Step

Expanded the focused test file first to encode the review requirements before
changing the implementation.

Command:

```bash
python3 -m pytest -q tests/test_vllm024_profile_report.py
```

Output:

```text
FFFFFF.                                                                  [100%]
6 failed, 1 passed in 0.98s
```

The failures were the expected contract gaps:

- missing `runtime_family`, `runtime_provenance`, `model_checkpoint`, and
  `setup_signature`
- AngelSlim rows were still loaded
- exact matching was too weak and allowed mismatched setup/runtime rows to
  bind to baselines
- duplicate baselines were not deduplicated or rejected
- render still attempted `int(row.k)` and crashed on missing `K`
- `setup_signature` still included top-level `batch_sizes` and
  `prompt_count_loaded`, which blocked targeted retry matching
- recursive key exclusion also erased nested fields that should have remained
  part of exact setup matching
- real-input tests were hard-pinned to the local 60-file corpus

## Implementation

Reworked `scripts/vllm024_profile_report.py` around the review findings:

### Loading

- detect `runtime_family` from the payload runtime object
- ignore non-vLLM-native payloads during `load_profile_results`
- normalize one row per persisted batch result
- preserve:
  - `runtime_family`
  - `runtime` display label
  - `runtime_provenance` as normalized runtime JSON excluding job-local env
    noise (`SLURM_JOB_ID`, `CUDA_VISIBLE_DEVICES`)
  - `model_checkpoint` as the exact target checkpoint path
  - `setup_signature` as normalized config JSON excluding only
    speculative-method-specific knobs:
    `draft_model`, `mode`, `speculative_config`, `tag`

### Matching

- exact baseline key now uses:
  `runtime_family, runtime_provenance, model_checkpoint, domain, temperature, top_p, batch_size, isl, osl, context_profile, position_encoding, cuda_graph, setup_signature`
- baseline lookup deterministically prefers `complete` over `partial`
- any remaining duplicate exact baseline keys raise:
  `ValueError("ambiguous duplicate baseline exact keys")`
- merge now uses `validate="many_to_one"` to block row multiplication

### Rendering

- render filters to `runtime_family == "vllm_native"`
- missing or malformed `K` renders as `n/a`
- source paths and text cells are HTML-escaped
- AngelSlim rows are excluded even if they are passed in a mixed DataFrame

### Second Review Adjustments

- `setup_signature` now excludes `batch_sizes` and `prompt_count_loaded` only
  at the top-level config, alongside the speculative-only knobs already
  excluded
- nested config values are normalized recursively without applying the
  top-level exclusion list to child dictionaries
- the synthetic test suite now proves:
  - targeted retry `[16, 32]` rows match a full-sweep baseline at row batches
    `16` and `32`
  - a nested setup difference with the same field names still prevents a match
  - real-input integration tests skip cleanly when the 60-file corpus is not
    available locally

## Green Step

Command:

```bash
python3 -m pytest -q tests/test_vllm024_profile_report.py
```

Output:

```text
.......                                                                  [100%]
7 passed in 0.54s
```

Second review verification:

```bash
python3 -m pytest -q tests/test_vllm024_profile_report.py
```

Output:

```text
........                                                                 [100%]
8 passed in 0.69s
```

## Additional Verification

Pyright:

```bash
pyright scripts/vllm024_profile_report.py tests/test_vllm024_profile_report.py
```

Output:

```text
0 errors, 0 warnings, 0 informations
```

Real-input smoke check on the 60 intentionally untracked Task 4 JSON files:

```bash
python3 - <<'PY'
from pathlib import Path
from scripts.vllm024_profile_report import load_profile_results, match_profile_baselines
root = Path('experiments/vllm_024_dynamicsd/report/20260704_vllm_native_completed')
loaded = load_profile_results(sorted(root.rglob('*.json')))
rows = match_profile_baselines(loaded)
print('loaded_rows', len(loaded))
print('matched_rows', len(rows))
print('runtime_family', loaded['runtime_family'].value_counts().to_dict())
print('source_status', loaded['source_status'].value_counts().to_dict())
print('profiles', loaded['context_profile'].value_counts().to_dict())
print('methods', loaded['method'].value_counts().to_dict())
print('unmatched_nonbaseline', int(rows.loc[rows['method'].ne('baseline') & rows['throughput_speedup'].isna()].shape[0]))
print('partial_matched', int(rows.loc[rows['source_status'].eq('partial') & rows['throughput_speedup'].notna()].shape[0]))
PY
```

Output:

```text
loaded_rows 154
matched_rows 154
runtime_family {'vllm_native': 154}
source_status {'complete': 136, 'partial': 18}
profiles {'Native 32K': 114, 'YaRN total-128K': 20, 'YaRN 64K': 20}
methods {'baseline': 32, 'dflash': 32, 'pard': 32, 'suffix': 32, 'pard2': 26}
unmatched_nonbaseline 0
partial_matched 18
```

## Self-Review

Commands:

```bash
git diff --check -- scripts/vllm024_profile_report.py tests/test_vllm024_profile_report.py .superpowers/sdd/task-3-report.md
git status --short -- scripts/vllm024_profile_report.py tests/test_vllm024_profile_report.py .superpowers/sdd/task-3-report.md
```

Outputs:

```text
git diff --check:
[no output]

git status --short:
 M scripts/vllm024_profile_report.py
 M tests/test_vllm024_profile_report.py
 M .superpowers/sdd/task-3-report.md
```

Notes:

- The 60 raw JSON files under
  `experiments/vllm_024_dynamicsd/report/20260704_vllm_native_completed/`
  were read for verification only and were not staged or modified.
- The task remains scoped to the owned report module, owned tests, and the
  requested task report.
