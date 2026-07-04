# Task 3 Report: Normalize vLLM-Native Profile Results

## Scope

- Created `scripts/vllm024_profile_report.py`
- Created `tests/test_vllm024_profile_report.py`

## Requirements Source

Command:

```bash
sed -n '1,240p' .superpowers/sdd/task-3-brief.md
```

Output:

```text
### Task 3: Normalize vLLM-Native Profile Results

**Files:**
- Create: `scripts/vllm024_profile_report.py`
- Create: `tests/test_vllm024_profile_report.py`

**Interfaces:**
- Produces: `load_profile_results(paths: Iterable[Path]) -> pd.DataFrame`
- Produces: `match_profile_baselines(rows: pd.DataFrame) -> pd.DataFrame`
- Produces: `render_profile_section(rows: pd.DataFrame) -> str`

- [ ] **Step 1: Write failing parser tests**

Cover baseline and speculative JSON fixtures for Native 32K, YaRN 64K, and
YaRN total-128K. Verify one row per batch size, exact setup matching,
throughput speedup, latency speedup, acceptance, mean accepted length, K, and
unmatched-baseline behavior.

- [ ] **Step 2: Run the focused parser test and confirm failure**

Run: `python3 -m pytest -q tests/test_vllm024_profile_report.py`

Expected: import failure because the profile module does not exist.

- [ ] **Step 3: Implement normalization and matching**

Normalize `config` and each `results` item into canonical columns. Derive the
profile from ISL/OSL and position encoding, derive method/K from
`speculative_config`, and join against exact baseline keys.

- [ ] **Step 4: Implement compact profile HTML**

Render separate Native 32K, YaRN 64K, and total-128K tables with methodology,
throughput, speedup, latency speedup, acceptance, mean length, and source.

- [ ] **Step 5: Run the focused parser tests**

Run: `python3 -m pytest -q tests/test_vllm024_profile_report.py`

Expected: all profile tests pass.
```

## Red Step

Added focused tests first in `tests/test_vllm024_profile_report.py` to lock:

- real-input normalization over the 60 checked-in vLLM-native JSON files
- one row per persisted batch result, including timeout-interrupted partial
  sources
- exact baseline matching on runtime/model/domain/temperature/top-p/batch/ISL/
  OSL/profile/position encoding/CUDA graph/setup
- unmatched-baseline behavior when setup diverges
- compact HTML rendering with visually distinct `complete` and `partial`
  source badges

Command:

```bash
python3 -m pytest -q tests/test_vllm024_profile_report.py
```

Output:

```text
FFFF                                                                     [100%]
=================================== FAILURES ===================================
E       AssertionError: profile report module is not implemented
4 failed in 3.06s
```

## Implementation

Implemented `scripts/vllm024_profile_report.py` with three public entry points:

- `load_profile_results(paths)` flattens each JSON payload into one row per
  persisted batch result, preserving partial rows from interrupted files
- `match_profile_baselines(rows)` joins only against exact vLLM-native
  baseline rows and computes throughput and latency speedups
- `render_profile_section(rows)` renders separate Native 32K, YaRN 64K, and
  YaRN total-128K tables with inline source-status badges

Key normalization decisions:

- source file status is normalized to `complete` or `partial`
- method is derived from `config["mode"]`
- `k` is derived from `config["speculative_config"]["num_speculative_tokens"]`
- position encoding is derived from the model path (`native` vs `yarn4`)
- profile is derived from ISL/OSL plus position encoding
- baseline matching excludes AngelSlim by using only vLLM-native baseline rows
  and the exact join key:
  `runtime, model, domain, temperature, top_p, batch_size, isl, osl, context_profile, position_encoding, cuda_graph, setup`

## Green Step

Command:

```bash
python3 -m pytest -q tests/test_vllm024_profile_report.py
```

Output:

```text
....                                                                     [100%]
4 passed in 0.57s
```

## Additional Verification

Compile check:

```bash
python3 -m py_compile scripts/vllm024_profile_report.py tests/test_vllm024_profile_report.py
```

Exit status: `0`

Real-input smoke check:

```bash
python3 - <<'PY'
from pathlib import Path
from scripts.vllm024_profile_report import load_profile_results, match_profile_baselines
root = Path('experiments/vllm_024_dynamicsd/report/20260704_vllm_native_completed')
rows = match_profile_baselines(load_profile_results(sorted(root.rglob('*.json'))))
print('rows', len(rows))
print('source_status', rows['source_status'].value_counts().to_dict())
print('profiles', rows['context_profile'].value_counts().to_dict())
print('methods', rows['method'].value_counts().to_dict())
print('unmatched_nonbaseline', int(rows.loc[rows['method'].ne('baseline') & rows['throughput_speedup'].isna()].shape[0]))
print('partial_rows', int(rows.loc[rows['source_status'].eq('partial')].shape[0]))
print('partial_matched', int(rows.loc[rows['source_status'].eq('partial') & rows['throughput_speedup'].notna()].shape[0]))
PY
```

Output:

```text
rows 154
source_status {'complete': 136, 'partial': 18}
profiles {'Native 32K': 114, 'YaRN total-128K': 20, 'YaRN 64K': 20}
methods {'baseline': 32, 'dflash': 32, 'pard': 32, 'suffix': 32, 'pard2': 26}
unmatched_nonbaseline 0
partial_rows 18
partial_matched 18
```

## Self-Review

Commands:

```bash
git diff --check -- scripts/vllm024_profile_report.py tests/test_vllm024_profile_report.py
git status --short -- scripts/vllm024_profile_report.py tests/test_vllm024_profile_report.py .superpowers/sdd/task-3-report.md
```

Outputs:

```text
git diff --check:
[no output]

git status --short:
?? scripts/vllm024_profile_report.py
?? tests/test_vllm024_profile_report.py
?? .superpowers/sdd/task-3-report.md
```

Notes:

- Changes stay within the owned script, owned test file, and the requested
  task report.
- Raw report inputs under
  `experiments/vllm_024_dynamicsd/report/20260704_vllm_native_completed/`
  were read but not modified.
- Builder integration is intentionally left for later tasks.
