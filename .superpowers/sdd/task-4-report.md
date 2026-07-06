# Task 4 Report: SPEED-Bench Dataset Adapter

## Scope

Owned files:

- `experiments/vllm_024_dynamicsd/speedbench_dataset.py`
- `experiments/vllm_024_dynamicsd/stage_speedbench.sh`
- `tests/test_vllm024_dynamicsd.py`

Unrelated worktree state was left untouched.

## RED

I wrote the Task 4 tests first, then ran the focused subset before creating the
adapter module or staging script.

Command:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py -k speedbench_dataset
```

Output:

```text
5 failed, 85 deselected in 0.30s
```

Expected failures:

- `speedbench_dataset.py` did not exist
- `stage_speedbench.sh` did not exist

## Implementation

### Dataset adapter

Created `speedbench_dataset.py` with:

- pinned SPEED-Bench and Model Optimizer revision constants
- typed `SpeedBenchRecord` rows with canonical SHA256 hashes
- deterministic record normalization that preserves full multi-turn `turns`
- explicit rejection for masked or unresolved rows
- balanced Sync-RL overlay selection as exactly three 16-row batches with
  6/5/5 entropy rotation
- deterministic prepared-data manifest generation with relative parquet paths,
  pinned provenance, nominal ISL, nullable actual tokenizer ISL, and per-file
  checksums

### Staging script

Created `stage_speedbench.sh` with:

- pinned `nvidia/SPEED-Bench` dataset revision
  `487aa718444e816458d1a0a52bfce7a454285cf4`
- pinned `NVIDIA/Model-Optimizer` revision
  `43fee0cd70fa9e5f85782d52a4bd8ad9c8b88446`
- deterministic run root naming based on the pinned revisions
- no home-directory mounts or paths
- `DRY_RUN` and `TEST_ONLY` temp-root behavior so non-mutating modes do not
  leave staged artifacts behind
- explicit dataset-license and Model Optimizer license capture
- deterministic manifest/checksum generation for every resolved parquet file
- a local patch step that forces Model Optimizer `prepare_data.py` to load the
  pinned SPEED-Bench dataset revision instead of an unpinned latest revision

### Tests

Added coverage for:

- balanced 48-row throughput selection
- same-seed deterministic selection and different-seed reshuffling
- multi-turn preservation
- masked-row rejection
- manifest pinning and checksum generation
- dry-run staging script provenance and license behavior
- home-path regression guard updated to include `stage_speedbench.sh`

## Focused GREEN

Command:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py -k speedbench_dataset
```

Output:

```text
5 passed, 85 deselected in 0.14s
```

## Broader Verification

### Shell syntax

Command:

```bash
bash -n experiments/vllm_024_dynamicsd/stage_speedbench.sh
```

Result:

```text
exit 0
```

### Targeted Pyright

Command:

```bash
pyright experiments/vllm_024_dynamicsd/speedbench_dataset.py tests/test_vllm024_dynamicsd.py
```

Output:

```text
0 errors, 0 warnings, 0 informations
```

### Full requested test file

Command:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py
```

Output:

```text
90 passed in 12.15s
```

### Diff hygiene

Command:

```bash
git diff --check -- experiments/vllm_024_dynamicsd/speedbench_dataset.py experiments/vllm_024_dynamicsd/stage_speedbench.sh tests/test_vllm024_dynamicsd.py
```

Result:

```text
no output
```

## Concerns

- The local environment does not expose `pyright` as `python3 -m pyright`; the
  verification used the installed `pyright` binary directly.
- The staging script’s runtime path that executes Model Optimizer was not run
  end to end here because the task required non-mutating verification only.

## Commit

Command:

```bash
git add experiments/vllm_024_dynamicsd/speedbench_dataset.py experiments/vllm_024_dynamicsd/stage_speedbench.sh tests/test_vllm024_dynamicsd.py
git add -f .superpowers/sdd/task-4-report.md
git commit -s -m "feat: add pinned SPEED-Bench adapter"
```

Result:

```text
signed commit created successfully for the four Task 4 files
```

## Task 4 Review Fixes

### Review findings addressed

1. Switched the prepared-data contract to the real upstream layout:
   `output_dir/speed/<config>/test.parquet`.
2. Removed nested unescaped shell interpolation from the rendered staging path
   and passed the inner payload via positional args and arrays.
3. Made both `DRY_RUN` and `TEST_ONLY` skip `git pull` and leave the intended
   source/result roots untouched.
4. Replaced fixed `/tmp` cleanup with unique `mktemp -d` workspaces and trap
   cleanup.
5. Verified both dataset and Model Optimizer license files are present and
   nonempty, hashed them, and recorded their relative names and hashes.
6. Derived one sorted relative parquet set under the prepared `speed` root and
   used that exact set for manifest entries and checksum output, rejecting
   missing or unexpected parity.

### RED

Command:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py -k 'speedbench_dataset or speedbench_stage'
```

Output before the fixes:

```text
7 failed, 4 passed, 85 deselected in 1.51s
```

Observed failures:

- manifest builder still assumed `prepared/<config>/test.parquet`
- manifest builder did not accept license roots or hash license files
- dry-run still rendered the old manifest root
- `TEST_ONLY` still called `git pull`
- scheduler identifiers were not validated
- fixed `/tmp` cleanup remained in the rendered staging path
- the rendered sbatch payload still relied on unsafe nested interpolation

### GREEN

Command:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py -k 'speedbench_dataset or speedbench_stage'
```

Output:

```text
11 passed, 85 deselected in 1.19s
```

### Review-fix verification

Shell syntax:

```bash
bash -n experiments/vllm_024_dynamicsd/stage_speedbench.sh
```

Result:

```text
exit 0
```

Targeted Pyright:

```bash
pyright experiments/vllm_024_dynamicsd/speedbench_dataset.py tests/test_vllm024_dynamicsd.py
```

Output:

```text
0 errors, 0 warnings, 0 informations
```

Full requested test file:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py
```

Output:

```text
96 passed in 14.97s
```

Diff hygiene:

```bash
git diff --check -- experiments/vllm_024_dynamicsd/speedbench_dataset.py experiments/vllm_024_dynamicsd/stage_speedbench.sh tests/test_vllm024_dynamicsd.py .superpowers/sdd/task-4-report.md
```

Result:

```text
no output
```

### Remaining concern

The staging path was exercised through rendered-script hostile-value execution
and `TEST_ONLY`, but not through a real cluster submission with upstream
network fetches.
