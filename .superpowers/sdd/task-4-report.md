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
