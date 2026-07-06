# Task 5 RED/GREEN Report

## Scope

Implemented SPEED-Bench official and Sync-RL overlay runners for
`experiments/vllm_024_dynamicsd`.

Files changed:
- Created `experiments/vllm_024_dynamicsd/benchmark_speedbench_sync_rollout.py`
- Created `experiments/vllm_024_dynamicsd/submit_speedbench_k_calibration.sh`
- Created `experiments/vllm_024_dynamicsd/submit_nemotron_speedbench_sync_mtp_matrix.sh`
- Created `experiments/vllm_024_dynamicsd/summarize_speedbench_sync_rollout.py`
- Modified `tests/test_vllm024_dynamicsd.py`
- Updated `.superpowers/sdd/task-5-report.md`

## RED

Command:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py -k speedbench_sync
```

Result:

```text
11 failed, 97 deselected in 0.44s
```

Expected failure reasons:
- Missing `benchmark_speedbench_sync_rollout.py`
- Missing `summarize_speedbench_sync_rollout.py`
- Missing `submit_speedbench_k_calibration.sh`
- Missing `submit_nemotron_speedbench_sync_mtp_matrix.sh`

Representative failure:

```text
AssertionError: missing Task 5 SPEED-Bench overlay runner:
experiments/vllm_024_dynamicsd/benchmark_speedbench_sync_rollout.py
```

## GREEN

Focused Task 5 command:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py -k speedbench_sync
```

Result:

```text
11 passed, 97 deselected in 1.17s
```

Experiment test file:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py
```

Result:

```text
108 passed in 13.36s
```

Shell syntax:

```bash
bash -n \
  experiments/vllm_024_dynamicsd/submit_speedbench_k_calibration.sh \
  experiments/vllm_024_dynamicsd/submit_nemotron_speedbench_sync_mtp_matrix.sh
```

Result: exit 0.

Dry-run launchers:

```bash
DRY_RUN=true CLUSTER=lyris RUN_ID=task5-cal-smoke \
  K_VALUES='1 3' CONCURRENCIES='1 8 32 64' \
  experiments/vllm_024_dynamicsd/submit_speedbench_k_calibration.sh

DRY_RUN=true CLUSTER=ptyche MODELS='ultra super' \
  RUN_ID=task5-nemotron-smoke \
  experiments/vllm_024_dynamicsd/submit_nemotron_speedbench_sync_mtp_matrix.sh
```

Result: both exit 0.

TEST_ONLY launcher stubs:

```text
checked calibration baseline/static generated sbatch and run_benchmark.sh
checked Nemotron baseline/mtp_static/mtp_dynamic generated sbatch and run_benchmark.sh
```

Targeted Pyright:

```bash
pyright \
  experiments/vllm_024_dynamicsd/benchmark_speedbench_sync_rollout.py \
  experiments/vllm_024_dynamicsd/summarize_speedbench_sync_rollout.py \
  tests/test_vllm024_dynamicsd.py
```

Result:

```text
0 errors, 0 warnings, 0 informations
```

Python compile check:

```bash
python3 -m compileall -q \
  experiments/vllm_024_dynamicsd/benchmark_speedbench_sync_rollout.py \
  experiments/vllm_024_dynamicsd/summarize_speedbench_sync_rollout.py
```

Result: exit 0.

Repository-wide suite:

```bash
PYTHONPATH=scripts python3 -m pytest -q
```

Result:

```text
186 passed, 28 subtests passed in 21.17s
```

Plain repository-wide command:

```bash
python3 -m pytest -q
```

Result:

```text
1 error during collection:
ModuleNotFoundError: No module named 'vllm024_dflare_report'
```

Cause: existing `scripts/build_latest_specdec_html_pages.py` imports
`vllm024_dflare_report` as a top-level module; plain pytest does not add
`scripts/` to `PYTHONPATH`. The suite passes with `PYTHONPATH=scripts`.

## Coverage Notes

- Official and overlay rows are separated by explicit `cohort` checks; summary
  comparisons raise on official/overlay mismatch.
- Official SPEED-Bench runs delegate to the staged ModelOpt benchmark command
  instead of rewriting multi-turn prompts locally.
- Overlay prompts preserve prepared token IDs exactly, with no local padding or
  truncation.
- Overlay batches use the Task 4 balanced 48-prompt selection.
- Overlay requests use Task 1 request-plan resolution and exact-work gates.
- Async overlay execution uses `AsyncLLM.generate` streams with an explicit
  `asyncio.gather` barrier and records TTFT and completion timing.
- Acceptance windows are reported by output position with contributor counts.
- Calibration starts with active concurrencies `1 8 32 64`.
- Nemotron baseline, native MTP, and DynamicMTP are emitted as distinct
  supported variants; Eagle remains an unsupported manifest row.
- New launchers render safe generated scripts under `DRY_RUN` and `TEST_ONLY`,
  use `--segment=${NODES}` defaults, avoid `--gres`, and contain no `/home/`
  paths.
