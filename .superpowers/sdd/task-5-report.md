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

## Review Fix RED/GREEN

Review items fixed:
- Official runner now executes pinned ModelOpt
  `examples/specdec_bench/run.py` with the upstream CLI and adapts its
  `metrics.json` into official rows.
- Overlay prompts are prepared from Task 4 manifest/checksum-backed Parquet,
  tokenized once with the target tokenizer, and expanded from exactly 48 unique
  balanced prompts independent of active concurrency.
- Overlay barrier batches cycle deterministic aliases for repeated work, keep
  original prompt IDs in provenance, and enforce request-plan exact work.
- Nemotron SPEED-Bench launcher reuses the coordinated Ray contract for Ultra
  and emits the matrix MTP/Mamba/model-loader/fuse settings for Ultra/Super.
- Acceptance reporting now separates vLLM draft-position acceptance counters
  from completion-position contributor-count windows and documents the
  limitation.
- Runner and summarizer require strict provenance, runtime image SHA values,
  active non-baseline SpecDec counters, and latency tail aggregates.
- Overlay `AsyncEngineArgs` now receives the PIECEWISE `compilation_config`.

### RED

Command:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py -k speedbench_sync
```

Result before review-fix implementation:

```text
11 failed, 9 passed, 97 deselected in 3.29s
```

Representative expected failures:
- Missing `run_official_speedbench`
- Missing manifest-backed `build_overlay_from_prepared_parquet`
- Missing `expand_overlay_barrier_batches`
- Parser rejected `--prepared-manifest` and `--prepared-checksums`
- Summary accepted missing/unknown provenance
- Launchers still rendered `--prepared-jsonl`/`overlay_prompts.jsonl`
- Nemotron launcher did not render `run_multinode_ray.sh` Ray coordination

### GREEN

Focused Task 5 review command:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py -k speedbench_sync
```

Result:

```text
21 passed, 97 deselected in 2.90s
```

Experiment test file:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py
```

Result:

```text
118 passed in 15.77s
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
DRY_RUN=true CLUSTER=lyris RUN_ID=task5-review-cal-smoke \
  K_VALUES='1 3' CONCURRENCIES='1 8 32 64' \
  experiments/vllm_024_dynamicsd/submit_speedbench_k_calibration.sh

DRY_RUN=true CLUSTER=ptyche MODELS='ultra super' \
  RUN_ID=task5-review-nemotron-smoke \
  experiments/vllm_024_dynamicsd/submit_nemotron_speedbench_sync_mtp_matrix.sh
```

Result: both exit 0.

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

Whitespace check:

```bash
git diff --check
```

Result: exit 0.

Repository-wide suite:

```bash
PYTHONPATH=scripts python3 -m pytest -q
```

Result:

```text
196 passed, 28 subtests passed in 23.11s
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
`scripts/` to `PYTHONPATH`. The repository-wide suite passes with
`PYTHONPATH=scripts`.
