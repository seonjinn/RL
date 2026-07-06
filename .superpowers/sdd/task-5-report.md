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

## Final Instrumentation Fixes RED/GREEN

### Scope

- Added deterministic staging instrumentation for pinned ModelOpt
  `examples/specdec_bench/run.py`; the upstream checkout is copied, exact
  source hash and anchors are verified, and only the staged copy is patched.
- The instrumentation writes explicit Task 5 sidecars:
  `task5_timing_total_tokens.json`,
  `task5_resolved_vllm_config.json`, and `task5_instrumentation.json`.
- Official adaptation now requires those sidecars, sums real
  `Timing.total_tokens` captured from the live metric object, and extracts
  matching/provenance fields from the serialized resolved engine config.
- Removed impossible fixture assumptions that pinned upstream emits raw
  `Timing.total_tokens` or a `vllm_config.to_dict()` result in
  `configuration.json`.
- CLI parsing now resolves sampling defaults by cohort:
  overlay defaults to `temperature=1.0`, `top_p=1.0`; official leaves
  sampling unset unless explicitly supplied so pinned ModelOpt keeps its
  official protocol.

### RED

Focused Task 5 command after adding tests and before implementation:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py -k speedbench_sync
```

Result:

```text
11 failed, 29 passed, 97 deselected in 7.42s
```

Expected failures covered:

- Missing `modelopt_instrumentation_jsonable`, `sha256_text`,
  `stage_instrumented_modelopt_source`, and `parse_speedbench_args`
- Adapter still attempted to read raw `Timing.total_tokens` from
  `timing.json` instead of requiring `task5_timing_total_tokens.json`
- Existing official fixture no longer contained fabricated
  `configuration.serving_config.vllm_config`, so parsing failed before
  sidecar provenance could be used

### GREEN

Focused Task 5 command:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py -k speedbench_sync
```

Result:

```text
40 passed, 97 deselected in 7.53s
```

Experiment test file:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py
```

Result:

```text
137 passed in 21.40s
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

Shell syntax:

```bash
bash -n \
  experiments/vllm_024_dynamicsd/submit_speedbench_k_calibration.sh \
  experiments/vllm_024_dynamicsd/submit_nemotron_speedbench_sync_mtp_matrix.sh
```

Result: exit 0.

Python compile check:

```bash
python3 -m compileall -q \
  experiments/vllm_024_dynamicsd/benchmark_speedbench_sync_rollout.py \
  experiments/vllm_024_dynamicsd/summarize_speedbench_sync_rollout.py
```

Result: exit 0.

Dry-run launchers:

```bash
DRY_RUN=true CLUSTER=lyris RUN_ID=task5-final-instr-cal-smoke \
  K_VALUES='1 3' CONCURRENCIES='1 8 32 64' \
  experiments/vllm_024_dynamicsd/submit_speedbench_k_calibration.sh

DRY_RUN=true CLUSTER=ptyche MODELS='ultra super' \
  RUN_ID=task5-final-instr-nemotron-smoke \
  experiments/vllm_024_dynamicsd/submit_nemotron_speedbench_sync_mtp_matrix.sh
```

Result: both exit 0; rendered scripts show direct runtime digest resolution,
`args+=(--runtime-image-sha256 "${runtime_image_sha256}")`, overlay
`--temperature 1.0`, `--top-p 1.0`, and Ultra Ray environment setup.

Repository-wide suite:

```bash
PYTHONPATH=scripts python3 -m pytest -q
```

Result:

```text
215 passed, 28 subtests passed in 30.02s
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

Cause remains the existing `scripts/build_latest_specdec_html_pages.py`
top-level import path issue; the suite passes with `PYTHONPATH=scripts`.

## Second Review RED/GREEN

Second review items fixed:
- Official rows now parse pinned ModelOpt `timing.json`,
  `acceptance_rate.json`, `specbench_results.json`, and `configuration.json`,
  including list-wrapped metric schemas, and fail closed on missing, zero, or
  invalid metric/config fields.
- Official runs resolve the Task 4 prepared manifest/checksum-backed Parquet
  path and pass that exact path through `--dataset_path`.
- Official dynamic and MTP dynamic modes are rejected because the pinned
  ModelOpt CLI has no schedule support.
- Official provenance now comes from `configuration.json` for recorded fields;
  upstream-unrecorded dtype/compiler/runtime knobs are marked as
  `upstream-unrecorded` instead of synthesized.
- Ultra sbatch now exports the Task 3 Ray environment (`HEAD_NODE`, `HEAD_IP`,
  `RAY_PORT`, `RAY_SYNC_DIR`, `GPUS_PER_NODE`) and cleans the sync dir before
  the single coordinated `run_multinode_ray.sh` launch.
- Both launchers derive `BENCH_RUNTIME_IMAGE_SHA256` in sbatch from an explicit
  value, image `.sha256` sidecar, or `sha256sum`, and never pass
  `unknown`/placeholder values.
- Default SPEED-Bench prepared root now matches Task 4 exactly:
  `${LUSTRE_ROOT}/vllm024-dynamicsd/speedbench/speedbench-487aa718-43fee0cd`.
- Summary match/provenance fields now include the expanded performance and work
  fields from the review.
- Parquet reader failures now preserve the original exception cause with no
  binary JSONL fallback.

### RED

Focused second-review command before implementation:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py -k speedbench_sync
```

Result:

```text
8 failed, 18 passed, 97 deselected in 7.09s
```

Representative expected failures:
- Missing `resolve_prepared_dataset_path`
- Official adapter still expected synthetic `metrics.json`
- Parquet read fallback hid the real Parquet error cause
- Summary did not compare expanded performance/work fields
- Launchers did not use Task 4 prepared paths or sbatch-side runtime digest
- Ultra launcher did not export the Task 3 Ray coordination environment
- Official command did not pass an actual Parquet `--dataset_path`
- Official dynamic/MTP dynamic rejection was missing

Additional fail-closed config RED checks added during final review:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py -k official_rejects_missing_configuration_fields
python3 -m pytest -q tests/test_vllm024_dynamicsd.py -k official_rejects_missing_static_draft_config
```

Both initially failed with `Failed: DID NOT RAISE <class 'ValueError'>`,
confirming official adaptation was still falling back to synthetic config
values for missing upstream fields.

### GREEN

Focused Task 5 second-review command:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py -k speedbench_sync
```

Result:

```text
28 passed, 97 deselected in 3.26s
```

Experiment test file:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py
```

Result:

```text
125 passed in 15.96s
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
DRY_RUN=true CLUSTER=lyris RUN_ID=task5-second-review-cal-smoke \
  K_VALUES='1 3' CONCURRENCIES='1 8 32 64' \
  experiments/vllm_024_dynamicsd/submit_speedbench_k_calibration.sh

DRY_RUN=true CLUSTER=ptyche MODELS='ultra super' \
  RUN_ID=task5-second-review-nemotron-smoke \
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
203 passed, 28 subtests passed in 22.60s
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

## Third Review RED/GREEN

Third review items fixed:
- Generated Task 5 run scripts now resolve `runtime_image_sha256` at execution
  from explicit value, `BENCH_RUNTIME_IMAGE_SHA256`, `.sha256` sidecar, or
  `sha256sum`, then append `args+=(--runtime-image-sha256
  "${runtime_image_sha256}")` directly. The placeholder is no longer routed
  through `emit_arg_pair`.
- Official timing totals now come from per-turn `Timing.total_tokens` values
  and are checked against the timing statistics mean.
- Official config/provenance now reads fields from
  `configuration.serving_config` and resolved `serving_config.vllm_config`;
  match fields reject missing data instead of emitting `upstream-unrecorded`.
- Official `Average_AL` is reported only as mean accepted length. Scalar
  `acceptance_rate` is `null` with an explicit unavailable reason unless a
  valid numerator/denominator metric is added upstream.
- Task 5 Sync-RL/DynamicSD overlay launcher defaults now use
  `temperature=1.0` and `top_p=1.0`; official SPEED-Bench remains labeled as
  `official-modelopt`.

### RED

Focused third-review command before implementation:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py -k speedbench_sync
```

Result:

```text
9 failed, 27 passed, 97 deselected in 6.00s
```

Representative expected failures:
- Official config still emitted `dtype=upstream-unrecorded`
- Official total tokens used `mean * request_count` instead of raw per-turn
  timing totals
- Missing serving/vLLM config match fields did not reject
- `Average_AL > 1` was copied into scalar `acceptance_rate`
- Generated run scripts required `BENCH_RUNTIME_IMAGE_SHA256` and did not
  resolve explicit/sidecar digests themselves
- Task 5 overlay dry-runs still emitted `--temperature 0.0`

### GREEN

Focused Task 5 third-review command:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py -k speedbench_sync
```

Result:

```text
36 passed, 97 deselected in 7.24s
```

Experiment test file:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py
```

Result:

```text
133 passed in 23.01s
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
DRY_RUN=true CLUSTER=lyris RUN_ID=task5-third-review-cal-smoke \
  K_VALUES='1 3' CONCURRENCIES='1 8 32 64' \
  experiments/vllm_024_dynamicsd/submit_speedbench_k_calibration.sh

DRY_RUN=true CLUSTER=ptyche MODELS='ultra super' \
  RUN_ID=task5-third-review-nemotron-smoke \
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
211 passed, 28 subtests passed in 27.56s
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
