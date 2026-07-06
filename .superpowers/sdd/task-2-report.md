# Task 2 Report: SWE Barrier and Accuracy Runner

## Scope

Modified:

- `experiments/vllm_024_dynamicsd/benchmark_sync_rollout.py`
- `experiments/vllm_024_dynamicsd/submit_sync_rollout.sh`
- `experiments/vllm_024_dynamicsd/summarize_sync_rollout.py`
- `tests/test_vllm024_dynamicsd.py`

Created:

- `experiments/vllm_024_dynamicsd/submit_swe_sync_rollout_matrix.sh`

## Requirements Source

Command:

```bash
sed -n '1,260p' .superpowers/sdd/task-2-brief.md
```

Key requirements read before edits:

- add focused failing tests first
- reject prompt truncation for accuracy runs
- use Task 1 `RequestPlan` and `ResolvedRequest`
- apply per-request `max_tokens`, `min_tokens`, `ignore_eos`, and seed values
- produce per-request provenance, `resolved_request_plan.json`, optional response JSONL, and per-bucket statistics
- add strict summary matching on runtime image, model view hash, prompt hash, request-plan hash, graph mode, topology, sampling, and exact output work
- create `submit_swe_sync_rollout_matrix.sh`
- validate with focused/full tests and launcher syntax checks
- commit only Task 2 files with `-s`

## RED

Test-only patch added focused tests for:

- `tokenize_prompt(..., allow_truncation=False)` rejecting long prompts
- request-plan-controlled per-request sampling params and provenance
- response JSONL persistence and bucket stats
- SWE wrapper launcher rendering request plans and response outputs
- domain-neutral `SMOKE=false` prompt validation
- strict request-plan summary match
- exact output-work summary match

Command:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py -k 'truncation or response_output or swe_sync or request_plan_controls or bucket_stats or request_plan_hash or exact_output_work or domain_neutral'
```

Output:

```text
..FFFFFFF                                                                [100%]
FAILED tests/test_vllm024_dynamicsd.py::test_tokenize_prompt_rejects_truncation_when_disabled
FAILED tests/test_vllm024_dynamicsd.py::test_sync_rollout_request_plan_controls_sampling_and_provenance
FAILED tests/test_vllm024_dynamicsd.py::test_sync_rollout_response_jsonl_and_bucket_stats_preserve_provenance
FAILED tests/test_vllm024_dynamicsd.py::test_swe_sync_rollout_matrix_renders_request_plan_and_response_outputs
FAILED tests/test_vllm024_dynamicsd.py::test_sync_rollout_smoke_false_prompt_requirement_is_domain_neutral
FAILED tests/test_vllm024_dynamicsd.py::test_sync_rollout_summary_rejects_mismatched_request_plan_hash
FAILED tests/test_vllm024_dynamicsd.py::test_sync_rollout_summary_rejects_mismatched_exact_output_work
7 failed, 2 passed, 53 deselected in 0.28s
```

The failures were expected: missing new keyword/API types, missing SWE launcher, old math-specific validation text, and missing summary checks.

## Implementation

`benchmark_sync_rollout.py`:

- added `PromptRecord` and `RolloutRequest`
- made prompt truncation explicit with `allow_truncation`
- changed actual prompt loading and warmup tokenization to reject truncation
- loaded optional `--request-plan`
- validated `--max-model-len` against the plan
- used `resolve_request_plan()` to build measured rollout requests
- built `SamplingParams` per request with plan `max_tokens`, `min_tokens`, `ignore_eos`, and seed
- recorded `model_config_hash`, `runtime_image_sha256`, `prompt_set_hash`, `request_plan_hash`, request provenance, and per-bucket stats
- wrote `resolved_request_plan.json` for plan runs
- wrote optional response JSONL with prompt/sample/seed/cap provenance and output token hash

`submit_sync_rollout.sh`:

- added `REQUEST_PLAN`, `REQUEST_PLAN_IN_CONTAINER`, `RESOLVED_REQUEST_PLAN_OUTPUT`, `RESPONSE_OUTPUT`, and `RUNTIME_IMAGE_SHA256`
- translated project-local request plans to `/workspace/experiment/...` for container execution
- validated request-plan files before submission
- passed runtime image hash and plan/response args into the benchmark
- replaced the math-specific `SMOKE=false` prompt error with domain-neutral prompt-set wording

`submit_swe_sync_rollout_matrix.sh`:

- added a SWE-specific wrapper around `submit_sync_rollout.sh`
- supports `qwen30ba3b`, `qwen32`, and `qwen235b`
- supports `REQUEST_PROFILES=32k 64k`
- uses SWE-Bench verified prompts by default
- sets request-plan, max-model-len, max-new-token cap, response JSONL, and resolved-plan output per variant

`summarize_sync_rollout.py`:

- added strict match keys for runtime image SHA, model config hash, prompt set hash, and request-plan hash
- rejects summaries when output-token hashes differ from baseline when hashes are present

## Debugging Note

The first GREEN attempt failed during import on Python 3.14 because the test helper loads `benchmark_sync_rollout.py` with `importlib.util.module_from_spec()` without inserting it into `sys.modules`; dataclasses with future string annotations inspect `sys.modules` during class processing.

Fix:

- changed the two tiny record classes from `@dataclass` to `NamedTuple`
- preserved the same attribute and keyword-construction API used by tests

Focused rerun after that fix:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py -k 'truncation or response_output or swe_sync or request_plan_controls or bucket_stats or request_plan_hash or exact_output_work or domain_neutral'
```

Output:

```text
.........                                                                [100%]
9 passed, 53 deselected in 0.23s
```

## Verification

Exact focused selector from the brief:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py -k 'truncation or response_output or swe_sync'
```

Output:

```text
...                                                                      [100%]
3 passed, 59 deselected in 0.19s
```

Focused/full Task 2 test file:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py
```

Output:

```text
..............................................................           [100%]
62 passed in 6.29s
```

Launcher syntax validation:

```bash
bash -n experiments/vllm_024_dynamicsd/submit_sync_rollout.sh experiments/vllm_024_dynamicsd/submit_swe_sync_rollout_matrix.sh
```

Output:

```text
[no output]
```

Exit status: `0`

Repository-wide test attempt without extra path:

```bash
python3 -m pytest -q
```

Output:

```text
ERROR tests/test_build_latest_specdec_html_pages.py
ModuleNotFoundError: No module named 'vllm024_dflare_report'
1 error in 6.40s
```

Root cause checked:

- `scripts/build_latest_specdec_html_pages.py` imports `vllm024_dflare_report` as a top-level module
- the file exists at `scripts/vllm024_dflare_report.py`
- plain repo pytest does not put `scripts/` on `PYTHONPATH`

Repository-wide test with expected script import path:

```bash
PYTHONPATH=scripts python3 -m pytest -q
```

Output:

```text
............................................ [ 31%]
........................................................................ [ 82%]
........................                                                 [100%]
140 passed, 28 subtests passed in 16.94s
```

Whitespace check:

```bash
git diff --check
```

Output:

```text
[no output]
```

Exit status: `0`

## Self-Review

Checklist:

- Task 2 files only: yes
- unrelated changes preserved: yes
- tests written before implementation: yes, RED recorded
- request plan uses Task 1 interfaces: yes, via `load_request_plan()` and `resolve_request_plan()`
- no prompt truncation for loaded benchmark prompts: yes
- per-request caps and seeds: yes
- resolved plan output: yes
- optional response JSONL: yes
- per-bucket stats: yes
- strict summary provenance keys: yes
- exact output work checked when output hashes are present: yes
- launcher syntax checked: yes
- full pytest caveat documented: yes

Concern:

- Plain `python3 -m pytest -q` still needs the existing `PYTHONPATH=scripts` setup for unrelated report tests; with that path, the full suite passes.

## Review Fixes

### Review Requirements

Addressed reviewer/controller findings:

- 64K SWE runs materialize matched YaRN target and Eagle3 drafter views for every Qwen model with `materialize_long_context_model_views.py --max-position-embeddings 131072 --rope-factor 4.0`; 32K remains native.
- Smoke SWE runs now exercise the request-plan tail with 16 prompt assignments.
- Non-smoke SWE defaults are now the primary `16 prompts x 4 samples x 3 barriers`; `FULL_CONTRACT=true` or explicit `SAMPLES_PER_PROMPT=16` is required for 16 samples.
- Prompt provenance now hashes the actual tokenized prompt and preserves any source-provided `prompt_sha256` as `source_prompt_sha256`.
- Forced exact work emits ordered `planned_output_tokens` and `actual_output_tokens` and validates `ignore_eos=true` requests fill to `max_tokens`.
- Summary exact-work matching compares planned/actual token counts, not output hashes. Hashes remain a separate reported signal.
- Explicit `RESPONSE_OUTPUT` and `RESOLVED_REQUEST_PLAN_OUTPUT` paths are rejected for multi-variant runs unless set to `auto` or containing `{variant}`.
- New launcher arguments use escaped exports plus nested `bench_extra_args` arrays instead of hand-built single-quoted fragments.

### Review RED

Added focused failing tests before implementation:

- tokenized prompt hash vs source prompt hash preservation
- exact planned/actual token count emission and forced-output underfill rejection
- 64K YaRN target/draft materialization dry-run behavior
- smoke tail coverage and non-smoke/full-contract defaults
- explicit multi-variant output path rejection and `{variant}` expansion
- shell escaping for new output arguments containing single quotes
- summary acceptance of different temperature-1 hashes with equal exact work
- summary rejection of identical hashes with under-length exact work

Command:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py -k 'tokenized_prompt or exact_output_work or 64k_uses_matched_yarn or non_smoke_defaults or full_contract or shared_explicit_outputs or shell_escapes or different_hashes or underlength or request_plan_controls or response_jsonl or swe_sync_rollout_matrix'
```

Output:

```text
FFFFFFFFFFFF                                                             [100%]
FAILED tests/test_vllm024_dynamicsd.py::test_sync_rollout_request_plan_controls_sampling_and_provenance
FAILED tests/test_vllm024_dynamicsd.py::test_sync_rollout_hashes_actual_tokenized_prompt_and_preserves_source_hash
FAILED tests/test_vllm024_dynamicsd.py::test_sync_rollout_response_jsonl_and_bucket_stats_preserve_provenance
FAILED tests/test_vllm024_dynamicsd.py::test_sync_rollout_exact_output_work_emits_counts_and_rejects_underfill
FAILED tests/test_vllm024_dynamicsd.py::test_swe_sync_rollout_matrix_renders_request_plan_and_response_outputs
FAILED tests/test_vllm024_dynamicsd.py::test_swe_sync_rollout_64k_uses_matched_yarn_target_and_draft_views
FAILED tests/test_vllm024_dynamicsd.py::test_swe_sync_rollout_non_smoke_defaults_to_primary_four_samples
FAILED tests/test_vllm024_dynamicsd.py::test_swe_sync_rollout_full_contract_override_uses_sixteen_samples
FAILED tests/test_vllm024_dynamicsd.py::test_sync_rollout_rejects_shared_explicit_outputs_for_multi_variant_runs
FAILED tests/test_vllm024_dynamicsd.py::test_sync_rollout_accepts_variant_placeholders_and_shell_escapes_new_args
FAILED tests/test_vllm024_dynamicsd.py::test_sync_rollout_summary_allows_different_hashes_with_equal_exact_work
FAILED tests/test_vllm024_dynamicsd.py::test_sync_rollout_summary_rejects_identical_hashes_with_underlength_work
12 failed, 58 deselected in 1.10s
```

### Review GREEN

Focused review-fix tests:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py -k 'tokenized_prompt or exact_output_work or 64k_uses_matched_yarn or non_smoke_defaults or full_contract or shared_explicit_outputs or shell_escapes or different_hashes or underlength or request_plan_controls or response_jsonl or swe_sync_rollout_matrix'
```

Output:

```text
............                                                             [100%]
12 passed, 58 deselected in 0.79s
```

Full Task 2 test file:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py
```

Output:

```text
......................................................................   [100%]
70 passed in 4.46s
```

Shell syntax:

```bash
bash -n experiments/vllm_024_dynamicsd/submit_sync_rollout.sh experiments/vllm_024_dynamicsd/submit_swe_sync_rollout_matrix.sh
```

Output:

```text
[no output]
```

Exit status: `0`

Full suite with existing script import path:

```bash
PYTHONPATH=scripts python3 -m pytest -q
```

Output:

```text
............................................ [ 29%]
........................................................................ [ 78%]
................................                                         [100%]
148 passed, 28 subtests passed in 13.94s
```

Pyright, repository-wide:

```bash
pyright
```

Output summary:

```text
1244 errors, 7 warnings, 0 informations
```

The failures are pre-existing/unrelated and include older experiment scripts plus missing optional dependencies under `remote_worktree_edit`.

Pyright, touched Python files:

```bash
pyright experiments/vllm_024_dynamicsd/benchmark_sync_rollout.py experiments/vllm_024_dynamicsd/summarize_sync_rollout.py tests/test_vllm024_dynamicsd.py
```

Output:

```text
0 errors, 0 warnings, 0 informations
```

## Task 2 Follow-up Review Fixes

### RED

Added focused failing tests before implementation for:

- executable per-variant `run_benchmark.sh` generation and execution with a stub Python benchmark path
- hostile model/draft/request/output paths containing apostrophe, double quote, spaces, dollar sign, and literal command substitution without evaluation
- non-mutating `DRY_RUN` and `TEST_ONLY` planning that skips YaRN materialization while still rendering planned view and runner paths
- forced-request mask emission, forced-only exactness, forced planned-count comparison, and unforced normal-EOS underfill allowance

Command:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py -k 'exact_output_work or generated_runner or plan_modes or unforced_underfill or forced_planned_work_mismatch or identical_hashes_with_underlength_work'
```

Output:

```text
FFFFFF.                                                                  [100%]
FAILED tests/test_vllm024_dynamicsd.py::test_sync_rollout_exact_output_work_emits_counts_and_rejects_underfill
FAILED tests/test_vllm024_dynamicsd.py::test_sync_rollout_generated_runner_executes_with_hostile_paths
FAILED tests/test_vllm024_dynamicsd.py::test_swe_sync_rollout_plan_modes_do_not_materialize_yarn_or_run_dirs[true-false-[DRY-RUN]]
FAILED tests/test_vllm024_dynamicsd.py::test_swe_sync_rollout_plan_modes_do_not_materialize_yarn_or_run_dirs[false-true-[TEST-ONLY]]
FAILED tests/test_vllm024_dynamicsd.py::test_sync_rollout_summary_allows_unforced_underfill_with_matching_forced_work
FAILED tests/test_vllm024_dynamicsd.py::test_sync_rollout_summary_rejects_forced_planned_work_mismatch
6 failed, 1 passed, 68 deselected in 0.72s
```

### GREEN

Focused reviewer tests:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py -k 'exact_output_work or generated_runner or plan_modes or unforced_underfill or forced_planned_work_mismatch or identical_hashes_with_underlength_work'
```

Output:

```text
.......                                                                  [100%]
7 passed, 68 deselected in 1.21s
```

Full Task 2 test file:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py
```

Output:

```text
........................................................................ [ 96%]
...                                                                      [100%]
75 passed in 7.61s
```

Full suite with script import path:

```bash
PYTHONPATH=scripts python3 -m pytest -q
```

Output:

```text
............................................ [ 28%]
........................................................................ [ 75%]
.....................................                                    [100%]
153 passed, 28 subtests passed in 14.00s
```

Shell syntax validation:

```bash
bash -n experiments/vllm_024_dynamicsd/submit_sync_rollout.sh experiments/vllm_024_dynamicsd/submit_swe_sync_rollout_matrix.sh
```

Output:

```text
[no output]
```

Exit status: `0`

Targeted Pyright:

```bash
pyright experiments/vllm_024_dynamicsd/benchmark_sync_rollout.py experiments/vllm_024_dynamicsd/summarize_sync_rollout.py tests/test_vllm024_dynamicsd.py
```

Output:

```text
0 errors, 0 warnings, 0 informations
```

### Self-review Notes

- Replaced the nested `bash -lc` benchmark argument construction with a per-variant executable `run_benchmark.sh`; all rendered benchmark argument values are emitted once through `printf %q` into a real Bash array.
- `DRY_RUN` and `TEST_ONLY` now render planned run scripts and sbatch files without creating run directories, manifests, or YaRN model views.
- Summary validation now uses `forced_output_mask`: forced entries must match planned lengths, forced planned counts are compared across variants, unforced underfill is allowed, and output token hashes remain a separate signal.
