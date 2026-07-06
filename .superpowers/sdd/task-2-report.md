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
