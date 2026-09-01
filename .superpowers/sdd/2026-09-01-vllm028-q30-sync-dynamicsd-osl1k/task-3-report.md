# Task 3 report: one-engine execution and strict result validation

## Status

Complete. The Q30 package now provides deterministic prompt sealing and
external-worker request partitioning, an engine-independent local generation
boundary, concrete completion and finish timing, immutable result/provenance
records, strict typed and JSON validation, and atomic no-clobber publication.
The implementation does not import vLLM, select schedules, render SLURM, or
modify the approved Task 2 interfaces.

## Commits

- `6141b7eba` — `exp: implement Q30 one-engine result gate` (signed off)
- `0030cbbd5` — `docs: report Q30 one-engine result task` (signed off)
- `2f72560b2` — `fix: separate Q30 verifier and drafter evidence` (signed off)
- This report update is committed separately as the review-remediation
  documentation commit.

## TDD evidence

### RED: concrete fake-engine runner

Command:

```text
python3 -m pytest -q tests/test_vllm028_q30_sync_dynamicsd.py -k one_engine_runner
```

Result: exit 2 during collection with
`ModuleNotFoundError: No module named
'experiments.vllm_028_q30_sync_dynamicsd.benchmark'`. This was the intended
failure before the runner existed.

### GREEN: literal completion and barrier/token summary

The same command exited 0 with `1 passed, 17 deselected in 0.05s`. The fake
engine returned literal three-token and two-token completions; the result
reported two requests, five output tokens, 2.5 elapsed/barrier seconds,
2.0 output tokens/s, and per-request finish times of 1.0 and 2.5 seconds.
The requests carried seeds `20260901` and `20260902` plus `ignore_eos=False`.

### RED: strict worker gates and no-clobber publication

Command:

```text
python3 -m pytest -q tests/test_vllm028_q30_sync_dynamicsd.py -k 'worker_result_validation or k0_diagnostic or runtime_provenance or publication'
```

Result: exit 2 during collection with `ImportError: cannot import name
'publish_worker_result'`. This was the intended missing-validator/publication
failure.

### GREEN: strict worker gates and K0 evidence

The same command exited 0 with `18 passed, 18 deselected in 0.08s` after
implementing validation for OSL1K, DP1, FULL_AND_PIECEWISE graph evidence,
runtime identities, complete exact request work, summary arithmetic, immutable
provenance, and atomic no-clobber JSON. This initial implementation was then
superseded by the structured trace and width-semantics remediation documented
below.

### RED/GREEN: untrusted JSON and sealed prompt partitions

Command:

```text
python3 -m pytest -q tests/test_vllm028_q30_sync_dynamicsd.py -k 'json_result_boundary or prompt_manifest or barrier_prompt_partition or incomplete_engine_completion'
```

RED result: exit 2 during collection with `ImportError: cannot import name
'validate_result_payload'`. GREEN result: exit 0 with
`4 passed, 36 deselected in 0.04s`. JSON fields and types are reconstructed
strictly; incomplete JSON rows, modified prompt seals, incomplete manifests,
and incomplete engine completion sets are rejected. Worker 0 receives global
indices 0 through 127 and worker 15 receives 1920 through 2047 with the
approved global-index seed policy.

### RED/GREEN: clean unknown completion rejection

Command:

```text
python3 -m pytest -q tests/test_vllm028_q30_sync_dynamicsd.py -k unknown_engine_completion
```

RED result: exit 1 with `1 failed, 40 deselected in 0.07s`; an unknown engine
request ID leaked a `KeyError`. GREEN result: exit 0 with
`1 passed, 40 deselected in 0.04s`; the runner now rejects a non-exact
completion identity set with a contract `ValueError` before row construction.

### RED/GREEN: fixed-K selected-K evidence

Command:

```text
python3 -m pytest -q tests/test_vllm028_q30_sync_dynamicsd.py -k wrong_selected_k_histogram
```

RED result: exit 1 with `1 failed, 41 deselected in 0.08s` because a fixed-K2
result with a K0 histogram was accepted. GREEN result: exit 0 with
`1 passed, 41 deselected in 0.03s`; fixed rows now require both the selected
verifier K and selected-K histogram to match the plan.

## Review remediation

The review fixes preserve five independent concepts: Task 2 checkpoint block
size, configured drafter K, trace-observed physical query width,
trace-observed physical output width, and selected verifier K/histogram. The
exact s4166 capability rules are now method-aware: DFlash supports configured
K through 7 with query width `K + 1` and output width `K`; DSpark supports
configured K through 8 with query and output width `K`. Thus DFlash K7/query8/
output7 and DSpark K8/query8/output8 pass, while DFlash K8/query9 is rejected.

K0 DynamicSD keeps selected verifier K at zero while independently allowing a
positive configured/static drafter K and corresponding physical execution.
Baseline rows instead require all speculative counters to be zero, an empty
selected-K histogram, and no selected-K, configured-K, or trace fields.

`DrafterTraceEvidence` is a frozen record tied to a reproducible profiler
artifact. It requires a supported real trace source, artifact URI/path,
lowercase SHA256, positive byte size, consistent monotonic capture interval
and duration, kernel count/time, and observed query/output widths. Drafter
execution or absence derives only from this record; counter-only evidence and
absence claims without trace provenance fail validation. JSON parsing and
serialization preserve the immutable record and reject derived-field
tampering.

Selected-K validation now requires positive observations: fixed methods have
the exact configured key and iteration count, while DynamicSD has a nonempty
positive histogram bounded by configured and checkpoint capabilities with
meaningful iteration consistency. Completion `finish_seconds` values are
relative to run start and must be finite, nonnegative, nondecreasing, and no
greater than total elapsed time.

### RED: structured evidence and corrected width/timing contract

Command:

```text
python3 -m pytest -q tests/test_vllm028_q30_sync_dynamicsd.py -k 's4166 or baseline_requires or selected_k_histogram or trace_evidence or k0_diagnostic or finish_durations'
```

Result: exit 2 during collection with `ImportError: cannot import name
'DrafterTraceEvidence'` from `results.py`. This was the intended failure before
the reviewed contract existed.

An additional strict-type RED command,
`python3 -m pytest -q tests/test_vllm028_q30_sync_dynamicsd.py -k
trace_evidence`, initially exited 1 with `7 passed, 1 failed, 54 deselected`:
the Boolean value `True` was incorrectly accepted as draft kernel time.

### GREEN: review contract

The strict trace command then exited 0 with `8 passed, 54 deselected in
0.05s`. After resolving one transient full-suite failure by ordering the K0
completeness check before capability validation, the complete focused suite
passed as recorded below.

## Final verification

- `python3 -m pytest -q tests/test_vllm028_q30_sync_dynamicsd.py` — exit 0,
  `63 passed in 0.09s`.
- `python3 -m pytest -q tests/test_vllm028_nemotron_bf16_matrix.py tests/test_vllm028_mrv2_patch_canary.py tests/test_vllm028_mrv2_patch_matrix.py`
  — exit 0, `88 passed in 2.32s`.
- `ruff check experiments/vllm_028_q30_sync_dynamicsd tests/test_vllm028_q30_sync_dynamicsd.py`
  — exit 0, `All checks passed!`.
- `pyright experiments/vllm_028_q30_sync_dynamicsd tests/test_vllm028_q30_sync_dynamicsd.py`
  — exit 0, `0 errors, 0 warnings, 0 informations`.
- `git diff --cached --check` before the implementation commit — exit 0.
- `rg -n "configured_draft_width|physical_draft_width|finished_at_seconds|drafter_execution_evidence_source|drafter_execution_count" experiments/vllm_028_q30_sync_dynamicsd tests/test_vllm028_q30_sync_dynamicsd.py`
  — exit 1 with no matches, confirming removal of the superseded fields.

An earlier combined Ruff/Pyright check reported seven Pyright errors in the
new protocol and untrusted-object parsing. The protocol stub and strict typed
parsers were corrected before the final clean check above.

## Files changed

- `experiments/vllm_028_q30_sync_dynamicsd/benchmark.py`
- `experiments/vllm_028_q30_sync_dynamicsd/results.py`
- `tests/test_vllm028_q30_sync_dynamicsd.py`
- `.superpowers/sdd/2026-09-01-vllm028-q30-sync-dynamicsd-osl1k/task-3-report.md`

## Concerns

- Both passing pytest suites emitted existing, non-failing temporary-directory
  cleanup warnings (`OSError: [Errno 66] Directory not empty`) from the shared
  staging-test environment.
- Drafter execution is intentionally adapter-supplied structured trace
  evidence; no kernel absence is inferred from proposed/accepted-token
  counters, and missing trace provenance cannot establish absence.
- Schedule selection and SLURM rendering remain unimplemented for Tasks 4 and
  5, as required by the Task 3 boundary.
- The no-subagent instruction prevented an independent reviewer dispatch; a
  local staged-diff review and the full validation gates were used instead.
