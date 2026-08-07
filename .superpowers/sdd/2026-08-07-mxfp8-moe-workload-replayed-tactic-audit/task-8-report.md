# Task 8 Report: Micro, Generation, and GSM8K Correctness Gates

## Delivered

- Added `validate_micro()` with fail-closed checks for tactic failures,
  nonfinite outputs and metrics, nondeterministic CUDA Graph replay, incomplete
  replay evidence, cosine similarity below `0.999`, and stock-relative MXFP8
  max-error outliers.
- Added a typed `MicroCorrectnessEvidence` contract. `validate_micro()` is the
  single authoritative evidence gate and fails closed when its optional second
  argument is absent; the CLI only deserializes evidence before calling it.
- Micro evidence covers unchanged routing counts, separate FC1
  activated-intermediate and FC2 reduced-output stock comparisons, upstream
  MXFP8 numerical bounds, and balanced/high-skew BF16/Python MoE references.
- Added exact deterministic-generation comparison with matched provenance,
  greedy decoding, prompt hashes, example IDs, and token IDs. Token mismatches
  are reported by example ID.
- Added strict matched GSM8K loading for the unchanged evaluator's
  `results.json` and `per_example.jsonl` artifacts. It requires dataset SHA256
  `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`,
  exactly 1,319 canonical IDs, internally consistent aggregates, and matched
  model revision, tokenizer revision, generation arguments, runtime
  fingerprint, and evaluator settings.
- Added exact two-sided binomial McNemar calculation without SciPy and a
  deterministic 10,000-sample paired bootstrap confidence interval using seed
  `20260807`. Promotion passes only when `p >= 0.05` and the interval includes
  zero.
- Both CLIs return nonzero for malformed evidence and promotion-blocking gate
  results.

## Review Corrections

- Every BF16/Python reference now carries an exact `signature_key`, tactic
  pair, `skew_class`, and `comparison_target="fc2_final"`. Unrelated rows,
  FC1 targets, skew mismatches, and references attached to failed measurements
  block promotion even if other valid representative references are present.
- Generation and GSM8K configs are compared as canonical sorted JSON so JSON
  types remain significant. Numeric decoding fields explicitly reject
  booleans, including `False` versus `0` and `True` versus `1` cases.
- GSM8K aggregate `exact_match` must equal the exact recomputed
  `correct / 1319` float; near-but-not-equal values are rejected.

## TDD Evidence

Initial RED:

```text
PYTHONPATH="$PWD" .venv/bin/python -m pytest -q \
  tests/experiments/test_mxfp8_moe_tactic_correctness.py

ModuleNotFoundError: No module named
'experiments.mxfp8_moe_tactic_audit.compare_gsm8k'
```

Final GREEN:

```text
PYTHONPATH="$PWD" .venv/bin/python -m pytest -q \
  tests/experiments/test_mxfp8_moe_tactic_correctness.py

36 passed, 16 warnings in 6.67s
```

Review follow-up RED evidence included the missing typed-contract import,
programmatic micro calls passing without evidence, unrelated extra reference
rows being ignored, boolean numeric configs comparing equal to integers, and a
near-but-not-equal aggregate accuracy passing the old `math.isclose` check.

The warnings are pre-existing pytest temporary-directory cleanup warnings on
macOS. No Task 8 assertion warned or failed.

## Static Verification

```text
.venv/bin/ruff check \
  experiments/mxfp8_moe_tactic_audit/validate_correctness.py \
  experiments/mxfp8_moe_tactic_audit/compare_gsm8k.py \
  tests/experiments/test_mxfp8_moe_tactic_correctness.py

All checks passed!
```

```text
.venv/bin/pyright --pythonpath .venv/bin/python \
  experiments/mxfp8_moe_tactic_audit/validate_correctness.py \
  experiments/mxfp8_moe_tactic_audit/compare_gsm8k.py

0 errors, 0 warnings, 0 informations
```

## Deferred Execution

No live stock/candidate 1,319-example GSM8K evaluation was run in Task 8.
Task 11 owns the external evaluation execution and will feed its unchanged
evaluator artifacts into these gates.
