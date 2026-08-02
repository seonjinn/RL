# Task 1 gate remediation report

Status: green locally; no scheduler contact was made.

Implementation:

- `validate_campaign_gate.py` accepts only absolute regular non-symlink JSON
  gates with caller-supplied lowercase SHA256, exact schemas, exact expected
  provenance, and an attestation SHA256 recomputed from the selected runtime
  attestation file.
- The Qwen matrix defaults to smoke A/B/C/E for Qwen30, smoke A/B for
  Qwen235, and performance A/B for both models. It validates both dry-run
  controls before resolving leaves, requires the Qwen235 C/E R3 preflight,
  and requires promotion evidence for every performance request.
- Review fix round 1 removes profile sourcing entirely. The validator now
  parses only a fixed allowlist of literal profile assignments from a direct
  child of the real submitter's trusted `profiles/` directory. Gate, profile,
  and runtime bytes are each opened once with no-follow semantics and checked
  with `fstat`; hashes and parsers consume those same bytes. JSON duplicate
  keys and fractional job IDs are rejected.

Evidence:

- RED observed: `test_qwen30_performance_defaults_to_a_and_b` failed because
  the former default requested C/E and the A/B promotion gate correctly did
  not cover C.
- GREEN: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest
  --confcutdir=tests/unit/experiments
  tests/unit/experiments/test_matrix_submitters.py -q` reported `47 passed`.
- `bash -n` on the submitter, `.venv/bin/python -m py_compile` on the
  validator, and `git diff --check` exited zero.
- Real `TEST_ONLY=1` smoke render counts: Qwen30 = 4 matrix rows; Qwen235 =
  2 matrix rows (A/B only). Missing R3/promotion evidence exited 2 before any
  leaf/scheduler output.
- The added RED tests reproduced acceptance of an outside profile; green tests
  now reject outside, symlinked, and executable profile payloads without
  creating the scheduler marker. They also reject invalid clusters before path
  resolution, file swaps after descriptor open, duplicate JSON keys, and float
  job IDs.

SHA256:

- `validate_campaign_gate.py`: `0a10cb1d6c6aa24876ed7b9001d27760b0ab5830cb7fdeb2f804cd13f4f3ca02`

Concern:

- The standard unit-test fixture starts an unrelated Ray cluster, so the
  standard invocation was bounded at 20 seconds after Ray startup; the
  isolated suite above completed. Generated
  `tests/unit/unit_results*` artifacts were removed.
