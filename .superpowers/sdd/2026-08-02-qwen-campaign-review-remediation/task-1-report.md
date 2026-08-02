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

Evidence:

- RED observed: `test_qwen30_performance_defaults_to_a_and_b` failed because
  the former default requested C/E and the A/B promotion gate correctly did
  not cover C.
- GREEN: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest
  --confcutdir=tests/unit/experiments
  tests/unit/experiments/test_matrix_submitters.py -q` reported `38 passed`.
- `bash -n` on the submitter, `.venv/bin/python -m py_compile` on the
  validator, and `git diff --check` exited zero.
- Real `TEST_ONLY=1` smoke render counts: Qwen30 = 4 matrix rows; Qwen235 =
  2 matrix rows (A/B only). Missing R3/promotion evidence exited 2 before any
  leaf/scheduler output.

SHA256:

- `validate_campaign_gate.py`: `59dd83756915fdd93e53608f18bf16ea429cbafd3957116e4ebe8b4f01ea9dcf`

Concern:

- The standard unit-test fixture starts an unrelated Ray cluster, so the
  bounded isolated test invocation above was used; generated
  `tests/unit/unit_results*` artifacts were removed.
