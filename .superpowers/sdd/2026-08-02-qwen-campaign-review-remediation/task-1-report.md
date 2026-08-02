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
- Review fix round 2 adds the shared `profile_snapshot.py` reader. Gate
  validation returns the SHA256 of the exact profile bytes it read; the
  submitter forwards that digest to every leaf, and `run_scope.sh` safely
  snapshots the profile itself and rejects any post-validation replacement
  before `SBATCH:` output or contact.
- Review fix round 3 binds an immutable `QWEN_CAMPAIGN_ARM` in every Qwen
  condition leaf and repeats all required campaign-gate checks inside
  `run_scope.sh`. Direct Qwen235 R3 and every Qwen 20-step launch now require
  valid arm identity, a matching profile snapshot, and the relevant evidence.
- Review fix round 4 makes the submitter profile digest optional for direct
  condition launches. When supplied it must be a lowercase SHA256 matching the
  local snapshot; campaign evidence remains mandatory and is independently
  revalidated against that local snapshot.
- The round-4 direct-leaf regression creates a disposable experiment harness,
  proves Qwen235 C succeeds with valid R3 evidence and no submitter digest,
  and proves the same direct leaf fails before `SBATCH:` when evidence is
  absent.
- Round 5 moves the runtime-profile fixtures into each disposable copied
  experiment's trusted `profiles/` directory. It also covers direct Qwen30
  20-step promotion success and absent-evidence failure without a submitter
  digest, supplied digest mismatch rejection, explicit-profile Qwen30 smoke,
  and Qwen235 C rejection when individually valid R3 and promotion gates bind
  different profile snapshots.

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
- Focused launcher checks passed for safe profile rendering and runtime
  attestation command construction; an explicit validated-profile digest
  mismatch exited 2 before `SBATCH:`.
- Direct Qwen235 C and Qwen30 20-step A probes without evidence both exited 2
  with no `SBATCH:` output. The matrix suite remained green (47 passed) and
  the focused launcher profile/runtime subset remained green (2 passed).
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest
  --confcutdir=tests/unit/experiments -q
  tests/unit/experiments/test_matrix_submitters.py
  tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py` reported
  `104 passed`.

SHA256:

- `validate_campaign_gate.py`: `0a10cb1d6c6aa24876ed7b9001d27760b0ab5830cb7fdeb2f804cd13f4f3ca02`
- `profile_snapshot.py`: `ad7727cac5d265476cc79567ae05ad6c0e6d93aec1158e0a66f0daf4eca56ae0`
- `validate_campaign_gate.py` (round 2): `86583569f560eaa88cc82df310dfd99bf83de42c926f10cd5a02659c98c5ac53`
- `validate_campaign_gate.py` (round 3): `8e4b6fff945e9bae42c6e9d4d7b914171d791aa96d3b1286f2039c43752f6703`

Concern:

- The standard unit-test fixture starts an unrelated Ray cluster, so the
  standard invocation was bounded at 20 seconds after Ray startup; the
  isolated suite above completed. Generated
  `tests/unit/unit_results*` artifacts were removed.
