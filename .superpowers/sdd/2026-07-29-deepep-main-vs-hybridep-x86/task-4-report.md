# Task 4 report: standard DeepEP launcher and x86 profiles

## Scope

Implemented the standard DeepEP arm in the OCI-HSG GRPO launcher, added its
three x86 model profiles, and documented the immutable DeepEP/NCCL artifact
workflow. The implementation preserves the Task 3 renderer interface and does
not modify `nemo_rl/utils/venvs.py`.

## RED

Added launcher behavior tests before changing production code for:

- standard DeepEP flex dispatch (`deepep`, 20 SMs, no HybridEP SM override);
- invalid DeepEP variant;
- branch/variant mismatch;
- missing NCCL wheel;
- NCCL version mismatch;
- NCCL checksum mismatch;
- DeepEP metadata commit mismatch; and
- renderer inputs plus the persistent Ray-parent NCCL loader path.

The test-only RED commit is
`a2e7b05c796b655bb2389a434b7a658bf9e79d37`
(`test(hybridep): cover standard deepep launcher`). It contains no launcher,
profile, or documentation production changes.

The initial executable mode-only test failed as intended because the launcher
rejected `DISPATCHER_MODE=deepep` with:

```text
DISPATCHER_MODE must be either hybridep or recipe.
```

The artifact-backed cases require a writable `/lustre` fixture and are skipped
on this local macOS machine because `/lustre` is not mounted. They remain
executable on the target environment and validate the real subprocess rather
than mocked launcher behavior. Controller-run remote RED evidence is pending.

## GREEN

The production GREEN commit is this task's
`feat(hybridep): add standard deepep x86 profiles` commit, following the
test-only RED commit above.

The launcher now:

- accepts `recipe | deepep | hybridep`;
- forces a standard DeepEP run to use immutable DeepEP and NCCL artifacts;
- validates `DEEPEP_VARIANT`, selected branch, commit, both canonical wheel
  paths, both checksums, NCCL package/version `2.30.4`, and cross-artifact
  metadata before scheduling;
- passes `DEEPEP_VARIANT`, `NCCL_WHEEL`, and `NCCL_WHEEL_SHA256` with the
  existing DeepEP wheel inputs to `render_deepep_setup_command.sh`;
- prepends `${DEEPEP_OVERLAY}/nvidia/nccl/lib` to the Ray-parent
  `LD_LIBRARY_PATH` before `sbatch` launches Ray; and
- records the variant, branch, both wheel paths/hashes, NCCL version, resolved
  config hash, and runtime-probe field names in `submission.env`.

New profiles use their all-to-all recipes unchanged and pin:

```text
DISPATCHER_MODE=deepep
DEEPEP_VARIANT=deepep
DEFAULT_DEEPEP_COMMIT=dd758caf451848bd150e1046af3d0a73e5fff38d
REQUIRE_DEEPEP_WHEEL=true
REQUIRE_NCCL_WHEEL=true
```

## Verification

```text
bash -n scripts/experiments/oci-hsg/hybridep/submit_grpo.sh
bash -n scripts/experiments/oci-hsg/hybridep/models/*-x86-deepep.env
# exit 0

/private/tmp/nemo-rl-hybridep-x86-tests-20260728/bin/python -m pytest -q \
  tests/unit/tools/test_hybridep_submit_grpo.py \
  tests/unit/tools/test_hybridep_x86_contract.py
# blocked at collection: ModuleNotFoundError: No module named 'ray'

/private/tmp/nemo-rl-hybridep-x86-tests-20260728/bin/python -m pytest -q \
  --confcutdir=tests/unit/tools \
  tests/unit/tools/test_hybridep_submit_grpo.py \
  tests/unit/tools/test_hybridep_x86_contract.py
# 58 passed, 13 skipped in 14.18s

git diff --check
# exit 0

git diff 4c14b04266a0b3ed8ec6121fae387d77d869bf1d -- nemo_rl/utils/venvs.py
# no output
```

The 13 skips all require a writable `/lustre`; this local environment has no
`/lustre` mount. The skipped set includes the end-to-end standard-artifact
subprocess cases. Existing isolated renderer and x86 contract tests pass, but
controller-run remote RED/GREEN evidence for the `/lustre` cases remains
pending.

## Remote fixture follow-up

The first approved remote runs, jobs `480170` (RED) and `480171` (GREEN),
completed but skipped all eight new artifact-backed tests. The fixture checked
write access on `/lustre` itself, while the cluster grants access only to the
per-user Lustre subtree. A follow-up test-only fixture commit adds the optional
`NEMO_RL_TEST_LUSTRE_ROOT` root, requires its resolved path to remain under
`/lustre`, and creates/cleans only a direct `mkdtemp` child of that configured
root. The controller will cherry-pick it onto the temporary RED branch and
rerun RED/GREEN with the writable per-user root. That rerun evidence remains
pending.

Remote r3 then established the intended split: RED job `480213` failed as
expected, and GREEN job `480214` reached artifact validation. GREEN exposed a
test-helper defect: its generated `metadata.env` and wheel `METADATA` used
literal `\\n` text instead of newline characters. A second test-only follow-up
fixes those writers and directly asserts the generated metadata has separate
parseable lines. The controller rerun with this fix is pending.

Remote r4 confirmed RED job `480228` still failed as expected and GREEN job
`480230` ran all eight artifact-backed tests: seven passed. The remaining
failure was another test-only literal-`\\n` typo in the wiring test's
`submission.env` assertions; the file correctly contained real newline
characters. A final assertion-only follow-up corrects those comparisons. The
controller's final rerun is pending.

## Self-review

- The launcher keeps `/tmp` restricted to the existing bounded worker overlay;
  all validated artifacts, caches, container paths, and run roots resolve under
  `/lustre`.
- The Ray-parent export is outside the renderer's child setup command, so it
  survives into Ray startup rather than being limited to the worker setup
  shell.
- Existing HybridEP and recipe profiles retain their prior behavior; the new
  provenance checks activate for the standard DeepEP path.
- No Bridge submodule, `venvs.py`, or other unowned source file was modified.

## Concern

The supplied external Python lacks `ray`, so the prescribed pytest command
cannot collect. The documented `--confcutdir=tests/unit/tools` fallback is
green, but the local machine also cannot execute the `/lustre`-backed artifact
tests. The controller will run the RED and GREEN commits in the approved
container on a writable `/lustre` host; treat that evidence as pending rather
than locally validated.
