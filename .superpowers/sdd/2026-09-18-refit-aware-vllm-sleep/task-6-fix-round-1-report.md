# Task 6 Fix Round 1 Report

## Status

All three independent-review warnings are fixed in the Task 6-owned functional
harness, CPU regressions, and shared gate environment. No `nemo_rl/` production
file, original performance recipe, original BF16 recipe/wrapper, ledger entry,
or Task 7 file changed. Nothing was pushed and no GB200 execution is claimed.

The worktree started this round clean at `c63e0a5dde2014f6c1a0614a0ccc6ecd47d9c28c`
and remains based on the requested `80ce89281aef0906503e191c114c1245a2a74a33`.

## Signed Commits

Both code commits have SSH signature status `G` and a `Signed-off-by` trailer:

1. `cf73fcf7abcccd08e4e32ea47bfcc9dafa13fe83`
   (`test(refit): preserve manifest and Ray session lifecycle`)
2. `b8479e3e5518dd0204b80775339e48fd7a06ab2e`
   (`fix(refit): consume launcher provenance arguments`)

The documentation update containing this report is a separate signed commit.

## Warning Resolutions

### Resident Manifest

The Qwen fixture now performs the same initial communicator metadata exchange
explicitly and retains the resulting manifest on `Runtime`. It occurs after
policy construction and `generation.finish_generation()`, while Megatron
storage is resident and vLLM is asleep. Initial A then follows the normal IPC
synchronizer path and offloads policy storage.

The discarded-state failure node now derives its intentionally incomplete
receiver manifest from that retained initial manifest. There is no policy
`prepare_refit_info()` call between A/offload and destructive sleep/watchdog.
The lifecycle regression makes any second export fail when its fake policy is
offloaded, then verifies the exact event order: one resident export, initial
receiver install, destructive sleep, watchdog start, failure-manifest install.

### Ray Session

`refit_sleep_ray_session` is session-scoped and wraps `managed_ray_session`.
The manager calls NeMo-RL `init_ray()` only when the driver is disconnected,
keeps that connection across both function-scoped Qwen model fixtures and the
ABC fresh-B/fresh-C constructions, then calls `ray.shutdown()` once at pytest
session teardown. A connection owned by an outer caller is not disconnected.

Each Qwen fixture still shuts down generation, policy, and its virtual-cluster
placement groups. The BF16 node uses the same session fixture when selected by
its one-node-ID wrapper. A faithful CPU regression executes ABC and failure
fixture lifecycles sequentially and records one init and one final shutdown.

### Launcher Arguments

`refit_sleep.env` now allowlists exactly these `tools/launch` forms:

```text
logger.wandb.name=...
++git_meta=...
++container=...
```

Values must be nonempty and nonduplicate. They are exported as
`NRL_REFIT_SLEEP_LAUNCH_WANDB_NAME`, `NRL_REFIT_SLEEP_LAUNCH_GIT_META`, and
`NRL_REFIT_SLEEP_LAUNCH_CONTAINER`, which the existing sanitized provenance
capture records. Positional parameters are consumed and never appended to the
fixed pytest command. Every other argument still exits with status 2.

The executable shell regression sources the real environment with the exact
three launcher forms while stubbing only filesystem/runtime commands. A second
case adds `policy.generation.vllm_cfg.enforce_eager=true` and requires rejection.

## TDD Evidence

RED was observed before implementation:

- Lifecycle test collection failed because `managed_ray_session` did not exist.
- The real gate environment rejected the exact three mandatory launcher forms
  with `This wrapper accepts no pytest or recipe overrides`; both shell tests
  failed.

After the minimal implementation, focused GREEN evidence was:

```text
tests/unit/models/generation/test_refit_sleep_runtime.py: 3 passed
tests/unit/models/generation/test_refit_sleep_recipe.py: 9 passed
exact launcher-argument selection: 2 passed, 7 deselected
```

An intermediate GREEN attempt exposed a typo in the fake manifest sentinel and
macOS Bash 3.2's lack of associative arrays. The final shell implementation uses
portable scalar variables and the final regression output above is clean.

## Final Verification

The prior 114-test CPU set now contains four added regressions and reports 118
passes:

```text
Task 6 oracle/recipe/runtime/IPC files: 27 passed in 16.66s
existing weight synchronizer suite: 91 passed in 17.72s
selected recipe/suite checks: 5 passed, 11 deselected in 2.58s
functional CPU execution: 3 skipped in 0.02s
exact functional collection: 3 tests collected
```

All nine Task 6 Python files pass `ruff check`, `ruff format --check`, and
`compileall`. `bash -n` passes for `common.env`, `refit_sleep.env`, and both new
wrappers. Both wrappers pass `TEST_DRYRUN=1`. `git diff --check` passes. The
scope diff against base is empty for `nemo_rl/`, the original performance recipe
directory, and the original Qwen3.5 BF16 wrapper.

The exact collected node IDs remain:

```text
tests/functional/test_vllm_refit_sleep.py::test_qwen3_mxfp8_destructive_refit_abc
tests/functional/test_vllm_refit_sleep.py::test_qwen3_mxfp8_missing_manifest_after_discard
tests/functional/test_vllm_refit_sleep.py::test_qwen35_bf16_nccl_reshard_preserving_control
```

`tools/launch DRYRUN=1` still exits before argument construction on macOS because
its existing GNU-sed config expression does not match under BSD sed. No snapshot
or submission occurred. This launcher portability issue predates Task 6 and was
not broadened into this fix; the real environment's exact generated argument
forms are covered by the executable shell tests above.

## Required GB200 Run

No 4x4 MXFP8 or 6x4 BF16 job ran locally. After the final signed documentation
commit is available on the cluster under `/home`, use a reviewed immutable image
and set the exact final source identity:

```bash
export NRL_REFIT_SLEEP_EXPECTED_SHA=$(git rev-parse HEAD)
export NRL_REFIT_SLEEP_IMAGE="$NIGHTLY_SQSH"
export NRL_REFIT_SLEEP_IMAGE_DIGEST="sha256:$(sha256sum "$NIGHTLY_SQSH" | cut -d' ' -f1)"
export CONTAINER="$NRL_REFIT_SLEEP_IMAGE"
export MOUNTS="/home:/home,/lustre:/lustre,/raid/scratch:/raid/scratch"
export NRL_REFIT_SLEEP_ARTIFACT_DIR="/lustre/$USER/experiments/refit-sleep/$RUN_ID/qwen3"
export NRL_REFIT_SLEEP_CACHE_ROOT="/raid/scratch/$USER/refit-sleep-$RUN_ID"
export RAY_TMPDIR="$NRL_REFIT_SLEEP_CACHE_ROOT/ray"
export GPUS_PER_NODE=4
export COMMAND="bash tests/test_suites/llm/vllm-destructive-refit-qwen3-30ba3b-4n4g.sh"
sbatch --test-only --account="$ACCOUNT" --partition="$PARTITION" \
  --nodes=4 --ntasks-per-node=1 --exclusive --segment=4 --time=04:00:00 \
  --job-name=refit-sleep-qwen3 --export=ALL \
  --output="$NRL_REFIT_SLEEP_ARTIFACT_DIR/slurm-%j.out" ray.sub
```

For the BF16 control, use its new wrapper as `COMMAND`, a distinct artifact and
cache directory, `--nodes=6`, and `--segment=2`. The 4x4 launcher path may also
invoke the wrapper through `tools/launch`; the three generated provenance
arguments are now accepted without becoming recipe overrides. Do not set
`NRL_REFIT_SLEEP_TOLERANCES` until the first GB200 observation is reviewed.

## Changed Files This Round

```text
tests/functional/refit_sleep_runtime.py
tests/functional/refit_sleep_utils.py
tests/functional/test_vllm_refit_sleep.py
tests/test_suites/llm/refit_sleep.env
tests/unit/models/generation/test_refit_sleep_recipe.py
tests/unit/models/generation/test_refit_sleep_runtime.py
.superpowers/sdd/2026-09-18-refit-aware-vllm-sleep/task-6-implementer-report.md
.superpowers/sdd/2026-09-18-refit-aware-vllm-sleep/task-6-fix-round-1-report.md
```

## Residual Risks

- GB200 still must validate real Megatron resident/offload behavior, repeated
  Ray placement-group teardown in one driver session, CUDA IPC ACK completion,
  and total shutdown latency.
- No MXFP8 numerical tolerance is pinned. The first observation run must fail
  closed after writing evidence, then a reviewed tolerance file must be used for
  the acceptance rerun.
- The BF16 control delegates to a child process while the parent remains attached
  to Ray; that existing multi-driver cluster behavior cannot be exercised on
  macOS and remains part of the 6x4 run.
- `tools/launch` depends on GNU sed. Its BSD-sed dry-run failure is independent of
  the fixed provenance allowlist and remains an environment limitation.
