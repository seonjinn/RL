# Task 2 self-validating launch report

Status: local-only verification; no scheduler contact and no change to `ray.sub`.

Implementation:

- Model selectors now declare `NEMORL_TENSORBOARD_ENABLED`; Qwen3-235B is
  disabled while the four remaining selectors are enabled. `scope_matrix.py`
  parses this through `_selector_bool` and renders the literal selector value.
- `run_scope.sh` validates `TEST_ONLY` and `SBATCH_TEST_ONLY` before profile
  parsing or rendering. Only `0` and `1` are accepted, and the two modes are
  mutually exclusive.
- Router Replay NeMo-RL commands retain the exact rendered driver in an
  atomic command file and invoke `scripts/run_r3_validated_command.sh` with
  shell-quoted absolute positional paths. The wrapper records a pending state,
  isolates each Slurm job/restart trace beneath its run log directory, and
  writes atomic terminal env/JSON records for driver or checker outcomes.
- Real submissions accept only a positive decimal `sbatch --parsable` result.
  Scheduler test-only output is labelled and never interpreted as a job ID.
  Metadata is only published after that validation; its env commands are
  base64 encoded and JSON is authoritative for exact command strings and the
  scheduler argv list.

TDD evidence:

- RED: selector rendering initially forced
  `logger.tensorboard_enabled=true`; the Qwen3-235B selector regression
  failed until selector-driven rendering was implemented.
- RED: the wrapper checker-failure regression returned zero because `$?` was
  read after an `if` compound. It failed with expected exit `19`, observed
  `0`; the checker branch now captures its exit code in `else`.
- GREEN: focused selector/dry-run tests reported `4 passed`.
- GREEN: wrapper success, driver failure, and checker failure tests reported
  `3 passed`; the final isolated launcher suite reported `75 passed`.

Concern:

- The production trace checker path is `${REPO_ROOT}/tools/check_r3_trace.py`;
  this task owns the wrapper but not that checker implementation. The wrapper
  fails closed if the pinned-uv checker cannot run.

Review fix round 1:

- Metadata env now uses canonical identity keys and base64 fields for every
  path/template or arbitrary command; JSON uses a real scheduler argv list and
  a native TensorBoard boolean.
- The Router Replay helper reads a single `O_NOFOLLOW` descriptor, binds the
  exact command bytes to a submitted SHA256, checks the checker before the
  driver, and uses directory descriptors for trace deletion and atomic record
  publication. Driver failure records a null checker result.
- RED/GREEN: a substituted command-file digest regression is rejected before
  execution; wrapper terminal-state and substitution tests reported `4 passed`.
- The combined isolated matrix and launcher suite reported `123 passed`.

Review fix round 2:

- The copied-experiment fake-sbatch harness exercises a real accepted job ID,
  strictly parses every metadata env line, decodes all base64 fields, and
  compares them to authoritative JSON for both R3-off and R3-on submissions.
  It reported `2 passed`.

Review fix round 3:

- Wrapper tests prove the checker is not invoked and has a null exit result
  when the driver fails. Fake-sbatch malformed-output regressions cover empty,
  zero, fractional, and warning-contaminated responses (`7 passed` focused).
- The NeMo-RL wrapper preserves validated `NRL_SLURM_*` aliases into the Ray
  head container before `ray.sub` clears raw Slurm variables; the R3 helper
  uses those aliases exclusively.

Final remediation:

- The submitted command binds both driver and checker SHA256 values. The R3
  helper opens each source once with `O_NOFOLLOW`, validates the descriptor and
  digest, and executes those exact captured bytes through `/bin/bash` stdin and
  `uv run python -`. A driver-side checker mutation cannot change the checker
  that is executed.
- Driver bytes no longer gain a trailing newline. `BASH_ENV` and `ENV` are
  removed, signal exits are normalized while raw return codes are retained,
  and the secondary env record is written before the authoritative JSON
  record using same-directory atomic publication.
- Real-submit metadata now includes container path, runtime-attestation
  SHA256, numeric preflight identity, managed Python/uv identity, checker and
  driver provenance, and separate R3 record pattern/resolved initial path.
  Arbitrary values are base64 encoded in the strict env representation and
  preserved exactly in JSON.
- A scheduler-test-only fake harness proves that validation output is labelled
  and no run directory or metadata is published. A real newline-contaminated
  `sbatch --parsable` result is rejected before metadata publication.
- `virtual_cluster.py` captures the wrapper environment into Ray's runtime
  environment, and `worker_groups.py` merges that environment into actor
  creation, so the validated `NRL_R3_*` variables reach policy actors.
- Final independent re-review: `ADDRESSED`, with zero critical findings,
  warnings, or nits.
- Final combined launcher/submitter verification: `133 passed`; Bash syntax,
  Python compilation, Ruff checks, and `git diff --check` passed. No scheduler
  or remote-cluster contact occurred.
