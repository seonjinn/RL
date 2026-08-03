# Timeline

## 2026-08-02 12:21:26 PDT

- User asked for current status and proposed Qwen3-30B-A3B and Qwen3-235B experiments.
- Verified a clean worktree at `e1c24cd9d` and loaded the SLURM, auto-research, session-memory, brainstorming, and testing procedures.
- Found an existing Qwen3-30B-A3B selector and performance/R3 recipes, plus a Qwen3-235B-A22B 16n8g performance recipe but no corresponding experiment selector.
- Decided to recommend a staged campaign: isolate correctness on 30B first, then scale only a passing setting to 235B.

## 2026-08-02 12:34:00 PDT

- Fetched `origin/main`; it is one commit ahead of the experiment branch and includes Qwen3-235B-A22B 16n4g performance recipes.
- Verified active SSH ControlMaster connections to OCI-HSG and ptyche. OCI-HSG has the stronger immediate path because both Qwen model snapshots are already cached.
- Found OCI-HSG job `5794372`, a separate non-CG Qwen3-235B-A22B 16n4g run, completed 20 steps on 64 GPUs with exit code zero.
- W&B steps 6-20 for job `5794372`: 298.849 s/step, 157.450 E2E tokens/s/GPU, 235.862 generation tokens/s/GPU, 907.408 policy-training tokens/s/GPU, and 2849.001 logprob tokens/s/GPU. Mean reward was 0.57370; mean/max token multiplicative probability error was 1.0995/1.6399.
- Audited the R3/CG matrix and confirmed arm D (`R3 on + moe_router CG`) is not a valid correctness experiment until routed expert IDs are made graph inputs or copied into graph-owned stable buffers.

## 2026-08-02 12:42:00 PDT

- User approved the staged Qwen30-to-Qwen235 campaign design.
- Wrote and self-reviewed `docs/superpowers/specs/2026-08-02-qwen-moe-router-cuda-graph-validation-design.md`.
- Committed the design as `d94ccc8d8` (`docs: design Qwen MoE router CG validation`).
- Paused before implementation for the mandatory written-spec review gate.

## 2026-08-02 15:54:34 PDT

- Merged current `origin/main` as `7ace20e3d` while preserving the Bridge
  gitlink `69c29747e` and MCore gitlink `5d320e339`.
- Added the Qwen3-235B-A22B 16n4g selector, safe persistent A/B/C/E condition
  matrix, fail-closed R3/router graph guard, and R3-aware result identity.
- Documented reproducible five-step smoke and 20-step performance invocations.
  The unsafe R3-plus-router graph D arm is rejected locally before Slurm.
- OCI-HSG access, both model snapshots, and the nightly image remain available;
  a fresh remote campaign checkout and runtime attestation are still required.
  No campaign job has been submitted from this branch.

## 2026-08-02 18:50:00 PDT

- Completed review remediation through `aed7138f6`: content-bound campaign
  gates and profiles, self-validating Router Replay attempts, atomic exact run
  metadata, and identity-safe TensorBoard/W&B export.
- Final reporting audit found that graph arms required cache-miss telemetry but
  the worker emitted only hits, captures, replays, and evictions. Added a
  first-class lookup miss counter rather than incorrectly deriving misses from
  captures. Warming and captured outcomes are misses; captured outcomes alone
  increment capture count.
- Verified 89 lifecycle, 70 policy-worker/packing, and 59 algorithm telemetry
  tests locally. Qwen30 and Qwen235 smoke matrices also rendered successfully
  with `TEST_ONLY=1`, which neither created run directories nor contacted
  Slurm.
- Started independent final code, launcher, and documentation reviews. No push,
  OCI checkout mutation, scheduler query, or campaign GPU submission has been
  made yet.
- Committed the exact cache-miss telemetry as `f31d46874`. Independent final
  code review reported `ADDRESSED` with 0 critical findings, 0 warnings, and
  0 nits; its targeted aggregate was 292 passing tests. The three explicit
  invalid-input probes all exited 2 without an `SBATCH:` line.
- Documentation audit found two additional operational blockers: legacy Qwen
  performance/accuracy wrappers did not honor the campaign contract, and the
  Qwen235 R3 envelope asserted rather than bound its raw diagnostic execution.
  The generic wrappers now reject Qwen before launch output. R3 gate validation
  now rejects every hand-authored envelope until a content-bound Slurm producer
  exists, leaving Qwen235 A/B runnable and C/E dependency-blocked.
- Campaign matrix and launcher suites passed 45 and 96 tests respectively.
  The post-audit independent re-review reported `ADDRESSED` with 0 critical
  findings, 0 warnings, and 0 nits. Committed the fail-closed remediation as
  `75ddbef3d`.
