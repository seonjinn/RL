# Timeline

## 2026-08-09 08:40:00 PDT

- NeMo-RL integration commit `4e5f9bac7` pins Bridge `2f6338610`, which pins
  MCore candidate `2dbad0a2d`.
- Verified Bridge and MCore include their current upstream main commits and all
  three nested worktrees are clean at the intended SHAs.

## 2026-08-09 08:25:00 PDT

- ptyche job `2551742` completed the exact 4-node/16-GPU Nano HybridEP gate
  from MCore candidate `fc718cf4c`.
- Verified capture plus 20 changing-route replays and eager/graph parity for
  output, loss, routes, input gradients, all parameter gradients, THD padding,
  and simulated optimizer updates.
- Confirmed the product root cause: `_HybridEPManager.token_probs` and
  `dispatched_probs` retained completed eager autograd graphs across the later TE
  backward capture boundary. Value-preserving manager-side detach fixes capture
  while caller-held tensors preserve gradient ownership.
- Found that NeMo-RL candidate snapshot creation is not consumed by the worker
  launcher because `run_nemorl_scope.sub` clears `PYTHONPATH`. Decided to promote
  the candidate through Bridge/NeMo-RL gitlinks and refresh attestation before
  any 20-step performance claim.
- Merged current MCore main `d12f6c8c9` into candidate `2dbad0a2d` and pushed it.
- Merged current Bridge main `355ef3ea` into Bridge `2f6338610`, resolved the
  MCore FSDP API and dependency metadata conflicts, pinned MCore `2dbad0a2d`,
  passed `uv lock --check` and all pre-commit hooks, and pushed it.

## 2026-08-05 23:54:00 PDT

- All 12 corrected jobs were running after five minutes.
- Every row resolved Python 3.13.14, loaded the NeMo-RL config, and initialized all eight vLLM workers without a fatal marker.
- Generation-side vLLM CUDA Graph capture was active; policy-worker and TE partial-graph initialization remained the next gate.

## 2026-08-05 23:44:30 PDT

- The first matrix reached Ray but all 12 drivers exited before NeMo-RL with `No interpreter found for Python 3.13.14`.
- Compared successful and failed Slurm logs: successful 100-step jobs used the 2026-08-05 nightly ending in runtime job `5884993`; failed jobs used the direct launcher's stale 2026-08-01 default image.
- Resubmitted baseline as `5913139` and the other 11 rows as `5913180` through `5913200`, explicitly pinning the known-good `5884993` image.

## 2026-08-05 23:35:00 PDT

- Pushed exact campaign source `e95e40325` to `experiment/nano-cg-4axis-matrix-20260805`.
- Created fresh recursive OCI-HSG worktree with Bridge `0142aebf` and MCore `281200606`; the six-node scheduler preflight passed.
- Submitted baseline plus 11 valid four-axis scope rows as 12 independent 20-step jobs under `nemotron_n3_post` with 24 GPUs each, all-to-all dispatch, warmup 3, and checkpoints disabled.
- After five minutes all 12 jobs were running, seven had launched their Python driver, and no real error marker was present.

## 2026-08-05 23:21:25 PDT

- User approved rerunning Nano as individual and combined CUDA Graph scopes over `attn`, `mamba`, `moe_router`, and `moe_preprocess`.
- Verified that `moe_preprocess` requires `moe_router`, yielding 12 valid rows including baseline.
- Confirmed OCI-HSG 100-step jobs `5908997` and `5909007` completed with exit code zero from the persistent graph-bank checkout.
- Fast-forwarded the isolated local worktree to merged latest-main commit `4ed047b48`.
- Decided to reuse the committed persistent scope leaves and direct Nano submission path instead of adding another launcher abstraction.

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
- Committed the final runbook as `f9673c5a0` and pushed the reviewed branch to
  the seonjinn fork. The initial OCI read-only query could not resolve the
  cluster hostname. Public GlobalProtect portal resolution and HTTP were
  healthy, but macOS had no NVIDIA internal DNS resolver. Reloaded both
  GlobalProtect agents and opened the UI; fresh Connect/SAML/MFA is now the
  only blocker. No remote files, scheduler state, or GPUs were touched.
