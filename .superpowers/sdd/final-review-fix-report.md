# Final Whole-Branch Review Fix Report

Status: **DONE**

Branch: `codex/vllm024-dynamicsd`

Review context:

- Design: `docs/superpowers/specs/2026-07-06-vllm024-sync-rl-swe-speedbench-design.md`
- Plan: `docs/superpowers/plans/2026-07-06-vllm024-sync-rl-swe-speedbench.md`
- Progress ledger: `.superpowers/sdd/progress.md`
- Review package: `.superpowers/sdd/review-46c02e84..82562787.diff`
- Commit: the signed `HEAD` containing this report; the exact SHA is reported at handoff.
- Dependency follow-up: `e3e70ce8a6e51c3cb5ff00128805239d7cdecd25`
- Host-side staging follow-up: `952ff8d922f0bf73f7aeab545d3398cd2e899752`
- Subset-staging follow-up: `66217f2db4c065505107a827982c132a40ecf850`

## Finding Resolution

### 1. Nemotron SPEED-Bench smoke and tail canaries

- The smoke default is now only `baseline mtp_static`.
- Smoke uses active concurrency 1, one sample per prompt, and one rollout barrier.
- Explicit forced 32K and 64K tail plans were added with matching context limits.
- `mtp_dynamic` is omitted from smoke even when requested.
- Non-smoke `mtp_dynamic` requires a successful signed calibration artifact for
  the exact model and request profile; no default DynamicMTP schedule is launched.

### 2. SPEED overlay exact-work matching

- Overlay batches persist ordered `planned_output_tokens`,
  `actual_output_tokens`, and `forced_output_mask` arrays.
- Forced max/min work is validated against actual output lengths before a result
  is written.
- The SPEED summarizer requires a valid seed, validates all ordered work arrays,
  rejects forced underfill, and requires exact array equality against baseline.
- Overlay sampling remains `temperature=1.0` and `top_p=1.0`.

### 3. Persistent pinned ModelOpt source and official launcher

- Staging authenticates the pinned ModelOpt `run.py` hash, records the commit and
  tree, applies the pinned dataset revision patch, and persists the checkout at
  `<staged-run>/sources/modelopt` rather than leaving it in a job temporary tree.
- The staged run records source, patch, and patched-source hashes in
  `modelopt_source_identity.json` and the prepared manifest.
- `submit_speedbench_official_matrix.sh` provides official baseline/static
  dry-run, `TEST_ONLY`, and canary submission paths using that persisted source.
- Official and overlay execution remain separate cohorts and protocols.

### 4. Calibration repeats, summarization, and launch gate

- The calibration launcher requires exactly three repeats for every concurrency/K
  cell and supports both Eagle-3 static and native MTP static calibration.
- `summarize_speedbench_k_calibration.py` computes medians, selects the smallest K
  within 2 percent of the best median, and applies a monotone non-increasing fit.
- Artifacts include exact model/profile/runtime/request-plan/sampling identity,
  source result paths and hashes, selection policies, medians, fitted schedule,
  and a canonical SHA-256 content signature.
- Validation reopens and hashes every recorded source result, requires exactly
  repeats 1, 2, and 3 in every concurrency/K cell, rebuilds medians, 2-percent
  selection, and monotone fit, then compares the complete rebuilt artifact and
  signature with the submitted artifact.
- Validation rejects missing sources, forged and re-signed derived fields,
  invalid source hashes, inconsistent schedules, non-monotone K, bad
  signatures, and any identity or sampling mismatch.
- SWE DynamicSD and non-SPEED Nemotron DynamicMTP are opt-in and require a
  matching source-replayed artifact. Baseline and static variants remain
  available without calibration, and a matrix or environment schedule cannot
  launch either dynamic variant.

### 5. SWE provenance and exact work

- The synchronous runner now requires and records a nonempty runtime image SHA,
  node count, nonempty explicit vLLM executor backend, full submitted compilation
  config, model and drafter config/checkpoint/view-marker hashes, context profile,
  RoPE hash, and topology.
- Single-node launch defaults use vLLM `uni` for one rank and `mp` for local
  parallel execution; multi-node launch still requires an explicit backend.
- SWE and Nemotron wrappers pass stable context-profile identities.
- `summarize_sync_rollout.py` rejects missing, empty, `unknown`, or `None`
  provenance, requires every field to match, and requires exact planned/actual/
  forced work arrays to match baseline.

### 6. HTML support separation

- The SPEED runner declares distinct official and overlay mode capabilities.
- The report builder loads those declarations from the executable runner.
- The Nemotron table now has separate Official launcher support, Sync-RL overlay
  support, Official limitations, and Overlay gates columns.
- Native MTP static remains accurately listed for the overlay, while its
  official capability is identified as low-level runner support with no
  official Nemotron MTP launcher.
- Official DynamicMTP is shown as unsupported; overlay DynamicMTP is shown as
  signed-calibration-gated and excluded from smoke.
- The Qwen SWE table retains its original support/integration columns.

### 7. Scheduler template hardening

- Dynamic `#SBATCH --job-name` and `#SBATCH --output` lines were removed from:
  `submit_sync_rollout.sh`, `submit_speedbench_k_calibration.sh`, and
  `submit_nemotron_speedbench_sync_mtp_matrix.sh`.
- Job names, output paths, and dependencies are passed as quoted `sbatch` array
  arguments for both real and `--test-only` execution.
- Account, partition, time, node, segment, dependency, and related numeric values
  are validated before rendering.
- Hostile newline, injected directive, and command-substitution regression tests
  execute the generated templates through stub `sbatch` and prove no execution or
  directive injection occurs.

### 8. Exact SPEED staging dependencies

- `speedbench_requirements.lock` uses exact versions for every staging dependency.
- `pyarrow==21.0.0` satisfies the pinned `datasets==4.4.1` requirement proven by
  the remote dry run; the prepared-manifest fixture and expected provenance use
  the same exact version.
- Staging installs only from that lock.
- The prepared manifest records the lock filename, SHA-256, and all package
  versions.

### 9. Design and plan EOFs

- Both requested Markdown files now end with exactly one newline.
- A byte-level regression test prevents extra EOF blank lines.

## TDD Ledger

- Exact work and SWE provenance: 17 expected RED failures, then focused and
  surrounding GREEN runs.
- Persistent ModelOpt source, lock, and official launcher: 5 RED, then 5 GREEN.
- Calibration, tail profiles, smoke, and dynamic gate: 8 RED, then 8 GREEN.
- Scheduler hardening: 7 RED, then 7 GREEN.
- Capability HTML and EOF normalization: 4 RED, then 4 GREEN.
- Table-scoped HTML regression: 1 RED, then GREEN after moving the columns.
- vLLM backend correction: 2 RED, then 2 GREEN.
- Calibration provenance trust boundary: 1 RED, then GREEN after structural
  provenance validation.
- Fresh calibration source replay: 3 RED, then GREEN for a forged re-signed
  artifact, a missing source, and invalid repeat numbering.
- Fresh SWE and non-SPEED Nemotron launch gates: 4 RED, then GREEN for missing,
  matching, and mismatched calibration artifacts.
- Fresh official-launcher HTML semantics: 1 RED, then GREEN after separating
  MTP runner capability from launcher support.

## Verification

- `python3 -m pytest -q tests/test_vllm024_dynamicsd.py`
  - `193 passed in 31.95s`
- `python3 -m pytest -q tests/test_vllm024_report_integration.py`
  - `8 passed in 2.76s`
- `PYTHONPATH=scripts python3 -m pytest -q`
  - `279 passed, 28 subtests passed in 44.38s`
- Generated-script and stub-`sbatch` execution subset
  - `12 passed, 181 deselected in 3.75s`
- `find experiments/vllm_024_dynamicsd -maxdepth 1 -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n`
  - exit 0
- Targeted `python3 -m py_compile` for all changed Python and test modules
  - exit 0
- Targeted `pyright` for all changed Python and test modules
  - `0 errors, 0 warnings, 0 informations`
- `git diff --check`
  - exit 0
- Generated `docs/vllm_standalone_results_latest.html` from the updated builder
  and verified table-scoped official/overlay headers in report integration tests.

## Residual Validation

- No remote GPU or SLURM workload was submitted. Generated run scripts and
  scheduler calls were executed locally with controlled stubs as requested; the
  first real run should remain a `TEST_ONLY`/BS1 canary before full submission.
- No unrelated worktree changes were retained.
