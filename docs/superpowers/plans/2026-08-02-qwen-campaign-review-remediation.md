# Qwen CUDA Graph Campaign Review Remediation Plan

**Goal:** Close every blocking finding from the Tasks 2–4 campaign review before pushing or allocating GPUs.

**Architecture:** Keep the existing persistent leaf launchers, but add content-addressed promotion/preflight evidence, per-run in-job R3 validation, complete atomic submission metadata, model-specific TensorBoard policy, and strict TensorBoard/W&B export paths. All unsafe or unproven states fail before scheduler contact.

## Global invariants

- Preserve packing, exactly three CUDA Graph warmups, disabled checkpoints, W&B project `sna-cg-study`, OCI `batch`, and four GPUs per node.
- Never accept a bare boolean as correctness evidence. Gate artifacts are regular, non-symlink JSON files with an explicitly supplied SHA256 and matching source/runtime provenance.
- Qwen235 defaults to A/B only. C/E require a passing routed-expert completeness artifact.
- Every 20-step arm requires a passing five-step promotion artifact covering that exact model, arm, R3 state, source, and container.
- Every R3 job validates its own run-unique trace after the driver and fails the SLURM job when validation fails.
- No push or GPU submission occurs until the final review is clean.

### Task R1: Add immutable route and smoke-promotion gates

**Files:**
- Create `experiments/cuda_graph/nemotron_thd_te_graph_20260731/validate_campaign_gate.py`.
- Modify `experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_qwen_router_validation.sh`.
- Modify `tests/unit/experiments/test_matrix_submitters.py`.

**Contract:**

- Validate `TEST_ONLY` and `SBATCH_TEST_ONLY` as mutually exclusive `0|1` values before leaf resolution.
- Default Qwen30 smoke to A/B/C/E and Qwen235 smoke to A/B.
- Require `R3_PREFLIGHT_FILE` plus `R3_PREFLIGHT_SHA256` before Qwen235 C/E. Validate a passed Qwen235 route-completeness record, positive job ID, exact diagnostic settings, and source/container/runtime provenance.
- Require `SMOKE_PROMOTION_FILE` plus `SMOKE_PROMOTION_SHA256` for every performance request. Validate five steps, finite/correctness-passed evidence, positive job IDs, requested arm coverage, R3 identity, and provenance.
- Reject missing, symlinked, malformed, digest-mismatched, failed, wrong-model, wrong-arm, or provenance-mismatched evidence before invoking any leaf.
- Use TDD and commit with Signed-off-by.

### Task R2: Make launches self-validating and fully attested

**Files:**
- Modify every model selector under `experiments/cuda_graph/nemotron_thd_te_graph_20260731/models/`.
- Modify `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scope_matrix.py`.
- Modify `experiments/cuda_graph/nemotron_thd_te_graph_20260731/run_scope.sh`.
- Create `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/run_r3_validated_command.sh`.
- Modify `tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py`.

**Contract:**

- Add a required `NEMORL_TENSORBOARD_ENABLED=true|false` selector field. Keep Qwen235 false; render the selector value instead of forcing true.
- Validate launcher booleans before profile/classification/scheduler output.
- For R3, derive a trace directory containing SLURM job and restart IDs beneath the unique run log directory, clear only that directory, run the driver, then run `tools/check_r3_trace.py --require-forward-verify --require-cp-identity` in the same head container.
- Atomically write a per-job R3 validation record with trace path, exact checker, driver/checker status, and `pending|passed|failed|not_run_driver_failed`; propagate a nonzero result.
- Capture `sbatch --parsable` safely. Distinguish scheduler test-only output from a real job ID; require a positive numeric real job ID.
- Atomically publish run metadata containing exact rendered command, sbatch command, output pattern and resolved output path, job ID, topology, R3 state, source/runtime/container provenance, and R3 validation path.
- Use TDD, including driver/checker failure propagation and fake-sbatch metadata tests, then commit with Signed-off-by.

### Task R3: Make both result exporters identity-safe

**Files:**
- Modify `experiments/cuda_graph/nemotron_thd_te_graph_20260731/export_tensorboard.py`.
- Create `experiments/cuda_graph/nemotron_thd_te_graph_20260731/export_wandb.py`.
- Modify `tests/unit/experiments/test_export_tensorboard.py`.
- Create or modify a focused W&B exporter unit test under `tests/unit/experiments/`.

**Contract:**

- Parse `run-metadata.env` as untrusted data without sourcing it: reject malformed/duplicate keys and shell payloads.
- New exports require either explicit Router Replay identity or metadata; when both are present they must agree. Cross-check all available CLI identity fields against metadata.
- Reuse the canonical metric aliases, strict finite/completeness validation, provenance/parity validation, graph requirements, and atomic JSONL writer for W&B.
- Read W&B with `scan_history`, coalesce partial rows by optimizer step, and never substitute zero for absent graph/correctness/parity evidence.
- Keep Qwen235 TensorBoard disabled unless a separately recorded exact-runtime compatibility smoke permits changing the selector.
- Use injected fake W&B protocols for tests; no network is used by unit tests. Commit with Signed-off-by.

### Task R4: Re-verify, document, review, and push

**Files:**
- Modify the campaign README and durable session files.

**Contract:**

- Document gate schemas, preflight creation, A/B versus C/E promotion, R3 validation records, model-specific logger behavior, metadata-safe export commands, and W&B fallback.
- Run Bash syntax, Python compilation, all campaign unit tests, Qwen30/Qwen235 `TEST_ONLY`, invalid-input probes, and `git diff --check`.
- Run a dedicated review covering all seven original findings and any regression from remediation. Fix and re-review until clean.
- Push `experiment/thd-cg-hybrid-nemotron-20260731` only after the review is clean. Then resume OCI attestation and smoke submission.
