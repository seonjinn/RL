# Task 3 Report: Model and Drafter Matrix

## Scope

Owned files:

- `experiments/vllm_024_dynamicsd/model_method_matrix.json`
- `experiments/vllm_024_dynamicsd/submit_swe_sync_rollout_matrix.sh`
- `experiments/vllm_024_dynamicsd/submit_nemotron_sync_rl_mtp_matrix.sh`
- `tests/test_vllm024_dynamicsd.py`

Unrelated changes were left untouched.

## Binding Compatibility Facts Applied

- Qwen3-30B-A3B, Qwen3-32B, and Qwen3-235B-A22B launch only supported
  baseline and Eagle-3 rows from this matrix.
- PARD is tracked as `INTEGRATION` for the large Qwen sync-rollout matrix and
  is not silently launched before the runner supports it.
- PARD-2 is `UNSUPPORTED` unless an exact target-compatible checkpoint exists;
  it is unsupported for Qwen3-30B-A3B and Qwen3-32B, and not validated for
  Qwen3-235B-A22B.
- DFlash and DFlare public assets are treated as Qwen3-8B-specific and
  unsupported for the large Qwen and Nemotron rows here.
- Nemotron Super and Ultra launch only baseline, native MTP, and dynamic
  native MTP rows.
- Pinned checkpoints, topologies, and context policies are represented in the
  JSON matrix and consumed by the wrappers.

## RED

I added focused tests first, then ran them before creating the matrix or
wrapper manifest behavior.

Command:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py -k 'model_method_matrix or qwen8_only_dflash or approved_compatibility or dry_run_does_not_call_sbatch_or_mutate_dirs or test_only_invokes_sbatch_and_cleans_temp_artifacts or nemotron_sync_rl_wrapper_records_unsupported_matrix_rows'
```

Output:

```text
6 failed, 74 deselected in 1.45s
```

Expected failures:

- `model_method_matrix.json` did not exist
- SWE dry-run/test-only had no wrapper-level `jobs.tsv`
- Nemotron dry-run had no wrapper-level `jobs.tsv`

## Implementation

### Matrix JSON

Created `model_method_matrix.json` with:

- schema version and stable method order
- pinned Qwen3-30B-A3B, Qwen3-32B, Qwen3-235B-A22B checkpoints
- pinned Nemotron Super and Ultra checkpoints
- pinned topology fields:
  - Qwen target TP and rollout node counts
  - Nemotron TP, nodes, segment, distributed backend, Mamba and MoE settings
- pinned context policies:
  - Qwen native 32K and YaRN factor-4 64K rollout profiles
  - Nemotron smoke/full sync-RL math rollout defaults
- compatibility status for each model/method pair with `supported`,
  `integration`, or `unsupported`
- structured `reason_code` plus human-readable reason text

### SWE Wrapper

Reworked `submit_swe_sync_rollout_matrix.sh` to:

- load model/profile settings from the matrix
- emit only supported `baseline`, `static`, and `dynamic` rows
- keep Qwen large-model PARD as `INTEGRATION` in `jobs.tsv`
- keep unsupported PARD-2, DFlash, DFlare, and Nemotron-only MTP rows in
  `jobs.tsv`
- materialize the matched 64K YaRN target and Eagle-3 draft views from the
  matrix policy
- fall back to a local temp manifest root during dry-run/test-only when the
  default result root is not writable

### Nemotron Wrapper

Reworked `submit_nemotron_sync_rl_mtp_matrix.sh` to:

- load pinned Super/Ultra topology and rollout defaults from the matrix
- emit only supported `baseline`, `mtp_static`, and `mtp_dynamic` rows
- record unsupported Eagle-3, PARD, PARD-2, DFlash, and DFlare rows in
  `jobs.tsv`
- fall back to a local temp manifest root during dry-run/test-only when the
  default result root is not writable

## Focused GREEN

Command:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py -k 'model_method_matrix or qwen8_only_dflash or approved_compatibility or dry_run_does_not_call_sbatch_or_mutate_dirs or test_only_invokes_sbatch_and_cleans_temp_artifacts or nemotron_sync_rl_wrapper_records_unsupported_matrix_rows'
```

Output:

```text
6 passed, 74 deselected in 1.68s
```

## Broader Task Verification

Command:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py -k 'swe_sync_rollout_matrix_renders_request_plan_and_response_outputs or swe_sync_rollout_64k_uses_matched_yarn_target_and_draft_views or swe_sync_rollout_non_smoke_defaults_to_primary_four_samples or swe_sync_rollout_full_contract_override_uses_sixteen_samples or nemotron_sync_rl_wrapper_covers_ultra_and_super_bf16 or model_method_matrix or qwen8_only_dflash or approved_compatibility or dry_run_does_not_call_sbatch_or_mutate_dirs or test_only_invokes_sbatch_and_cleans_temp_artifacts or nemotron_sync_rl_wrapper_records_unsupported_matrix_rows'
```

Output:

```text
11 passed, 69 deselected in 3.34s
```

## Shell Dry-Run Evidence

### SWE large-model matrix

Command:

```bash
tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/task3-swe-proof.XXXXXX")
RESULT_ROOT="$tmpdir/results" RUN_ID='task3-swe-proof' DRY_RUN=true \
CLUSTER=lyris REQUIRE_GIT_PULL=false MODELS='qwen32' REQUEST_PROFILES='64k' \
TEMPERATURES='0.0' VARIANTS='baseline static dynamic' \
bash experiments/vllm_024_dynamicsd/submit_swe_sync_rollout_matrix.sh
```

Observed manifest excerpt:

```text
status	model_key	profile_key	method	variant	temperature	run_dir	reason_code	reason
INTEGRATION	qwen32	64k	pard	-	0.0	-	runner_support_missing	...
UNSUPPORTED	qwen32	64k	pard2	-	0.0	-	exact_target_checkpoint_missing	...
UNSUPPORTED	qwen32	64k	dflash	-	0.0	-	qwen3_8b_public_asset_only	...
UNSUPPORTED	qwen32	64k	dflare	-	0.0	-	qwen3_8b_public_asset_only	...
SUPPORTED	qwen32	64k	baseline	baseline	0.0	.../matrix/baseline	-	-
SUPPORTED	qwen32	64k	eagle3	static	0.0	.../matrix/static	-	-
SUPPORTED	qwen32	64k	eagle3	dynamic	0.0	.../matrix/dynamic	-	-
```

Observed stdout facts:

- `materialize_long_context_model_views.py` is rendered for 64K
- `qwen32-target` and `qwen32-eagle3-draft` are pinned
- only supported `sync_variant=` rows are emitted
- no `sbatch` execution occurs in dry-run

### Nemotron matrix

Command:

```bash
tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/task3-nemotron-proof.XXXXXX")
RESULT_ROOT="$tmpdir/results" RUN_ID='task3-nemotron-proof' DRY_RUN=true \
CLUSTER=ptyche REQUIRE_GIT_PULL=false MODELS='ultra super' \
bash experiments/vllm_024_dynamicsd/submit_nemotron_sync_rl_mtp_matrix.sh
```

Observed manifest excerpt:

```text
status	model_key	profile_key	method	variant	run_dir	reason_code	reason
UNSUPPORTED	ultra	sync_rl_math	eagle3	-	-	nemotron_baseline_native_mtp_only	...
UNSUPPORTED	ultra	sync_rl_math	dflash	-	-	qwen3_8b_public_asset_only	...
SUPPORTED	ultra	sync_rl_math	baseline	baseline	.../baseline	-	-
SUPPORTED	ultra	sync_rl_math	mtp_static	mtp_static	.../mtp_static	-	-
SUPPORTED	ultra	sync_rl_math	mtp_dynamic	mtp_dynamic	.../mtp_dynamic	-	-
UNSUPPORTED	super	sync_rl_math	eagle3	-	-	nemotron_baseline_native_mtp_only	...
SUPPORTED	super	sync_rl_math	baseline	baseline	.../baseline	-	-
SUPPORTED	super	sync_rl_math	mtp_static	mtp_static	.../mtp_static	-	-
SUPPORTED	super	sync_rl_math	mtp_dynamic	mtp_dynamic	.../mtp_dynamic	-	-
```

Observed stdout facts:

- Ultra still renders TP8, two-node Ray-backed rows
- Super still renders TP2 single-node rows
- only supported `sync_variant=` rows are emitted

## Full Requested Test Run

Command:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py
```

Output:

```text
80 passed in 8.39s
```

## Self-Review

Commands:

```bash
bash -n experiments/vllm_024_dynamicsd/submit_swe_sync_rollout_matrix.sh
bash -n experiments/vllm_024_dynamicsd/submit_nemotron_sync_rl_mtp_matrix.sh
git diff --check -- experiments/vllm_024_dynamicsd/model_method_matrix.json experiments/vllm_024_dynamicsd/submit_swe_sync_rollout_matrix.sh experiments/vllm_024_dynamicsd/submit_nemotron_sync_rl_mtp_matrix.sh tests/test_vllm024_dynamicsd.py .superpowers/sdd/task-3-report.md
```

Results:

- both shell syntax checks passed
- `git diff --check` produced no output
- restored executable bits on both wrapper scripts after patching

## Commit

Planned command:

```bash
git add experiments/vllm_024_dynamicsd/model_method_matrix.json experiments/vllm_024_dynamicsd/submit_swe_sync_rollout_matrix.sh experiments/vllm_024_dynamicsd/submit_nemotron_sync_rl_mtp_matrix.sh tests/test_vllm024_dynamicsd.py .superpowers/sdd/task-3-report.md
git commit -s -m "feat: define supported SpecDec model matrix"
```

## Task 3 Review Fixes (2026-07-06)

### RED

Focused regression command:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py -k 'unknown_model_without_reusing_state or unknown_profile_without_reusing_state or exact_supported_variants_only or dry_run_does_not_call_sbatch_or_mutate_dirs or test_only_invokes_sbatch_and_cleans_temp_artifacts or records_unsupported_matrix_rows or uses_matrix_defaults_and_allows_env_overrides'
```

RED result before wrapper fixes:

```text
7 failed, 1 passed, 77 deselected in 7.59s
```

Observed failures:

- SWE wrapper returned exit code `0` after `MODELS='qwen32 doesnotexist'`
- SWE wrapper returned exit code `0` after `REQUEST_PROFILES='32k missing'`
- SWE and Nemotron wrappers persisted `RESULT_ROOT` during `DRY_RUN` / `TEST_ONLY`
- Nemotron wrapper ignored matrix `smoke` / `full` values
- Nemotron manifest supported rows did not guarantee `RESULT_ROOT/model_key/variant`

### Implementation

Review fixes applied:

- wrapped both matrix loaders in explicit state-clear / capture / status-check / `eval` helpers so unknown model/profile failures exit nonzero without reusing prior state
- moved `DRY_RUN` / `TEST_ONLY` manifest handling to temporary roots with `trap` cleanup and stdout manifest emission
- fixed Nemotron supported `run_dir` rows to `RESULT_ROOT/model_key/variant`
- consumed Nemotron `smoke` / `full` defaults from the matrix and reapplied explicit environment overrides after loading defaults
- tightened Python test helper typing for Pyright
- replaced the vacuous PARD assertion with an exact submitted variant set and explicit `sync_variant=pard` / `sync_variant=pard2` absence checks

### GREEN

Focused regression command after fixes:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py -k 'unknown_model_without_reusing_state or unknown_profile_without_reusing_state or exact_supported_variants_only or dry_run_does_not_call_sbatch_or_mutate_dirs or test_only_invokes_sbatch_and_cleans_temp_artifacts or records_unsupported_matrix_rows or uses_matrix_defaults_and_allows_env_overrides'
```

Output:

```text
8 passed, 77 deselected in 4.22s
```

### Full Test Pass

Command:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py
```

Output:

```text
85 passed in 12.46s
```

### Shell Validation

Syntax checks:

```bash
bash -n experiments/vllm_024_dynamicsd/submit_swe_sync_rollout_matrix.sh
bash -n experiments/vllm_024_dynamicsd/submit_nemotron_sync_rl_mtp_matrix.sh
```

Result:

```text
both commands exited 0
```

Targeted Pyright:

```bash
pyright tests/test_vllm024_dynamicsd.py
```

Output:

```text
0 errors, 0 warnings, 0 informations
```

SWE dry-run proof:

```bash
tmpdir=$(mktemp -d)
result_root="$tmpdir/swe-dry-result"
view_root="$tmpdir/swe-dry-view"
DRY_RUN=true CLUSTER=lyris REQUIRE_GIT_PULL=false MODELS=qwen32 REQUEST_PROFILES=64k \
TEMPERATURES=0.0 VARIANTS=baseline RUN_ID=manual-dry \
LONG_CONTEXT_VIEW_ROOT="$view_root" RESULT_ROOT="$result_root" \
bash experiments/vllm_024_dynamicsd/submit_swe_sync_rollout_matrix.sh
```

Observed stdout excerpt:

```text
[DRY-RUN] sync_variant=baseline
manifest=/.../vllm024-swe-matrix.4XI1ZF/jobs.tsv
INTEGRATION	qwen32	64k	pard	-	0.0	-	runner_support_missing	...
UNSUPPORTED	qwen32	64k	pard2	-	0.0	-	exact_target_checkpoint_missing	...
UNSUPPORTED	qwen32	64k	dflash	-	0.0	-	qwen3_8b_public_asset_only	...
SUPPORTED	qwen32	64k	baseline	baseline	0.0	.../swe-dry-result/manual-dry/qwen32/64k/t0p0/matrix/baseline	-	-
result_root_exists=no
view_root_exists=no
manifest_exists=no
```

SWE test-only proof:

```bash
PATH="$stub_bin:/usr/bin:/bin" SBATCH_LOG="$sbatch_log" CLUSTER=lyris DRY_RUN=false \
TEST_ONLY=true REQUIRE_GIT_PULL=false MODELS=qwen32 REQUEST_PROFILES=64k \
TEMPERATURES=0.0 VARIANTS=baseline RUN_ID=manual-test \
LONG_CONTEXT_VIEW_ROOT="$view_root" RESULT_ROOT="$result_root" \
bash experiments/vllm_024_dynamicsd/submit_swe_sync_rollout_matrix.sh
```

Observed stdout / scheduler excerpt:

```text
[TEST-ONLY] python3 ... materialize_long_context_model_views.py ...
[TEST-ONLY] sync_variant=baseline
manifest=/.../vllm024-swe-matrix.YhfanO/jobs.tsv
INTEGRATION	qwen32	64k	pard	-	0.0	-	runner_support_missing	...
SUPPORTED	qwen32	64k	baseline	baseline	0.0	.../swe-test-result/manual-test/qwen32/64k/t0p0/matrix/baseline	-	-
arg=--test-only
arg=/.../vllm024-sync-test-only.4e0NR8/baseline/submit.sbatch
result_root_exists=no
view_root_exists=no
manifest_exists=no
```

Nemotron dry-run proof:

```bash
tmpdir=$(mktemp -d)
result_root="$tmpdir/nemo-dry-result"
DRY_RUN=true CLUSTER=ptyche REQUIRE_GIT_PULL=false MODELS=ultra RUN_ID=manual-dry \
RESULT_ROOT="$result_root" \
bash experiments/vllm_024_dynamicsd/submit_nemotron_sync_rl_mtp_matrix.sh
```

Observed stdout excerpt:

```text
[DRY-RUN] sync_variant=baseline
[DRY-RUN] sync_variant=mtp_static
[DRY-RUN] sync_variant=mtp_dynamic
manifest=/.../vllm024-nemotron-matrix.wSJc4S/jobs.tsv
UNSUPPORTED	ultra	sync_rl_math	eagle3	-	-	nemotron_baseline_native_mtp_only	...
UNSUPPORTED	ultra	sync_rl_math	pard	-	-	nemotron_baseline_native_mtp_only	...
SUPPORTED	ultra	sync_rl_math	baseline	baseline	.../nemo-dry-result/ultra/baseline	-	-
SUPPORTED	ultra	sync_rl_math	mtp_static	mtp_static	.../nemo-dry-result/ultra/mtp_static	-	-
SUPPORTED	ultra	sync_rl_math	mtp_dynamic	mtp_dynamic	.../nemo-dry-result/ultra/mtp_dynamic	-	-
result_root_exists=no
manifest_exists=no
```

Nemotron test-only proof:

```bash
PATH="$stub_bin:/usr/bin:/bin" SBATCH_LOG="$sbatch_log" CLUSTER=ptyche DRY_RUN=false \
TEST_ONLY=true REQUIRE_GIT_PULL=false MODELS=ultra RUN_ID=manual-test \
RESULT_ROOT="$result_root" \
bash experiments/vllm_024_dynamicsd/submit_nemotron_sync_rl_mtp_matrix.sh
```

Observed stdout / scheduler excerpt:

```text
[TEST-ONLY] sync_variant=baseline
[TEST-ONLY] sync_variant=mtp_static
[TEST-ONLY] sync_variant=mtp_dynamic
manifest=/.../vllm024-nemotron-matrix.dfcATb/jobs.tsv
UNSUPPORTED	ultra	sync_rl_math	pard	-	-	nemotron_baseline_native_mtp_only	...
SUPPORTED	ultra	sync_rl_math	baseline	baseline	.../nemo-test-result/ultra/baseline	-	-
SUPPORTED	ultra	sync_rl_math	mtp_static	mtp_static	.../nemo-test-result/ultra/mtp_static	-	-
SUPPORTED	ultra	sync_rl_math	mtp_dynamic	mtp_dynamic	.../nemo-test-result/ultra/mtp_dynamic	-	-
arg=--test-only
arg=/.../vllm024-sync-test-only.OXGB8j/baseline/submit.sbatch
arg=/.../vllm024-sync-test-only.OXGB8j/mtp_static/submit.sbatch
arg=/.../vllm024-sync-test-only.OXGB8j/mtp_dynamic/submit.sbatch
result_root_exists=no
manifest_exists=no
```

### Self-Review

Command:

```bash
git diff --check -- experiments/vllm_024_dynamicsd/submit_swe_sync_rollout_matrix.sh experiments/vllm_024_dynamicsd/submit_nemotron_sync_rl_mtp_matrix.sh tests/test_vllm024_dynamicsd.py .superpowers/sdd/task-3-report.md
```

Result:

```text
no output
```

Manual review notes:

- loader helpers now clear state before load, after failed load, and after failed `eval`
- temp manifest cleanup prints unsupported/integration rows before removing the temp root
- Nemotron runtime defaults now come from the matrix, with explicit environment overrides reapplied afterward
- supported variant emission matches the approved compatibility facts; no `sync_variant=pard` or `sync_variant=pard2` launch rows remain
