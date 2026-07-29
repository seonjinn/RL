# MXFP8 Shmoo Production A/B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare offline-shmoo-qualified dense MXFP8 tactics against FlashInfer TRTLLM default tactic selection in a matched Qwen3-30B-A3B NeMo-RL rollout experiment.

**Architecture:** Track both the empty baseline manifest and qualified manifest inside one custom vLLM 0.20.2 wheel. NeMo-RL loads one of those package-relative files per arm, validates that every other resolved setting and immutable identity matches, and records runtime dispatch evidence before accepting the paired results.

**Tech Stack:** Bash, Python 3.13, pytest, Hydra/OmegaConf, vLLM 0.20.2, FlashInfer 0.6.8.post1, NeMo-RL, SLURM, OCI-HSG GB200, W&B.

## Global Constraints

- Baseline and shmoo arms must both use direct `flashinfer_trtllm`.
- Baseline and shmoo arms must both use adaptive layout with `switch_m=256`.
- Baseline must use empty tactic tables and runner tactic `-1`.
- Shmoo must use the 106 qualified exact-shape tactics.
- Baseline manifest SHA256 is `3c9f2be89e9053df62d07b937bbbf6f1d4bce39867825cda940271762708a447`.
- Qualified manifest SHA256 is `2baf01def8887db693c35b3070571ab7bb4e72ebfcf30c9fd8b587a3b7c9b2a2`.
- W&B project is `sna_mxfp8_kernel_test`.
- Production jobs use one warmup step and 20 measured steps.
- Production uses three sequential matched repeats on 4 OCI-HSG GB200 nodes with 4 GPUs per node.
- Every source change is committed with `git commit -s` and pushed before job submission.
- Every SLURM suite is checked with `sbatch --test-only` before production submission.

---

### Task 1: Package the no-shmoo TRTLLM baseline manifest in vLLM

**Files:**
- Create: `../vllm-v0202-mxfp8-adaptive-nemorl/vllm/model_executor/kernels/linear/mxfp8/tactic_configs/qwen3_30ba3b_tp1_v0202_rollout_trace_bootstrap.json`
- Modify: `../vllm-v0202-mxfp8-adaptive-nemorl/tests/kernels/quantization/test_mxfp8_tactic_config.py`

**Interfaces:**
- Consumes: `load_mxfp8_dense_runtime_config(reference, ..., package_config_dir)`
- Produces: a package-relative empty-tactic manifest with the fixed baseline SHA256

- [ ] **Step 1: Write the failing package-data contract test**

Add a test that loads
`qwen3_30ba3b_tp1_v0202_rollout_trace_bootstrap.json` from the real package
config directory and asserts the isolation policy:

```python
def test_packaged_qwen_baseline_uses_trtllm_default_tactic() -> None:
    config_dir = _MODULE_PATH.parent / "tactic_configs"
    config = load_mxfp8_dense_runtime_config(
        "qwen3_30ba3b_tp1_v0202_rollout_trace_bootstrap.json",
        actual_vllm_version="0.20.2",
        actual_flashinfer_version="0.6.8.post1",
        actual_compute_capability=(10, 0),
        actual_model="Qwen/Qwen3-30B-A3B",
        actual_tensor_parallel_size=1,
        package_config_dir=config_dir,
    )

    assert config.gemm_backend == "trtllm"
    assert config.layout == "adaptive"
    assert config.switch_m == 256
    assert config.default_tactic == -1
    assert config.tactics_8x4 == ()
    assert config.tactics_128x4 == ()
    assert config.source_sha256 == (
        "3c9f2be89e9053df62d07b937bbbf6f1d4bce39867825cda940271762708a447"
    )
```

- [ ] **Step 2: Run the test and verify the missing package file is the failure**

Run:

```bash
uv run --no-project pytest --confcutdir=tests/kernels/quantization \
  -q tests/kernels/quantization/test_mxfp8_tactic_config.py::test_packaged_qwen_baseline_uses_trtllm_default_tactic
```

Expected: FAIL because the package-relative baseline JSON does not exist.

- [ ] **Step 3: Add the canonical empty-tactic manifest**

Add the already-qualified trace-bootstrap document with:

```json
{
  "compatibility": {
    "compute_capability": "10.0",
    "flashinfer_version": "0.6.8.post1",
    "gpu_family": "GB200",
    "model": "Qwen/Qwen3-30B-A3B",
    "tensor_parallel_size": 1,
    "vllm_base_commit": "5246e3c5df5fb8266b50ceaa6eca2836fb2d13b1",
    "vllm_version": "0.20.2"
  },
  "mode": "adaptive",
  "policy": {
    "default_tactic": -1,
    "direct_trtllm": true,
    "gemm_backend": "trtllm",
    "layout": "adaptive",
    "pad_to_128": false,
    "quant_backend": "cuda",
    "require_8x4_quant": true,
    "require_direct_trtllm": true,
    "switch_m": 256
  },
  "provenance": {
    "container_sha256": "32f07be22293d9a3979e8ba04772ad48a8157dad04fd92577063ed4e07ab1493",
    "minimum_cosine_similarity": 0.999,
    "minimum_speedup_vs_default": 1.02,
    "qualification_repeat_count": 3,
    "qualification_scope": "nemo_rl_qwen3_30ba3b_mxfp8_rollout_trace_bootstrap",
    "source_hint_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "source_manifest_sha256": "439ab42af0bf2ba56a777e8939dc69167871bd6ac2454bce0f3292db84c70064"
  },
  "schema_version": 1,
  "tactics": {
    "128x4": [],
    "8x4": []
  }
}
```

- [ ] **Step 4: Verify the test, JSON checksum, and wheel-builder contracts**

Run:

```bash
uv run --no-project pytest --confcutdir=tests/kernels/quantization -q \
  tests/kernels/quantization/test_mxfp8_tactic_config.py \
  tests/kernels/quantization/test_build_mxfp8_custom_wheel.py
sha256sum \
  vllm/model_executor/kernels/linear/mxfp8/tactic_configs/qwen3_30ba3b_tp1_v0202_rollout_trace_bootstrap.json
```

Expected: all tests pass and SHA256 begins with
`3c9f2be89e9053df62d07b937bbbf6f1d4bce39867825cda940271762708a447`.

- [ ] **Step 5: Commit and push the vLLM package-data change**

```bash
git add \
  tests/kernels/quantization/test_mxfp8_tactic_config.py \
  vllm/model_executor/kernels/linear/mxfp8/tactic_configs/qwen3_30ba3b_tp1_v0202_rollout_trace_bootstrap.json
git diff --cached --check
git commit -s -m "feat(mxfp8): package Qwen3 no-shmoo baseline"
git push seonjinn sna/mxfp8-adaptive-v0.20.2-nemorl
git rev-parse HEAD
```

Record the printed full vLLM commit for Tasks 3 and 5.

### Task 2: Validate the paired manifests and no-shmoo runtime records

**Files:**
- Modify: `experiments/mxfp8_adaptive_rollout/parse_results.py`
- Modify: `tests/unit/experiments/test_mxfp8_adaptive_rollout_results.py`

**Interfaces:**
- Produces: `validate_ab_pair(..., expected_baseline_config_file, expected_baseline_config_sha256, expected_adaptive_config_file, expected_adaptive_config_sha256) -> None`
- Produces: `validate_default_runtime_dispatch(trace_paths: Sequence[Path], *, expected_config_sha256: str) -> dict[str, object]`
- Produces CLI: `validate-default-runtime --trace ... --expected-config-sha256 ... --output ...`

- [ ] **Step 1: Replace the old absent-config pair test with a failing two-manifest test**

Construct original/baseline and adaptive metadata that differ only in
`VLLM_MXFP8_DENSE_CONFIG_FILE` and their exact hashes. Call:

```python
parser.validate_ab_pair(
    baseline,
    adaptive,
    expected_baseline_config_file=(
        "qwen3_30ba3b_tp1_v0202_rollout_trace_bootstrap.json"
    ),
    expected_baseline_config_sha256="3" * 64,
    expected_adaptive_config_file=QUALIFIED_CONFIG_NAME,
    expected_adaptive_config_sha256=CONFIG_HASH,
)
```

Assert acceptance, then independently assert rejection for a wrong baseline
filename, wrong baseline hash, wrong adaptive filename, and wrong adaptive
hash.

- [ ] **Step 2: Run the focused pair tests and verify the signature mismatch**

```bash
uv run pytest -q \
  tests/unit/experiments/test_mxfp8_adaptive_rollout_results.py \
  -k "validate_ab_pair"
```

Expected: FAIL because `validate_ab_pair` still requires the original config
key to be absent and accepts only the adaptive manifest parameters.

- [ ] **Step 3: Implement the two-manifest pair contract**

Change `validate_ab_pair` to require the exact baseline and adaptive
filenames/hashes, strip `VLLM_MXFP8_DENSE_CONFIG_FILE` from both resolved
configs, and compare every remaining field. Update `validate-pair` CLI flags
to:

```text
--expected-baseline-config-file
--expected-baseline-config-sha256
--expected-adaptive-config-file
--expected-adaptive-config-sha256
```

- [ ] **Step 4: Write failing default-runtime validation tests**

Write one accepted trace with records containing:

```python
{
    "event": "mxfp8_adaptive_dispatch",
    "backend": "trtllm",
    "config_sha256": BASELINE_HASH,
    "layout": "8x4",
    "m": 64,
    "n": 128,
    "k": 2048,
    "tactic": -1,
    "tactic_source": "runner_default",
}
```

Assert rejection when backend is not `trtllm`, hash differs, tactic is not
`-1`, tactic source is not `runner_default`, or no dispatch records exist.

- [ ] **Step 5: Run the new runtime tests and verify the missing function**

```bash
uv run pytest -q \
  tests/unit/experiments/test_mxfp8_adaptive_rollout_results.py \
  -k "default_runtime"
```

Expected: FAIL because `validate_default_runtime_dispatch` is absent.

- [ ] **Step 6: Implement runtime validation and its CLI**

Return and write this stable summary:

```python
{
    "backend": "trtllm",
    "config_sha256": expected_config_sha256,
    "dispatch_record_count": record_count,
    "runner_default_record_count": record_count,
    "tactic": -1,
}
```

The implementation must parse all supplied JSONL files, accept only
`mxfp8_adaptive_dispatch` events, and reject malformed or contradictory
eligible records.

- [ ] **Step 7: Run the complete parser tests and commit**

```bash
uv run pytest -q tests/unit/experiments/test_mxfp8_adaptive_rollout_results.py
git add \
  experiments/mxfp8_adaptive_rollout/parse_results.py \
  tests/unit/experiments/test_mxfp8_adaptive_rollout_results.py
git diff --cached --check
git commit -s -m "test(experiments): validate no-shmoo TRTLLM baseline"
```

### Task 3: Make the launcher run the isolated shmoo A/B contract

**Files:**
- Modify: `experiments/mxfp8_adaptive_rollout/run_ab.sh`
- Modify: `experiments/mxfp8_adaptive_rollout/cluster/oci-hsg.env`
- Modify: `tests/unit/experiments/test_mxfp8_adaptive_rollout_results.py`

**Interfaces:**
- Consumes: full vLLM commit printed by Task 1
- Consumes: `validate-default-runtime` and expanded `validate-pair` CLI from Task 2
- Produces: `smoke-ab` schedule with one matched one-step pair
- Produces: `ab` schedule with three matched 20-step pairs

- [ ] **Step 1: Write failing launcher behavior tests**

Run the launcher through fake `sbatch`/spool fixtures and inspect the emitted
jobs, resolved environment, and Hydra overrides. Require:

- W&B project `sna_mxfp8_kernel_test` in both arms
- measured-step default `20`
- run names `baseline-no-shmoo-trtllm-rN` and `shmoo-qualified-rN`
- baseline bootstrap and qualified manifest filenames with their exact SHA256
- baseline `validate-default-runtime` and adaptive `validate-runtime` outputs
- all four expected filename/hash flags on pair validation

Extend the profile behavior test to require `VLLM_OVERLAY_ROOT` and the spool
test to require it first in `PYTHONPATH`. Add a `smoke-ab` schedule test
requiring exactly two jobs and resolved `MEASURE_STEPS=1`.

- [ ] **Step 2: Run launcher tests and verify the old defaults fail**

```bash
uv run pytest -q \
  tests/unit/experiments/test_mxfp8_adaptive_rollout_results.py \
  -k "schedule or launcher or profile or spool"
```

Expected: FAIL on the old 3-step default, missing project, missing overlay,
and original arm with no manifest.

- [ ] **Step 3: Pin experiment identities and overlay**

Set `VLLM_COMMIT` to the exact 40-character stdout recorded by Task 1. Set
the other launcher constants exactly as follows:

```bash
BOOTSTRAP_CONFIG_SHA256="3c9f2be89e9053df62d07b937bbbf6f1d4bce39867825cda940271762708a447"
QUALIFIED_CONFIG_SHA256="2baf01def8887db693c35b3070571ab7bb4e72ebfcf30c9fd8b587a3b7c9b2a2"
WANDB_PROJECT="sna_mxfp8_kernel_test"
```

Set the OCI profile's default overlay to the commit-named path created in
Task 5 and make `run_in_allocation` validate the path, include it first in
`PYTHONPATH`, and reject an overlay whose Git `HEAD` differs from
`VLLM_COMMIT`.

- [ ] **Step 4: Configure both performance arms**

For `original`, load and checksum the package-relative bootstrap manifest,
set `config_hash=$BOOTSTRAP_CONFIG_SHA256`, and add the config filename Hydra
override. For `adaptive`, retain the qualified manifest behavior. Both arms
must set identical trace variables and:

```bash
"logger.wandb.project=$WANDB_PROJECT"
```

Use these W&B names:

```text
mxfp8-qwen-baseline-no-shmoo-trtllm-r1
mxfp8-qwen-shmoo-qualified-r1
```

with the repeat number changing for repeats 2 and 3.

- [ ] **Step 5: Validate runtime traces for both arms**

Always collect `adaptive_dispatch_*.jsonl`. Run `validate-default-runtime`
and write `default_tactic_coverage.json` for `original`; run
`validate-runtime` and write `tactic_coverage.json` for `adaptive`.

Pass all four expected baseline/adaptive filename/hash values to every
`validate-pair` invocation.

- [ ] **Step 6: Add the fixed smoke and production schedules**

Add top-level mode `smoke-ab` that schedules one original job followed by one
adaptive job and internally exports `MEASURE_STEPS=1`. Keep production `ab`
at three repeats and default 20 measured steps. Both schedules retain one
in-job warmup step.

- [ ] **Step 7: Run the launcher/parser tests and commit**

```bash
uv run pytest -q tests/unit/experiments/test_mxfp8_adaptive_rollout_results.py
bash -n experiments/mxfp8_adaptive_rollout/run_ab.sh
git add \
  experiments/mxfp8_adaptive_rollout/run_ab.sh \
  experiments/mxfp8_adaptive_rollout/cluster/oci-hsg.env \
  tests/unit/experiments/test_mxfp8_adaptive_rollout_results.py
git diff --cached --check
git commit -s -m "feat(experiments): isolate MXFP8 shmoo rollout benefit"
```

### Task 4: Update the experiment handoff documentation

**Files:**
- Modify: `experiments/mxfp8_adaptive_rollout/README.md`

**Interfaces:**
- Documents: exact baseline definition, W&B project, smoke command, production command, and accepted metrics

- [ ] **Step 1: Document the exact commands and interpretation**

Document that baseline is direct TRTLLM adaptive layout with empty tactic
tables, not stock vLLM. Provide:

```bash
ACTION=test-only bash experiments/mxfp8_adaptive_rollout/run_ab.sh smoke-ab
ACTION=submit bash experiments/mxfp8_adaptive_rollout/run_ab.sh smoke-ab
ACTION=test-only bash experiments/mxfp8_adaptive_rollout/run_ab.sh ab
ACTION=submit bash experiments/mxfp8_adaptive_rollout/run_ab.sh ab
```

- [ ] **Step 2: Review the rendered handoff against the launcher**

Manually verify the README names the exact W&B project, both arm names,
`smoke-ab`, the 20-step default, and commands that exist in `run_ab.sh`.
Human-facing prose is reviewed directly rather than locked to substring
assertions.

- [ ] **Step 3: Verify, commit, and push NeMo-RL**

```bash
uv run pytest -q tests/unit/experiments/test_mxfp8_adaptive_rollout_results.py
git add \
  experiments/mxfp8_adaptive_rollout/README.md
git diff --cached --check
git commit -s -m "docs(experiments): document MXFP8 shmoo A/B"
git push fork sna/mxfp8-adaptive-vllm-nemorl-main
```

### Task 5: Build and verify the final team wheel and runtime overlay

**Files:**
- Create remotely: the wheel at
  `mxfp8_adaptive_rollout/artifacts/custom-${VLLM_SHA:0:12}/`
  `vllm-0.20.2-1mxfp8g${VLLM_SHA:0:12}-cp38-abi3-manylinux_2_35_aarch64.whl`,
  where `VLLM_SHA` is assigned with `git rev-parse HEAD`
- Create remotely:
  `mxfp8_adaptive_rollout/runtime/vllm-${VLLM_SHA:0:12}-wheel-overlay`

**Interfaces:**
- Consumes: pushed vLLM commit from Task 1
- Produces: deterministic custom wheel, metadata, SHA256 sidecar, and runtime overlay

- [ ] **Step 1: Pull the pushed vLLM commit into a clean OCI-HSG worktree**

Use the existing partial clone, fetch
`sna/mxfp8-adaptive-v0.20.2-nemorl`, create a new detached worktree named by
the first 12 commit characters, and require a clean tracked tree.

- [ ] **Step 2: Build from the pinned official base wheel**

Use:

```text
vllm-0.20.2-cp38-abi3-manylinux_2_35_aarch64.whl
sha256=76ccf4c0554556c06f6b0fb1643742d4cf97dcc69f6ef3f04556d0764126035a
```

Run `tools/mxfp8/build_custom_wheel.py` with the exact source commit and a
create-only output directory.

- [ ] **Step 3: Verify wheel provenance and both manifests**

Require:

```text
embedded source_commit == pushed vLLM commit
baseline manifest SHA256 == 3c9f2be89e9053df62d07b937bbbf6f1d4bce39867825cda940271762708a447
qualified manifest SHA256 == 2baf01def8887db693c35b3070571ab7bb4e72ebfcf30c9fd8b587a3b7c9b2a2
```

Verify the wheel's `.sha256` sidecar with `sha256sum --check`.

- [ ] **Step 4: Create and verify the wheel overlay**

Extract the exact wheel over a clean detached worktree at the same commit.
Require no tracked Git differences, exact `HEAD`, both manifest hashes, and
matching embedded wheel provenance.

- [ ] **Step 5: Update the NeMo profile's literal overlay path if needed**

If the Task 3 profile was written before the final vLLM commit existed,
replace only its default `VLLM_OVERLAY_ROOT` with the exact commit-named
overlay, rerun focused tests, commit with `-s`, and push NeMo-RL again.

### Task 6: Run the compiled one-step matched smoke

**Files:**
- Create remotely: a new create-only smoke suite under `runs/`
- Create remotely: two SLURM logs and runtime dispatch summaries

**Interfaces:**
- Consumes: final pushed NeMo commit, final wheel overlay, both manifest hashes
- Produces: compiled/refit evidence for baseline and shmoo arms

- [ ] **Step 1: Pull the pushed NeMo branch and run `smoke-ab` test-only**

Use account `nemotron_sw_pre`, immutable container
`nemo_rl_nightly_20260728_5675575.sqsh`, and a new suite ID.

- [ ] **Step 2: Submit `smoke-ab` and monitor for five minutes**

The baseline job must complete before the dependent shmoo job starts.

- [ ] **Step 3: Verify the smoke acceptance evidence**

Require both jobs to finish `COMPLETED 0:0`, `enforce_eager=false`, successful
first weight refit, and direct TRTLLM dispatches. Require baseline tactic
summary to report only `-1`; require shmoo coverage to hit all 106 qualified
tactics without fallback for a qualified shape.

### Task 7: Run and report the 20-step production A/B

**Files:**
- Create remotely: one production suite containing six job directories
- Create remotely: parsed JSON and CSV summaries

**Interfaces:**
- Consumes: successful Task 6 smoke
- Produces: three matched baseline/shmoo performance ratios

- [ ] **Step 1: Run production `ab` with `ACTION=test-only`**

Confirm all six jobs are schedulable and their dependency chain alternates
baseline/shmoo for repeats 1 through 3.

- [ ] **Step 2: Submit the production suite and monitor the first five minutes**

Do not override `MEASURE_STEPS`; the launcher default must resolve to 20.

- [ ] **Step 3: Validate each completed pair before the next result is accepted**

Require matching NeMo/vLLM/container/model/topology/seed provenance, 21 total
steps per job, and passing baseline/shmoo runtime dispatch validation.

- [ ] **Step 4: Parse all six logs**

Generate create-only JSON and CSV summaries. Report per repeat and median:

```text
generation tokens/s
generation time
total step time
whole-run wall time
```

Compute speedup as `shmoo / baseline` for throughput and
`baseline / shmoo` for latency.

- [ ] **Step 5: Check W&B isolation and publish the final result**

Require all six runs to be in project `sna_mxfp8_kernel_test` with distinct
baseline/shmoo repeat names. Report job IDs, W&B run links, exact source
commits, wheel SHA256, manifest hashes, runtime tactic coverage, and matched
performance ratios.
