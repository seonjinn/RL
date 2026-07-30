# Qwen3-235B Thinking DFlash v2 NeMo-RL E2E Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run matched Qwen3-235B-A22B-Thinking-2507 NeMo-RL Math and SWE comparisons that measure no-SpecDec against DFlash v2 with vLLM 0.25.1, complete W&B provenance, and baseline-relative E2E metrics.

**Architecture:** A focused experiment package owns an immutable two-domain matrix, validates DFlash and CUDA Graph constraints, renders the existing NeMo-RL launchers, and normalizes results. A minimal tested extension teaches the shared low-level Math launcher to accept DFlash without enabling online drafter training. Math advances through 1, 3, and 20 steps; SWE advances through 1 and 3 steps after Math passes.

**Tech Stack:** Python 3.12+, PyYAML, Bash, pytest, NeMo-RL, Ray, vLLM 0.25.1, W&B, SLURM, Lyris GB200

## Global Constraints

- Verifier is exactly `Qwen/Qwen3-235B-A22B-Thinking-2507`.
- Verifier snapshot is exactly `/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/models--Qwen--Qwen3-235B-A22B-Thinking-2507/snapshots/6cbffae6d8e28b986a6b17bd36f42f9fa0f1f0a5`.
- Drafter is exactly `/home/sna/drafters/dflash_235bthink_v2`.
- Container is exactly `/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260715.sqsh`.
- Generation venv is exactly `/lustre/fsw/coreai_dlalgo_llm/users/sna/nrl_venvs_dynsd025`.
- Driver and generation workers must report `vllm==0.25.1`.
- Sampling is `temperature=1.0`, `top_p=1.0`, and no top-k truncation.
- DFlash uses draft TP1, `FLASH_ATTN`, draft `max_model_len=4096`, and FlashInfer autotune disabled.
- Math compares baseline K0 with DFlash v2 K3 at 1, 3, and 20 completed steps.
- SWE compares baseline K0 with DFlash v2 K5 at 1 and 3 completed steps.
- SpecDec uses FULL CUDA Graph mode and explicit `(K+1)`-multiple capture sizes through `effective_max_num_seqs * (K+1)`.
- `max_cudagraph_capture_size` is forbidden.
- Step 1 is excluded from the primary Math 20-step steady-state window; report steps 2-20.
- W&B is enabled for every submitted arm, and the exact run URL is required in the ledger.
- The launcher defaults to dry-run and requires `SUBMIT=true` for submission.
- Always run `git pull --ff-only` on Lyris immediately before submission.
- Always run `sbatch --test-only` before submission and monitor submitted jobs for at least five minutes.
- Preserve unrelated dirty-worktree changes and commit only files owned by this experiment.

---

## File Structure

- Create `experiments/qwen235b_thinking_dflash_nemorl_e2e/__init__.py`: package marker.
- Create `experiments/qwen235b_thinking_dflash_nemorl_e2e/matrix.yaml`: immutable Math/SWE contract.
- Create `experiments/qwen235b_thinking_dflash_nemorl_e2e/contract.py`: typed loading, validation, capture derivation, and environment rendering.
- Create `experiments/qwen235b_thinking_dflash_nemorl_e2e/preflight.py`: runtime, checkpoint, source, and W&B-safe provenance checks.
- Create `experiments/qwen235b_thinking_dflash_nemorl_e2e/submit_lyris_math.sh`: gated Math dry-run, test-only, and submission wrapper.
- Create `experiments/qwen235b_thinking_dflash_nemorl_e2e/submit_lyris_swe.sh`: gated SWE wrapper around the existing SWERL launcher.
- Create `experiments/qwen235b_thinking_dflash_nemorl_e2e/collect_results.py`: driver-log/W&B-ledger normalization and baseline matching.
- Create `experiments/qwen235b_thinking_dflash_nemorl_e2e/README.md`: exact commands and gate policy.
- Create `tests/test_qwen235b_thinking_dflash_nemorl_e2e.py`: local unit and shell-contract tests.
- Modify `experiments/eagle3_online/submit_nemorl_online_draft_specdec.sh`: accept static DFlash as a supported draft format.
- Modify `scripts/build_pages_index.py`: render final matched Math/SWE rows and W&B links.

### Task 1: Add static DFlash support to the low-level NeMo-RL launcher

**Files:**
- Modify: `experiments/eagle3_online/submit_nemorl_online_draft_specdec.sh`
- Create: `tests/test_submit_nemorl_online_dflash.py`

**Interfaces:**
- Consumes environment variables `DRAFT_FORMAT=dflash`, `SPECDEC_METHOD=dflash`, `DRAFT_MODEL`, `DRAFT_TP=1`, and `NUM_SPECULATIVE_TOKENS`.
- Produces Hydra overrides for `speculative_config.method=dflash`, model, K, and draft TP without enabling `policy.draft`.
- The Math wrapper in Task 3 consumes this interface.

- [ ] **Step 1: Write the failing shell-contract test**

Create a test that copies the launcher to a temporary directory, supplies fake
non-empty container/config/model files, and invokes dry-run with:

```python
def test_dflash_dry_run_renders_static_speculator(tmp_path: Path) -> None:
    result = run_low_level_launcher(
        tmp_path,
        {
            "DRAFT_FORMAT": "dflash",
            "SPECDEC_METHOD": "dflash",
            "DRAFT_MODEL": "/home/sna/drafters/dflash_235bthink_v2",
            "DRAFT_TP": "1",
            "NUM_SPECULATIVE_TOKENS": "3",
            "POLICY_DRAFT_ENABLED": "false",
            "DRY_RUN": "true",
        },
    )
    assert result.returncode == 0
    assert "speculative_config.method=dflash" in result.stdout
    assert "speculative_config.model=/home/sna/drafters/dflash_235bthink_v2" in result.stdout
    assert "speculative_config.draft_tensor_parallel_size=1" in result.stdout
    assert "policy.draft.enabled=true" not in result.stdout
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
pytest -q tests/test_submit_nemorl_online_dflash.py
```

Expected: FAIL with `DRAFT_FORMAT must be auto, eagle3, pard, pard2, suffix, ngram, or pld`.

- [ ] **Step 3: Implement the minimal DFlash branch**

Add `dflash` to the draft-format mapping and validation:

```bash
elif [[ "${DRAFT_FORMAT}" == "dflash" && -z "${SPECDEC_METHOD:-}" ]]; then
  SPECDEC_METHOD="dflash"
fi
```

Add a `dflash)` case that requires `SPECDEC_METHOD=dflash`, a non-empty
`DRAFT_MODEL`, `DRAFT_TP=1`, and forces:

```bash
POLICY_DRAFT_ENABLED=false
SPECDEC_PARALLEL_DRAFTING=false
INCLUDE_DRAFT_TP=true
```

Add `dflash` to the model-backed method arm that renders `model` and
`draft_tensor_parallel_size`.

- [ ] **Step 4: Run focused and syntax tests**

Run:

```bash
pytest -q tests/test_submit_nemorl_online_dflash.py
bash -n experiments/eagle3_online/submit_nemorl_online_draft_specdec.sh
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/eagle3_online/submit_nemorl_online_draft_specdec.sh \
  tests/test_submit_nemorl_online_dflash.py
git commit -s -m "feat: support static DFlash in NeMo-RL launcher"
```

### Task 2: Define the immutable two-domain experiment contract

**Files:**
- Create: `experiments/qwen235b_thinking_dflash_nemorl_e2e/__init__.py`
- Create: `experiments/qwen235b_thinking_dflash_nemorl_e2e/matrix.yaml`
- Create: `experiments/qwen235b_thinking_dflash_nemorl_e2e/contract.py`
- Modify: `tests/test_qwen235b_thinking_dflash_nemorl_e2e.py`

**Interfaces:**
- Produces `ExperimentContract`, `DomainContract`, `ArmContract`, `load_contract(path)`, `derive_capture_sizes(max_num_seqs, k)`, and `render_arm_env(contract, domain, arm, stage, run_id)`.
- Tasks 3, 4, and 6 consume the rendered environment and `comparison_key()`.

- [ ] **Step 1: Write failing contract tests**

```python
def test_matrix_pins_thinking_target_and_vllm0251() -> None:
    contract = load_contract(MATRIX)
    assert contract.target_model == "Qwen/Qwen3-235B-A22B-Thinking-2507"
    assert contract.runtime_vllm == "0.25.1"
    assert contract.temperature == 1.0
    assert contract.top_p == 1.0
    assert set(contract.domains["math"].stages) == {"smoke", "pilot", "measurement"}
    assert contract.domains["math"].arms["dflash_v2"].k == 3
    assert contract.domains["swe"].arms["dflash_v2"].k == 5


def test_capture_sizes_cover_exact_verify_batch() -> None:
    assert derive_capture_sizes(max_num_seqs=64, k=3)[-1] == 256
    assert all(size % 4 == 0 for size in derive_capture_sizes(64, 3))
    assert derive_capture_sizes(max_num_seqs=16, k=5)[-1] == 96
    assert all(size % 6 == 0 for size in derive_capture_sizes(16, 5))


def test_contract_rejects_max_capture_shortcut(tmp_path: Path) -> None:
    bad = tmp_path / "matrix.yaml"
    bad.write_text(MATRIX.read_text() + "\nmax_cudagraph_capture_size: 256\n")
    with pytest.raises(ContractError, match="max_cudagraph_capture_size"):
        load_contract(bad)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
pytest -q tests/test_qwen235b_thinking_dflash_nemorl_e2e.py -k contract
```

Expected: import failure because the experiment package does not exist.

- [ ] **Step 3: Create the exact matrix**

`matrix.yaml` must encode:

```yaml
schema_version: 1
target_model: Qwen/Qwen3-235B-A22B-Thinking-2507
target_snapshot: /lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/models--Qwen--Qwen3-235B-A22B-Thinking-2507/snapshots/6cbffae6d8e28b986a6b17bd36f42f9fa0f1f0a5
runtime_vllm: 0.25.1
container: /lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260715.sqsh
venv: /lustre/fsw/coreai_dlalgo_llm/users/sna/nrl_venvs_dynsd025
temperature: 1.0
top_p: 1.0
top_k: -1
drafter:
  checkpoint: /home/sna/drafters/dflash_235bthink_v2
  tensor_parallel_size: 1
  max_model_len: 4096
  attention_backend: FLASH_ATTN
  enable_flashinfer_autotune: false
domains:
  math:
    config: examples/configs/recipes/llm/performance/grpo-qwen3-235b-32n4g.yaml
    max_num_seqs: 64
    stages: {smoke: 1, pilot: 3, measurement: 20}
    arms:
      baseline: {method: baseline, k: 0}
      dflash_v2: {method: dflash, k: 3}
  swe:
    config: examples/nemo_gym/grpo_qwen3_235b_thinking_swe2_smoke.yaml
    dataset: data/swe2/train-pool224.jsonl
    validation_dataset: data/swe2/val-mini3.jsonl
    max_num_seqs: 16
    stages: {smoke: 1, pilot: 3}
    arms:
      baseline: {method: baseline, k: 0}
      dflash_v2: {method: dflash, k: 5}
```

- [ ] **Step 4: Implement strict dataclasses and environment rendering**

The loader must reject any target, version, sampling, DFlash, graph, K, or
stage mismatch. `render_arm_env()` must emit:

```text
MODEL_PATH=/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/models--Qwen--Qwen3-235B-A22B-Thinking-2507/snapshots/6cbffae6d8e28b986a6b17bd36f42f9fa0f1f0a5
NEMO_RL_VENV_DIR=/lustre/fsw/coreai_dlalgo_llm/users/sna/nrl_venvs_dynsd025
MAX_STEPS=1
WANDB_ENABLED=true
GENERATION_TEMPERATURE=1.0
GENERATION_TOP_P=1.0
GENERATION_TOP_K=-1
```

`MAX_STEPS` is `1`, `3`, or `20` according to the selected stage; SWE exposes
only `1` and `3`.

For DFlash it must also emit `DRAFT_FORMAT=dflash`, `SPECDEC_METHOD=dflash`,
`DRAFT_MODEL`, `DRAFT_TP=1`, K, FULL graph mode, and the derived capture list.
Baseline must emit `ENABLE_VLLM_SPECDEC=false`.

- [ ] **Step 5: Run tests**

Run:

```bash
pytest -q tests/test_qwen235b_thinking_dflash_nemorl_e2e.py -k contract
python3 -m py_compile experiments/qwen235b_thinking_dflash_nemorl_e2e/contract.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add experiments/qwen235b_thinking_dflash_nemorl_e2e/__init__.py \
  experiments/qwen235b_thinking_dflash_nemorl_e2e/matrix.yaml \
  experiments/qwen235b_thinking_dflash_nemorl_e2e/contract.py \
  tests/test_qwen235b_thinking_dflash_nemorl_e2e.py
git commit -s -m "feat: define Qwen235B Thinking DFlash E2E contract"
```

### Task 3: Add runtime, checkpoint, and W&B-safe preflight

**Files:**
- Create: `experiments/qwen235b_thinking_dflash_nemorl_e2e/preflight.py`
- Modify: `tests/test_qwen235b_thinking_dflash_nemorl_e2e.py`

**Interfaces:**
- Produces `RuntimeProbe`, `CheckpointRecord`, `validate_runtime(driver, worker, expected_version)`, `checkpoint_record(path)`, and a JSON provenance CLI.
- Submission wrappers consume a successful JSON record.

- [ ] **Step 1: Write failing preflight tests**

```python
def test_preflight_rejects_worker_vllm020() -> None:
    driver = RuntimeProbe("driver", "0.25.1", "/dyn/python", "/dyn/vllm")
    worker = RuntimeProbe("worker", "0.20.0", "/opt/python", "/opt/vllm")
    with pytest.raises(PreflightError, match="worker vLLM"):
        validate_runtime(driver, worker, "0.25.1")


def test_checkpoint_requires_config_and_safetensors(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(PreflightError, match="safetensors"):
        checkpoint_record(tmp_path)


def test_provenance_omits_wandb_secret() -> None:
    assert "WANDB_API_KEY" not in safe_environment({"WANDB_API_KEY": "secret", "RUN_ID": "x"})
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
pytest -q tests/test_qwen235b_thinking_dflash_nemorl_e2e.py -k preflight
```

Expected: FAIL because `preflight.py` does not exist.

- [ ] **Step 3: Implement probes and provenance**

Probe `vllm.__version__`, `vllm.__file__`, `sys.executable`, Ray version, and
machine architecture inside both driver and generation-worker environments.
Record config SHA256 plus safetensors filename, size, and mtime. Never record
the W&B key or complete environment.

- [ ] **Step 4: Run tests**

```bash
pytest -q tests/test_qwen235b_thinking_dflash_nemorl_e2e.py -k preflight
python3 -m py_compile experiments/qwen235b_thinking_dflash_nemorl_e2e/preflight.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/qwen235b_thinking_dflash_nemorl_e2e/preflight.py \
  tests/test_qwen235b_thinking_dflash_nemorl_e2e.py
git commit -s -m "feat: gate Qwen235B DFlash E2E provenance"
```

### Task 4: Build the gated Math submission wrapper

**Files:**
- Create: `experiments/qwen235b_thinking_dflash_nemorl_e2e/submit_lyris_math.sh`
- Modify: `tests/test_qwen235b_thinking_dflash_nemorl_e2e.py`

**Interfaces:**
- Inputs: `ARM=baseline|dflash_v2`, `STAGE=smoke|pilot|measurement`, `MODE=dry-run|test-only|submit`.
- Consumes contract rendering and the shared low-level launcher.
- Produces `artifacts/qwen235b_thinking_dflash_nemorl_e2e/$RUN_ID/jobs.csv` and `provenance.json`.

- [ ] **Step 1: Write failing wrapper tests**

```python
def test_math_wrapper_defaults_to_dry_run() -> None:
    result = run_math_wrapper("baseline", "smoke")
    assert result.returncode == 0
    assert "MODE=dry-run" in result.stdout
    assert "MAX_STEPS=1" in result.stdout
    assert "TARGET_MODEL_ID=Qwen/Qwen3-235B-A22B-Thinking-2507" in result.stdout


def test_dflash_measurement_requires_passing_pilot() -> None:
    result = run_math_wrapper("dflash_v2", "measurement", gate_root="/missing")
    assert result.returncode == 2
    assert "passing pilot gate" in result.stderr
```

- [ ] **Step 2: Run tests and verify RED**

```bash
pytest -q tests/test_qwen235b_thinking_dflash_nemorl_e2e.py -k math_wrapper
```

Expected: FAIL because `submit_lyris_math.sh` does not exist.

- [ ] **Step 3: Implement dry-run, test-only, and submit modes**

The wrapper must:

1. render the exact arm environment;
2. verify source commit, container, venv, target, drafter, and W&B login;
3. run the resolved Hydra configuration probe;
4. reject `attention_backend` inside `SpeculativeConfig` if the installed
   vLLM schema does not accept it, while preserving the known-good runtime
   attention environment;
5. run `sbatch --test-only`;
6. require `MODE=submit` for submission;
7. write the SLURM job ID immediately;
8. extract and append the exact W&B URL from the driver log.

- [ ] **Step 4: Run local tests and shell validation**

```bash
pytest -q tests/test_qwen235b_thinking_dflash_nemorl_e2e.py -k math_wrapper
bash -n experiments/qwen235b_thinking_dflash_nemorl_e2e/submit_lyris_math.sh
```

Expected: PASS.

- [ ] **Step 5: Commit and push before remote work**

```bash
git add experiments/qwen235b_thinking_dflash_nemorl_e2e/submit_lyris_math.sh \
  tests/test_qwen235b_thinking_dflash_nemorl_e2e.py
git commit -s -m "feat: add gated Qwen235B DFlash Math launcher"
git push
```

### Task 5: Run Math gates and promote the matched pair

**Files:**
- Create through the wrapper: `artifacts/qwen235b_thinking_dflash_nemorl_e2e/$RUN_ID/jobs.csv`
- Create through the wrapper: `artifacts/qwen235b_thinking_dflash_nemorl_e2e/$RUN_ID/provenance.json`

**Interfaces:**
- Consumes Task 4 launcher and passing prior-stage records.
- Produces one matched baseline/DFlash job pair per promotion tier.

- [ ] **Step 1: Refresh Lyris and run read-only preflight**

```bash
ssh login-lyris "cd /project/coreai_dlalgo_llm/users/sna/RL-latest-main-canary-20260618 && git pull --ff-only"
MODE=dry-run ARM=baseline STAGE=smoke \
  bash experiments/qwen235b_thinking_dflash_nemorl_e2e/submit_lyris_math.sh
MODE=dry-run ARM=dflash_v2 STAGE=smoke \
  bash experiments/qwen235b_thinking_dflash_nemorl_e2e/submit_lyris_math.sh
```

Expected: both resolve the same verifier, recipe, topology, sampling, and
runtime; only SpecDec fields differ.

- [ ] **Step 2: Run scheduler preflight**

```bash
MODE=test-only ARM=baseline STAGE=smoke \
  bash experiments/qwen235b_thinking_dflash_nemorl_e2e/submit_lyris_math.sh
MODE=test-only ARM=dflash_v2 STAGE=smoke \
  bash experiments/qwen235b_thinking_dflash_nemorl_e2e/submit_lyris_math.sh
```

Expected: both `sbatch --test-only` calls are accepted.

- [ ] **Step 3: Submit one-step jobs**

```bash
MODE=submit ARM=baseline STAGE=smoke \
  bash experiments/qwen235b_thinking_dflash_nemorl_e2e/submit_lyris_math.sh
MODE=submit ARM=dflash_v2 STAGE=smoke \
  bash experiments/qwen235b_thinking_dflash_nemorl_e2e/submit_lyris_math.sh
```

- [ ] **Step 4: Monitor for five minutes and classify**

Check `squeue`, `sacct`, and the last 100 driver-log lines. A smoke passes only
when it completes generation, refit, policy update, one full step, and records
an exact W&B run URL.

- [ ] **Step 5: Promote both arms to three steps**

Repeat Steps 1-4 with `STAGE=pilot`. Reject promotion if either arm has eager
verification fallback, missing acceptance metrics, CUDA errors, incomplete
steps, or mismatched provenance.

- [ ] **Step 6: Promote both arms to twenty steps**

Repeat Steps 1-4 with `STAGE=measurement`. Use `WALLTIME=05:00:00`, the Lyris
maximum for `gb200`. Keep jobs independent; do not allow one arm's failure to
cancel the other.

### Task 6: Collect Math results and update HTML

**Files:**
- Create: `experiments/qwen235b_thinking_dflash_nemorl_e2e/collect_results.py`
- Create: `experiments/qwen235b_thinking_dflash_nemorl_e2e/README.md`
- Modify: `tests/test_qwen235b_thinking_dflash_nemorl_e2e.py`
- Modify: `scripts/build_pages_index.py`
- Regenerate: `docs/specdec_reports_index_latest.html`
- Regenerate: `public/index.html`

**Interfaces:**
- Produces `math_results.csv`, `swe_results.csv`, and `summary.json`.
- HTML consumes normalized rows with exact W&B URLs.

- [ ] **Step 1: Write failing aggregation tests**

```python
def test_collector_matches_only_same_domain_stage_and_setup() -> None:
    rows = load_fixture_rows("math_matched.json")
    result = summarize(rows)
    assert result["dflash_v2"]["generation_tps_speedup"] == pytest.approx(1.25)
    assert result["dflash_v2"]["e2e_tps_speedup"] == pytest.approx(1.08)
    assert result["dflash_v2"]["metric_window"] == "2-20"


def test_collector_rejects_missing_wandb_url() -> None:
    rows = load_fixture_rows("math_missing_wandb.json")
    with pytest.raises(ResultError, match="W&B"):
        summarize(rows)
```

- [ ] **Step 2: Run tests and verify RED**

```bash
pytest -q tests/test_qwen235b_thinking_dflash_nemorl_e2e.py -k collector
```

Expected: FAIL because `collect_results.py` does not exist.

- [ ] **Step 3: Implement exact metric extraction**

Extract steps 2-20 for Math and the completed common window for SWE. Report
generation tok/s/GPU, generation time, E2E time, E2E tok/s/GPU, acceptance,
mean accepted length, reward, completed steps, job ID, and exact W&B URL.
Never compute speedup without a matched baseline.

- [ ] **Step 4: Add the report section**

Render a `Qwen3-235B Thinking DFlash v2 NeMo-RL E2E` section containing:

- matched baseline and DFlash rows;
- Math K3 and SWE K5 labels;
- stage and metric window;
- four baseline-relative performance metrics;
- acceptance and reward;
- job IDs and exact W&B links;
- failure notes for non-promoted gates.

- [ ] **Step 5: Verify collectors and HTML**

```bash
pytest -q tests/test_qwen235b_thinking_dflash_nemorl_e2e.py
python3 -m py_compile \
  experiments/qwen235b_thinking_dflash_nemorl_e2e/contract.py \
  experiments/qwen235b_thinking_dflash_nemorl_e2e/preflight.py \
  experiments/qwen235b_thinking_dflash_nemorl_e2e/collect_results.py \
  scripts/build_pages_index.py
python3 scripts/build_pages_index.py
git diff --check
```

Expected: all tests pass; generated HTML has balanced table columns and valid
W&B links.

- [ ] **Step 6: Commit**

```bash
git add experiments/qwen235b_thinking_dflash_nemorl_e2e/collect_results.py \
  experiments/qwen235b_thinking_dflash_nemorl_e2e/README.md \
  tests/test_qwen235b_thinking_dflash_nemorl_e2e.py \
  scripts/build_pages_index.py \
  docs/specdec_reports_index_latest.html \
  public/index.html
git commit -s -m "docs: report Qwen235B Thinking DFlash E2E results"
```

### Task 7: Add and run the gated SWE comparison

**Files:**
- Create: `experiments/qwen235b_thinking_dflash_nemorl_e2e/submit_lyris_swe.sh`
- Modify: `tests/test_qwen235b_thinking_dflash_nemorl_e2e.py`
- Update after completion: experiment result artifacts and HTML from Task 6.

**Interfaces:**
- Inputs: `ARM=baseline|dflash_v2`, `STAGE=smoke|pilot`, `MODE=dry-run|test-only|submit`.
- Consumes the existing SWERL launcher, the Task 2 contract, and a passing Math measurement record.

- [ ] **Step 1: Write failing SWE wrapper tests**

Assert baseline and DFlash render the same Thinking verifier, SWE2 datasets,
topology, concurrency, and sampling. Assert DFlash alone renders K5 and six-token
CUDA Graph multiples. Assert W&B is enabled.

- [ ] **Step 2: Run tests and verify RED**

```bash
pytest -q tests/test_qwen235b_thinking_dflash_nemorl_e2e.py -k swe_wrapper
```

Expected: FAIL because `submit_lyris_swe.sh` does not exist.

- [ ] **Step 3: Implement the wrapper**

Wrap
`/lustre/fsw/coreai_dlalgo_llm/users/sna/swerl_fullgrpo_launchers/20260721_swerl_235b_dflash_v1_smoke/run_swerl_235b_dflash_smoke.sh`,
preserve its checksum gate, enable W&B, use per-job venv/cache paths, and require
the bounded collector/empty-input fixes before submission.

- [ ] **Step 4: Run local validation and commit**

```bash
pytest -q tests/test_qwen235b_thinking_dflash_nemorl_e2e.py -k swe_wrapper
bash -n experiments/qwen235b_thinking_dflash_nemorl_e2e/submit_lyris_swe.sh
git add experiments/qwen235b_thinking_dflash_nemorl_e2e/submit_lyris_swe.sh \
  tests/test_qwen235b_thinking_dflash_nemorl_e2e.py
git commit -s -m "feat: add gated Qwen235B DFlash SWE launcher"
git push
```

- [ ] **Step 5: Run matched smoke and pilot tiers**

Run dry-run and `sbatch --test-only` for both arms, submit both one-step arms,
monitor for five minutes, and promote both to three steps only if the matched
smokes pass. Do not submit a twenty-step SWE job under the 5-hour partition
limit.

- [ ] **Step 6: Collect and publish SWE results**

Run `collect_results.py`, regenerate both HTML outputs, verify exact W&B links,
and commit only the experiment result artifacts and report changes.
