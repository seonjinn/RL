# vLLM 0.24 Sync-RL SWE and SPEED-Bench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build reproducible SWE and SPEED-Bench speculative-decoding benchmarks with 32K/64K long-tail Sync-RL workloads across supported Qwen and Nemotron model/method pairs.

**Architecture:** Extract request planning and barrier accounting into a pure Python core shared by the existing offline runner and a new asynchronous SPEED-Bench overlay. Keep official SPEED-Bench execution, forced-length performance replay, and natural-EOS accuracy evaluation as separate cohorts with strict provenance-based matching.

**Tech Stack:** Python 3.12, vLLM 0.24, pytest, Bash, SLURM, Hugging Face Datasets, NVIDIA Model Optimizer SPEED-Bench framework, JSON/JSONL, GitLab Pages.

## Global Constraints

- CUDA graphs use `PIECEWISE`; `enforce_eager=false` remains the default.
- Qwen 64K runs require matched YaRN target and Eagle drafter views.
- Forced-length comparisons require exact planned token equality.
- DFlare and patched PARD-2 use matched same-runtime baselines.
- Official SPEED-Bench and Sync-RL overlay results are never merged.
- Submit only after local tests, commit, push, `TEST_ONLY=true`, and a five-minute early-failure monitor.

---

### Task 1: Request-Plan Core

**Files:**
- Create: `experiments/vllm_024_dynamicsd/sync_rollout_core.py`
- Create: `experiments/vllm_024_dynamicsd/profiles/swe_sync_32k.json`
- Create: `experiments/vllm_024_dynamicsd/profiles/swe_sync_64k.json`
- Modify: `tests/test_vllm024_dynamicsd.py`

**Interfaces:**
- Produces: `LengthBucket`, `RequestPlan`, `ResolvedRequest`, `load_request_plan()`, `resolve_request_plan()`, `validate_context_window()`, and `summarize_barrier_tail()`.
- Consumes: prompt IDs, prompt token lengths, samples per prompt, rollout batch index, and model max length.

- [ ] **Step 1: Write failing tests for the 8:4:3:1 allocation, stable plan hash, exact planned tokens, and context overflow.**

```python
def test_resolve_request_plan_is_deterministic_and_exact() -> None:
    core = load_sync_rollout_core_module()
    plan = core.load_request_plan(EXPERIMENT / "profiles/swe_sync_32k.json")
    first = core.resolve_request_plan(plan, prompt_ids=[f"p{i}" for i in range(16)], samples_per_prompt=4, seed_start=7)
    second = core.resolve_request_plan(plan, prompt_ids=[f"p{i}" for i in range(16)], samples_per_prompt=4, seed_start=7)
    assert first == second
    assert sum(request.max_tokens for request in first) == 589824
    assert {request.max_tokens for request in first} == {4096, 8192, 16384, 32768}
```

- [ ] **Step 2: Run the focused tests and confirm they fail because the core module does not exist.**

Run: `pytest -q tests/test_vllm024_dynamicsd.py -k request_plan`

Expected: FAIL during module loading.

- [ ] **Step 3: Implement frozen typed dataclasses, canonical JSON hashing, weighted allocation, and explicit overflow errors.**

```python
@dataclass(frozen=True)
class ResolvedRequest:
    prompt_id: str
    sample_index: int
    seed: int
    max_tokens: int
    min_tokens: int
    ignore_eos: bool

def validate_context_window(prompt_tokens: int, output_cap: int, max_model_len: int) -> None:
    if prompt_tokens + output_cap > max_model_len:
        raise ValueError(
            f"context overflow: prompt={prompt_tokens} output={output_cap} max={max_model_len}"
        )
```

- [ ] **Step 4: Run focused and full local tests.**

Run: `pytest -q tests/test_vllm024_dynamicsd.py -k request_plan`

Expected: PASS.

Run: `pytest -q tests/test_vllm024_dynamicsd.py`

Expected: all tests pass.

- [ ] **Step 5: Commit the request-plan core.**

```bash
git add experiments/vllm_024_dynamicsd/sync_rollout_core.py experiments/vllm_024_dynamicsd/profiles tests/test_vllm024_dynamicsd.py
git commit -s -m "feat: add Sync-RL long-tail request plans"
```

### Task 2: SWE Barrier and Accuracy Runner

**Files:**
- Modify: `experiments/vllm_024_dynamicsd/benchmark_sync_rollout.py`
- Modify: `experiments/vllm_024_dynamicsd/submit_sync_rollout.sh`
- Create: `experiments/vllm_024_dynamicsd/submit_swe_sync_rollout_matrix.sh`
- Modify: `experiments/vllm_024_dynamicsd/summarize_sync_rollout.py`
- Modify: `tests/test_vllm024_dynamicsd.py`

**Interfaces:**
- Consumes: `RequestPlan` and `ResolvedRequest` from Task 1.
- Produces: per-request provenance, `resolved_request_plan.json`, optional response JSONL, per-bucket statistics, and matched speedup summaries.

- [ ] **Step 1: Add failing tests for no prompt truncation, per-request sampling caps, response persistence, domain-neutral launch validation, and summary match keys.**

```python
def test_tokenize_prompt_rejects_truncation() -> None:
    sync = load_sync_rollout_module()
    with pytest.raises(ValueError, match="prompt exceeds max_prompt_tokens"):
        sync.tokenize_prompt(FakeTokenizer(range(32)), "x", 16, allow_truncation=False)
```

- [ ] **Step 2: Run focused tests and confirm the new arguments are absent.**

Run: `pytest -q tests/test_vllm024_dynamicsd.py -k 'truncation or response_output or swe_sync'`

Expected: FAIL with missing argument or launcher errors.

- [ ] **Step 3: Implement request-plan-aware generation and provenance output.**

```python
sampling = SamplingParams(
    temperature=args.temperature,
    top_p=args.top_p,
    max_tokens=request.max_tokens,
    min_tokens=request.min_tokens,
    ignore_eos=request.ignore_eos,
    seed=request.seed,
)
```

- [ ] **Step 4: Add strict summary matching on runtime image, model view hash, prompt hash, request-plan hash, graph mode, topology, sampling, and exact output work.**

```python
MATCH_FIELDS = (
    "runtime_image_sha256",
    "model_config_hash",
    "prompt_set_hash",
    "request_plan_hash",
    "cudagraph_mode",
    "tensor_parallel_size",
    "temperature",
    "top_p",
)
```

- [ ] **Step 5: Run Python and shell validation.**

Run: `pytest -q tests/test_vllm024_dynamicsd.py`

Expected: all tests pass.

Run: `bash -n experiments/vllm_024_dynamicsd/submit_sync_rollout.sh experiments/vllm_024_dynamicsd/submit_swe_sync_rollout_matrix.sh`

Expected: exit 0.

- [ ] **Step 6: Commit the SWE runner.**

```bash
git add experiments/vllm_024_dynamicsd/benchmark_sync_rollout.py experiments/vllm_024_dynamicsd/submit_sync_rollout.sh experiments/vllm_024_dynamicsd/submit_swe_sync_rollout_matrix.sh experiments/vllm_024_dynamicsd/summarize_sync_rollout.py tests/test_vllm024_dynamicsd.py
git commit -s -m "feat: add SWE Sync-RL barrier benchmark"
```

### Task 3: Model and Drafter Matrix

**Files:**
- Create: `experiments/vllm_024_dynamicsd/model_method_matrix.json`
- Modify: `experiments/vllm_024_dynamicsd/submit_swe_sync_rollout_matrix.sh`
- Modify: `experiments/vllm_024_dynamicsd/submit_nemotron_sync_rl_mtp_matrix.sh`
- Modify: `tests/test_vllm024_dynamicsd.py`

**Interfaces:**
- Consumes: pinned checkpoint paths and method compatibility records.
- Produces: supported job rows and structured unsupported reasons.

- [ ] **Step 1: Write failing tests that validate unique model-method-profile keys and reject unsupported pairs before submission.**

```python
def test_large_model_matrix_rejects_qwen8_only_dflash() -> None:
    matrix = json.loads((EXPERIMENT / "model_method_matrix.json").read_text())
    qwen32 = next(item for item in matrix["models"] if item["key"] == "qwen32")
    assert qwen32["methods"]["dflash"]["status"] == "unsupported"
```

- [ ] **Step 2: Run the focused test and confirm the manifest is absent.**

Run: `pytest -q tests/test_vllm024_dynamicsd.py -k model_method_matrix`

Expected: FAIL because `model_method_matrix.json` does not exist.

- [ ] **Step 3: Add pinned targets, drafters, topologies, context handling, and unsupported reasons.**

```json
{
  "schema_version": 1,
  "models": [
    {
      "key": "qwen30ba3b",
      "methods": {
        "eagle3": {"status": "supported"},
        "pard": {"status": "integration"},
        "pard2": {"status": "unsupported", "reason": "missing target-dimension checkpoint"}
      }
    }
  ]
}
```

- [ ] **Step 4: Make launchers emit only supported rows and preserve unsupported rows in `jobs.tsv` with status `UNSUPPORTED`.**

Run: `DRY_RUN=true experiments/vllm_024_dynamicsd/submit_swe_sync_rollout_matrix.sh`

Expected: rendered Qwen Eagle jobs, Nemotron MTP jobs, and explicit unsupported method rows without `sbatch` calls.

- [ ] **Step 5: Run tests and commit.**

Run: `pytest -q tests/test_vllm024_dynamicsd.py`

Expected: all tests pass.

```bash
git add experiments/vllm_024_dynamicsd/model_method_matrix.json experiments/vllm_024_dynamicsd/submit_swe_sync_rollout_matrix.sh experiments/vllm_024_dynamicsd/submit_nemotron_sync_rl_mtp_matrix.sh tests/test_vllm024_dynamicsd.py
git commit -s -m "feat: define supported SpecDec model matrix"
```

### Task 4: SPEED-Bench Dataset Adapter

**Files:**
- Create: `experiments/vllm_024_dynamicsd/speedbench_dataset.py`
- Create: `experiments/vllm_024_dynamicsd/stage_speedbench.sh`
- Modify: `tests/test_vllm024_dynamicsd.py`

**Interfaces:**
- Produces: pinned prepared-data manifest, balanced throughput selections, preserved multi-turn records, nominal and actual ISL metadata.
- Consumes: upstream Model Optimizer `prepare_data.py` outputs.

- [ ] **Step 1: Write failing tests for balanced 6/5/5 batch allocation, deterministic selection, multi-turn preservation, and masked-row rejection.**

```python
def test_speedbench_batches_balance_entropy_classes() -> None:
    adapter = load_speedbench_dataset_module()
    batches = adapter.select_sync_overlay_rows(fake_speed_rows(), seed=1234)
    assert [adapter.count_categories(batch) for batch in batches] == [
        {"low_entropy": 6, "mixed": 5, "high_entropy": 5},
        {"low_entropy": 5, "mixed": 6, "high_entropy": 5},
        {"low_entropy": 5, "mixed": 5, "high_entropy": 6},
    ]
```

- [ ] **Step 2: Run focused tests and verify failure.**

Run: `pytest -q tests/test_vllm024_dynamicsd.py -k speedbench_dataset`

Expected: FAIL because the adapter is absent.

- [ ] **Step 3: Implement typed records, deterministic balancing, canonical hashes, and prepared-data validation.**

```python
@dataclass(frozen=True)
class SpeedBenchRecord:
    question_id: str
    category: str
    turns: tuple[str, ...]
    source: str
    nominal_isl: int
```

- [ ] **Step 4: Add a staging script that pins both repositories and checksums every resolved parquet file.**

Run: `bash -n experiments/vllm_024_dynamicsd/stage_speedbench.sh`

Expected: exit 0.

- [ ] **Step 5: Run tests and commit.**

Run: `pytest -q tests/test_vllm024_dynamicsd.py`

Expected: all tests pass.

```bash
git add experiments/vllm_024_dynamicsd/speedbench_dataset.py experiments/vllm_024_dynamicsd/stage_speedbench.sh tests/test_vllm024_dynamicsd.py
git commit -s -m "feat: add pinned SPEED-Bench adapter"
```

### Task 5: SPEED-Bench Official and Sync-RL Runners

**Files:**
- Create: `experiments/vllm_024_dynamicsd/benchmark_speedbench_sync_rollout.py`
- Create: `experiments/vllm_024_dynamicsd/submit_speedbench_k_calibration.sh`
- Create: `experiments/vllm_024_dynamicsd/submit_nemotron_speedbench_sync_mtp_matrix.sh`
- Create: `experiments/vllm_024_dynamicsd/summarize_speedbench_sync_rollout.py`
- Modify: `tests/test_vllm024_dynamicsd.py`

**Interfaces:**
- Consumes: prepared SPEED-Bench records, request-plan core, and model-method matrix.
- Produces: official runner manifests, asynchronous barrier metrics, K calibration summaries, and separate official/overlay report rows.

- [ ] **Step 1: Add failing tests for preserved token IDs, async barrier completion times, warmup shape, K-tier reachability, and official/overlay separation.**

```python
def test_speedbench_summary_never_merges_official_and_overlay() -> None:
    summary = load_speedbench_summary_module()
    with pytest.raises(ValueError, match="cohort mismatch"):
        summary.compare_rows(fake_official_row(), fake_overlay_row())
```

- [ ] **Step 2: Run focused tests and verify failure.**

Run: `pytest -q tests/test_vllm024_dynamicsd.py -k speedbench_sync`

Expected: FAIL because the runner and summary modules are absent.

- [ ] **Step 3: Implement AsyncLLM request streaming with an explicit gather barrier and position-window acceptance accounting.**

```python
completed = await asyncio.gather(
    *(run_one_request(engine, request, sampling) for request in batch)
)
barrier_finished_at = max(item.finished_at for item in completed)
```

- [ ] **Step 4: Implement calibration launchers for concurrency 1, 8, 32, and 64 before the full sweep.**

Run: `DRY_RUN=true experiments/vllm_024_dynamicsd/submit_speedbench_k_calibration.sh`

Expected: separate baseline and fixed-K jobs with pinned ISL, model, method, and repeat metadata.

- [ ] **Step 5: Run tests and shell validation.**

Run: `pytest -q tests/test_vllm024_dynamicsd.py`

Expected: all tests pass.

Run: `bash -n experiments/vllm_024_dynamicsd/submit_speedbench_k_calibration.sh experiments/vllm_024_dynamicsd/submit_nemotron_speedbench_sync_mtp_matrix.sh`

Expected: exit 0.

- [ ] **Step 6: Commit the SPEED-Bench runners.**

```bash
git add experiments/vllm_024_dynamicsd/benchmark_speedbench_sync_rollout.py experiments/vllm_024_dynamicsd/submit_speedbench_k_calibration.sh experiments/vllm_024_dynamicsd/submit_nemotron_speedbench_sync_mtp_matrix.sh experiments/vllm_024_dynamicsd/summarize_speedbench_sync_rollout.py tests/test_vllm024_dynamicsd.py
git commit -s -m "feat: add SPEED-Bench Sync-RL evaluation"
```

### Task 6: Documentation, Verification, and Submission

**Files:**
- Modify: `experiments/vllm_024_dynamicsd/README.md`
- Modify: `experiments/vllm_024_dynamicsd/PLAN.md`
- Modify: `experiments/vllm_024_dynamicsd/report/README.md`
- Modify: `public/reports/vllm_standalone_results_latest.html`

**Interfaces:**
- Consumes: validated dry-run manifests and result summaries.
- Produces: reproducible launch instructions and a report that distinguishes supported, pending, failed, and unsupported cells.

- [ ] **Step 1: Document cohort boundaries, exact model/method support, 32K/64K profiles, pinned SPEED-Bench revisions, and interpretation limits.**

- [ ] **Step 2: Run the complete local verification suite.**

Run: `pytest -q tests/test_vllm024_dynamicsd.py`

Expected: all tests pass.

Run: `bash -n experiments/vllm_024_dynamicsd/*.sh`

Expected: exit 0.

- [ ] **Step 3: Commit and push before any remote submission.**

```bash
git add experiments/vllm_024_dynamicsd/README.md experiments/vllm_024_dynamicsd/PLAN.md experiments/vllm_024_dynamicsd/report/README.md public/reports/vllm_standalone_results_latest.html
git commit -s -m "docs: describe SWE and SPEED-Bench SpecDec matrix"
git push origin codex/vllm024-dynamicsd
```

- [ ] **Step 4: Validate scheduling with no submissions.**

Run remotely on each selected cluster after `git pull --ff-only`:

```bash
CLUSTER=lyris TEST_ONLY=true ./experiments/vllm_024_dynamicsd/submit_swe_sync_rollout_matrix.sh
CLUSTER=ptyche TEST_ONLY=true ./experiments/vllm_024_dynamicsd/submit_nemotron_speedbench_sync_mtp_matrix.sh
```

Expected: all requested shapes pass `sbatch --test-only`; unsupported cells remain manifest-only.

- [ ] **Step 5: Submit canaries and monitor for five minutes.**

```bash
CLUSTER=lyris SMOKE=true ./experiments/vllm_024_dynamicsd/submit_swe_sync_rollout_matrix.sh
CLUSTER=ptyche SMOKE=true ./experiments/vllm_024_dynamicsd/submit_nemotron_speedbench_sync_mtp_matrix.sh
```

Expected: jobs enter RUNNING or remain validly PENDING without immediate import, checkpoint, context, OOM, or speculative-counter failures.

- [ ] **Step 6: Pull only manifests, logs, and result artifacts, then regenerate summaries and the HTML report.**

Run: `python experiments/vllm_024_dynamicsd/summarize_sync_rollout.py --help`

Expected: exit 0 and documented summary arguments.
