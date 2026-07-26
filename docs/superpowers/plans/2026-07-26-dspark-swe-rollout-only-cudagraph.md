# DSpark SWE Rollout-Only CUDA Graph Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run DSpark B8/B16 V1/V2 on the canonical Qwen3-235B NemoGym SWE2 rollout-only workload with CUDA Graph enabled and compare valid throughput and acceptance metrics against non-speculative decoding and DFlash V1/V2.

**Architecture:** Add a focused experiment package that owns the immutable matrix, validates checkpoint provenance, renders exact Hydra overrides for the existing Lyris NemoGym launcher, and collects normalized results. Transfer the four DSpark checkpoints from AWS-DFW to Lyris with checksum verification, gate the matrix with a graph-enabled DSpark B8 V1 smoke, then submit the remaining variants through the unchanged canonical launcher.

**Tech Stack:** Python 3.12, pytest, Bash, SLURM, vLLM 0.25.1, NeMo-RL NemoGym rollout benchmark, PBSS/rclone, HTML.

## Global Constraints

- Target model is exactly `Qwen/Qwen3-235B-A22B-Thinking-2507`.
- Dataset is exactly `data/swe2/val-mini3.jsonl`, containing Astropy instances `12907`, `13236`, and `13398`.
- Use three prompts, one generation per prompt, natural EOS, at most eight agent turns, `temperature=1.0`, `top_p=1.0`, and seed `42`.
- Use one Lyris GB200 node, four GPUs, target TP4, PP1, EP1, BF16, and vLLM 0.25.1.
- Set `policy.generation.vllm_cfg.enforce_eager=false`.
- Set `compilation_config.cudagraph_mode=FULL` and `cudagraph_capture_sizes=[6,12,24,48,96]`.
- Never silently fall back to eager mode.
- Keep draft `max_model_len=4096` for the primary DFlash/DSpark comparison so speculative decoding has the same active context envelope.
- DFlash uses K5; DSpark B8 uses K8; DSpark B16 uses K16.
- Newly measured speedups use the matched non-speculative baseline denominator unless the existing job `2451569` passes the full provenance equality check.
- Preserve the user's unrelated dirty-worktree changes.

---

### Task 1: Immutable Benchmark Matrix

**Files:**
- Create: `experiments/dspark_swe_rollout_benchmark/__init__.py`
- Create: `experiments/dspark_swe_rollout_benchmark/matrix.json`
- Create: `experiments/dspark_swe_rollout_benchmark/contract.py`
- Create: `tests/test_dspark_swe_rollout_contract.py`

**Interfaces:**
- Consumes: the exact variant and runtime constraints in the approved design.
- Produces: `load_contract(path: Path) -> BenchmarkContract` and `render_hydra_overrides(contract: BenchmarkContract, variant_name: str) -> list[str]`.

- [ ] **Step 1: Write the failing contract tests**

```python
from pathlib import Path

import pytest

from experiments.dspark_swe_rollout_benchmark.contract import (
    load_contract,
    render_hydra_overrides,
)


MATRIX = Path("experiments/dspark_swe_rollout_benchmark/matrix.json")


def test_matrix_contains_exact_primary_variants() -> None:
    contract = load_contract(MATRIX)
    assert list(contract.variants) == [
        "baseline",
        "dflash_v1_k5",
        "dflash_v2_k5",
        "dspark_b8_v1",
        "dspark_b8_v2",
        "dspark_b16_v1",
        "dspark_b16_v2",
    ]


def test_dspark_b8_v1_renders_full_cudagraph_overrides() -> None:
    contract = load_contract(MATRIX)
    overrides = render_hydra_overrides(contract, "dspark_b8_v1")
    assert "policy.generation.vllm_cfg.enforce_eager=false" in overrides
    assert (
        "++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=FULL"
        in overrides
    )
    assert any("cudagraph_capture_sizes=[6,12,24,48,96]" in value for value in overrides)
    assert any("speculative_config.method=dspark" in value for value in overrides)
    assert any("speculative_config.num_speculative_tokens=8" in value for value in overrides)
    assert any("speculative_config.max_model_len=4096" in value for value in overrides)


def test_contract_rejects_eager_mode(tmp_path: Path) -> None:
    bad = tmp_path / "matrix.json"
    bad.write_text(MATRIX.read_text().replace('"enforce_eager": false', '"enforce_eager": true'))
    with pytest.raises(ValueError, match="enforce_eager"):
        load_contract(bad)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
pytest -q tests/test_dspark_swe_rollout_contract.py
```

Expected: collection fails because the package and contract module do not exist.

- [ ] **Step 3: Add the exact matrix**

The JSON must encode:

```json
{
  "schema_version": 1,
  "target_model": "Qwen/Qwen3-235B-A22B-Thinking-2507",
  "dataset": "data/swe2/val-mini3.jsonl",
  "num_prompts": 3,
  "num_generations": 1,
  "temperature": 1.0,
  "top_p": 1.0,
  "seed": 42,
  "enforce_eager": false,
  "cudagraph_mode": "FULL",
  "cudagraph_capture_sizes": [6, 12, 24, 48, 96],
  "draft_max_model_len": 4096,
  "variants": {
    "baseline": {"method": "baseline", "k": 0, "checkpoint": null},
    "dflash_v1_k5": {"method": "dflash", "k": 5, "checkpoint": "/lustre/fsw/coreai_dlalgo_llm/users/sna/drafters/dflash_235bthink_v1"},
    "dflash_v2_k5": {"method": "dflash", "k": 5, "checkpoint": "/lustre/fsw/coreai_dlalgo_llm/users/sna/drafters/dflash_235bthink_v2"},
    "dspark_b8_v1": {"method": "dspark", "k": 8, "checkpoint": "/lustre/fsw/coreai_dlalgo_llm/users/sna/drafters/dspark_235b_v2mix_b8_v1"},
    "dspark_b8_v2": {"method": "dspark", "k": 8, "checkpoint": "/lustre/fsw/coreai_dlalgo_llm/users/sna/drafters/dspark_235b_v2mix_b8_v2"},
    "dspark_b16_v1": {"method": "dspark", "k": 16, "checkpoint": "/lustre/fsw/coreai_dlalgo_llm/users/sna/drafters/dspark_235b_v2mix_b16_v1"},
    "dspark_b16_v2": {"method": "dspark", "k": 16, "checkpoint": "/lustre/fsw/coreai_dlalgo_llm/users/sna/drafters/dspark_235b_v2mix_b16_v2"}
  }
}
```

- [ ] **Step 4: Implement strict dataclasses and override rendering**

`contract.py` must reject:

- non-FULL graph mode;
- eager mode;
- capture sizes other than `[6, 12, 24, 48, 96]`;
- DFlash K other than 5;
- B8/B16 K mismatches;
- missing absolute checkpoint paths for speculative variants.

The renderer must add method, model, K, draft TP1, draft max model length 4096,
FLASH_ATTN, FlashInfer autotune false, metrics logger true, and the exact graph
settings. Baseline must not render a `speculative_config`.

- [ ] **Step 5: Run tests and verify GREEN**

Run:

```bash
pytest -q tests/test_dspark_swe_rollout_contract.py
python3 -m py_compile experiments/dspark_swe_rollout_benchmark/contract.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add experiments/dspark_swe_rollout_benchmark/__init__.py \
  experiments/dspark_swe_rollout_benchmark/matrix.json \
  experiments/dspark_swe_rollout_benchmark/contract.py \
  tests/test_dspark_swe_rollout_contract.py
git commit -s -m "feat: define DSpark SWE rollout benchmark contract"
```

---

### Task 2: Checkpoint Provenance and Transfer Manifest

**Files:**
- Create: `experiments/dspark_swe_rollout_benchmark/checkpoints.json`
- Create: `experiments/dspark_swe_rollout_benchmark/checkpoint_manifest.py`
- Create: `tests/test_dspark_checkpoint_manifest.py`

**Interfaces:**
- Consumes: four AWS checkpoint directories and their three serving files.
- Produces: `build_manifest(source: Path, label: str) -> dict[str, object]` and `verify_manifest(destination: Path, manifest: Mapping[str, object]) -> None`.

- [ ] **Step 1: Write failing checksum tests**

```python
import hashlib
from pathlib import Path

import pytest

from experiments.dspark_swe_rollout_benchmark.checkpoint_manifest import (
    build_manifest,
    verify_manifest,
)


def test_manifest_records_only_serving_files(tmp_path: Path) -> None:
    for name, content in {
        "config.json": b"{}",
        "config.py": b"class C: pass\n",
        "model.safetensors": b"weights",
        "optimizer_state_dict.pt": b"do-not-transfer",
    }.items():
        (tmp_path / name).write_bytes(content)
    manifest = build_manifest(tmp_path, "dspark_b8_v1")
    assert list(manifest["files"]) == ["config.json", "config.py", "model.safetensors"]
    assert manifest["files"]["model.safetensors"]["sha256"] == hashlib.sha256(b"weights").hexdigest()


def test_verify_rejects_corrupted_destination(tmp_path: Path) -> None:
    for name in ("config.json", "config.py", "model.safetensors"):
        (tmp_path / name).write_text(name)
    manifest = build_manifest(tmp_path, "dspark_b8_v1")
    (tmp_path / "model.safetensors").write_text("corrupt")
    with pytest.raises(ValueError, match="model.safetensors"):
        verify_manifest(tmp_path, manifest)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
pytest -q tests/test_dspark_checkpoint_manifest.py
```

Expected: import failure because `checkpoint_manifest.py` does not exist.

- [ ] **Step 3: Implement streaming SHA256 and strict file inventory**

Use 8 MiB chunks and refuse symlinks for the destination verification step.
Include source path, label, file size, SHA256, timestamp, and source
`training_state.json` global step in each manifest.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
pytest -q tests/test_dspark_checkpoint_manifest.py
```

Expected: all tests pass.

- [ ] **Step 5: Generate AWS source manifests read-only**

Run the module over:

```text
/lustre/fsw/portfolios/nemotron/users/sna/dflash_training/output/dspark_235b_v3_b8/0
/lustre/fsw/portfolios/nemotron/users/sna/dflash_training/output/dspark_235b_v3_b8/1
/lustre/fsw/portfolios/nemotron/users/sna/dflash_training/output/dspark_235b_v3_b16/0
/lustre/fsw/portfolios/nemotron/users/sna/dflash_training/output/dspark_235b_v3_b16/1
```

Store the generated metadata in `checkpoints.json`. Do not include optimizer or
scheduler state in the transfer.

- [ ] **Step 6: Commit**

```bash
git add experiments/dspark_swe_rollout_benchmark/checkpoints.json \
  experiments/dspark_swe_rollout_benchmark/checkpoint_manifest.py \
  tests/test_dspark_checkpoint_manifest.py
git commit -s -m "feat: validate DSpark serving checkpoint provenance"
```

---

### Task 3: Reproducible Lyris Launcher Wrapper

**Files:**
- Create: `experiments/dspark_swe_rollout_benchmark/submit_lyris.sh`
- Create: `tests/test_dspark_swe_rollout_submit.py`

**Interfaces:**
- Consumes: `matrix.json`, `contract.py`, canonical remote launcher `experiments/nemogym_swe1_specdec/submit_lyris.sh`.
- Produces: dry-run command/provenance output and submitted SLURM job IDs.

- [ ] **Step 1: Write failing dry-run tests**

The test invokes:

```bash
bash experiments/dspark_swe_rollout_benchmark/submit_lyris.sh \
  --test-only --variant dspark_b8_v1
```

Assert output contains:

```text
method=dspark
num_speculative_tokens=8
enforce_eager=false
cudagraph_mode=FULL
cudagraph_capture_sizes=[6,12,24,48,96]
num_prompts=3
num_generations=1
```

Also assert the script exits 2 for an unknown variant and never emits
`enforce_eager=true`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
pytest -q tests/test_dspark_swe_rollout_submit.py
```

Expected: failure because `submit_lyris.sh` does not exist.

- [ ] **Step 3: Implement the wrapper**

The wrapper must:

1. render overrides locally with `contract.py`;
2. support `--test-only` without SSH or state changes;
3. preflight Lyris checkpoint files and checksum manifest;
4. verify the canonical launcher SHA256
   `683807778c9435291b2524cc51a53a2b6d2f8ce89c5213245e71ca6227be4b1d`;
5. verify remote repo HEAD `b9c29565bde277e997eb969af9cc47da55ef4d16`;
6. check `squeue --start` before submission;
7. invoke `MODE=submit VARIANT=baseline` with `CONFIG`, `DATA`,
   `NUM_PROMPTS=3`, `NUM_GENS=1`, `METRICS=true`, a unique `TAG`, and
   `EXTRA_OVERRIDES`;
8. append job ID, run directory, matrix SHA256, checkpoint SHA256, and exact
   command to `latest_jobs.tsv`.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
pytest -q tests/test_dspark_swe_rollout_submit.py
bash -n experiments/dspark_swe_rollout_benchmark/submit_lyris.sh
```

Expected: all tests pass and shell syntax is valid.

- [ ] **Step 5: Commit**

```bash
git add experiments/dspark_swe_rollout_benchmark/submit_lyris.sh \
  tests/test_dspark_swe_rollout_submit.py
git commit -s -m "feat: add DSpark SWE rollout submission wrapper"
```

---

### Task 4: Result Collector and Speedup Validation

**Files:**
- Create: `experiments/dspark_swe_rollout_benchmark/collect_results.py`
- Create: `tests/test_dspark_swe_rollout_collect.py`
- Create at runtime: `experiments/dspark_swe_rollout_benchmark/report/data/results.csv`
- Create at runtime: `experiments/dspark_swe_rollout_benchmark/report/data/results.json`

**Interfaces:**
- Consumes: copied Ray driver logs, SLURM state, provenance files, and vLLM metrics.
- Produces: normalized rows with throughput, speedups, acceptance, mean accepted length, token totals, graph evidence, and validity status.

- [ ] **Step 1: Write failing collector tests with synthetic log fixtures**

Test the following behavior:

```python
assert row.output_tokens == 37033
assert row.model_call_seconds == pytest.approx(277.999)
assert row.tok_s == pytest.approx(133.21, rel=1e-3)
assert row.speedup_vs_baseline == pytest.approx(1.524, rel=1e-3)
assert row.cudagraph_mode == "FULL"
assert row.cudagraph_capture_sizes == (6, 12, 24, 48, 96)
```

Add rejection cases for:

- incomplete trajectories;
- missing graph evidence;
- eager fallback;
- fatal engine errors;
- missing speculative metrics for a DSpark row.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
pytest -q tests/test_dspark_swe_rollout_collect.py
```

Expected: import failure because the collector does not exist.

- [ ] **Step 3: Implement log parsing and matched baseline logic**

Compute:

```text
throughput = sum(completion_tokens) / sum(model_call_seconds)
speedup_vs_baseline = variant_throughput / matched_baseline_throughput
speedup_vs_dflash_v2 = variant_throughput / dflash_v2_throughput
```

Only emit a numeric speedup if target model, dataset hash, prompts,
generations, sampling, GPU topology, runtime commit, container, graph mode,
capture sizes, and draft active-context envelope match.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
pytest -q tests/test_dspark_swe_rollout_collect.py
python3 -m py_compile experiments/dspark_swe_rollout_benchmark/collect_results.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/dspark_swe_rollout_benchmark/collect_results.py \
  tests/test_dspark_swe_rollout_collect.py
git commit -s -m "feat: collect DSpark rollout speedup metrics"
```

---

### Task 5: Transfer, Smoke, and Full Matrix Execution

**Files:**
- Modify at runtime: `experiments/dspark_swe_rollout_benchmark/latest_jobs.tsv`
- Create at runtime: `experiments/dspark_swe_rollout_benchmark/report/STATUS.md`

**Interfaces:**
- Consumes: committed experiment scripts, AWS source checkpoints, PBSS transfer tools, Lyris launcher.
- Produces: checksum-verified Lyris checkpoints and live SLURM jobs.

- [ ] **Step 1: Push the exact implementation commits**

Run:

```bash
git fetch origin
git rev-list --left-right --count origin/main...HEAD
git push origin main
```

Expected before push: no remote-only commits. Expected after push: `0 0`.

- [ ] **Step 2: Submit AWS PBSS uploads**

Use the `pdx-slurm-transfer` upload helper for each serving-only checkpoint
bundle with account `nemotron_sw_post` and partition `cpu_datamover`. Capture
all four upload job IDs.

- [ ] **Step 3: Monitor uploads and submit Lyris downloads**

Wait for `COMPLETED 0:0`, then download into the four exact Lyris destinations
from `matrix.json` using account `coreai_dlalgo_llm` and partition `cpu`.
Verify every file against `checkpoints.json`.

- [ ] **Step 4: Run the scheduler preflight**

Run `submit_lyris.sh --test-only` for all DSpark variants, followed by the
cluster scheduling check. Do not submit if checkpoint validation, launcher
hash, repo HEAD, or graph contract differs.

- [ ] **Step 5: Submit DSpark B8 V1 CUDA Graph smoke**

Use one prompt and one generation, but retain:

```text
enforce_eager=false
cudagraph_mode=FULL
cudagraph_capture_sizes=[6,12,24,48,96]
```

Do not use an eager-only smoke.

- [ ] **Step 6: Monitor the smoke for at least five minutes**

Check:

```bash
squeue -j <jobid> -o '%i %T %M %l %R'
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed -n
grep -nE 'Traceback|ERROR|EngineDead|CUDA out of memory|NCCL|fallback|enforce_eager' <logs>
grep -nE 'CUDAGraph|capture|dspark|acceptance' <logs>
```

Proceed only after model load, DSpark activation, graph capture, non-empty
generation, and speculative metrics are proven.

- [ ] **Step 7: Submit DSpark B8/B16 V1/V2**

Submit the four production jobs. Reuse DFlash jobs `2451570` and `2477286`
only if the collector confirms exact provenance equality with baseline
`2451569`; otherwise submit a fresh matched baseline and DFlash rows.

- [ ] **Step 8: Monitor every new job for at least five minutes**

Record job IDs, queue states, start estimates, run directories, and any early
failure evidence in `STATUS.md`.

- [ ] **Step 9: Commit operational manifests**

```bash
git add experiments/dspark_swe_rollout_benchmark/latest_jobs.tsv \
  experiments/dspark_swe_rollout_benchmark/report/STATUS.md
git commit -s -m "docs: record DSpark SWE rollout jobs"
git push origin main
```

---

### Task 6: Final Report and HTML Update

**Files:**
- Create: `experiments/dspark_swe_rollout_benchmark/README.md`
- Create: `experiments/dspark_swe_rollout_benchmark/report/REPORT.md`
- Modify: `docs/dflash_drafter_training.html`
- Modify: `experiments/dflash_loss_ab/report/data/rollout_4way.csv`

**Interfaces:**
- Consumes: validated `results.csv`, `results.json`, job manifests, and failure logs.
- Produces: user-facing comparison and reproducible experiment documentation.

- [ ] **Step 1: Collect completed rows**

Run:

```bash
python3 experiments/dspark_swe_rollout_benchmark/collect_results.py \
  --jobs experiments/dspark_swe_rollout_benchmark/latest_jobs.tsv \
  --csv experiments/dspark_swe_rollout_benchmark/report/data/results.csv \
  --json experiments/dspark_swe_rollout_benchmark/report/data/results.json
```

Expected: completed valid rows have throughput, graph evidence, token totals,
and speculative metrics. Partial or failed rows contain no invented speedup.

- [ ] **Step 2: Write the experiment report**

Include:

- exact cohort and graph contract;
- checkpoint SHA256 and training step;
- throughput and speedup versus baseline/DFlash;
- acceptance and mean accepted length;
- generated token totals and trajectory completion;
- all failures, hypotheses, disproofs, and fixes;
- clear separation of throughput speedup from identical-work latency speedup.

- [ ] **Step 3: Update the existing HTML page**

Add a focused DSpark section after the DFlash rollout table. Preserve all
existing content and unrelated user edits. Include partial-state labels until
all required rows validate.

- [ ] **Step 4: Run final verification**

Run:

```bash
pytest -q \
  tests/test_dspark_swe_rollout_contract.py \
  tests/test_dspark_checkpoint_manifest.py \
  tests/test_dspark_swe_rollout_submit.py \
  tests/test_dspark_swe_rollout_collect.py
python3 -m py_compile \
  experiments/dspark_swe_rollout_benchmark/contract.py \
  experiments/dspark_swe_rollout_benchmark/checkpoint_manifest.py \
  experiments/dspark_swe_rollout_benchmark/collect_results.py
bash -n experiments/dspark_swe_rollout_benchmark/submit_lyris.sh
python3 scripts/build_latest_specdec_html_pages.py
python3 scripts/build_pages_index.py
git diff --check
```

Expected: all tests and compilers pass, HTML builders complete, and no
whitespace errors are reported.

- [ ] **Step 5: Commit and push the final report**

```bash
git add experiments/dspark_swe_rollout_benchmark \
  experiments/dflash_loss_ab/report/data/rollout_4way.csv \
  docs/dflash_drafter_training.html
git commit -s -m "docs: report DSpark SWE rollout speedups"
git push origin main
```
