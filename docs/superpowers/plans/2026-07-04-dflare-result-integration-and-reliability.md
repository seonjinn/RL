# DFlare Result Integration and Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish every completed July 3-4 vLLM 0.24 result and make AngelSlim DFlare result collection CPU-only, partial-result-safe, and retryable at long context.

**Architecture:** A pure Python transport module converts AngelSlim response objects into compact CPU-only records and atomically writes rank-local partial JSON before the distributed collective. A separate vLLM-native normalizer expands each benchmark JSON into one row per batch size, performs exact setup-key baseline matching, and renders profile-separated HTML without mixing AngelSlim and vLLM runtimes.

**Tech Stack:** Python 3.12, pandas, pytest, AngelSlim/PyTorch distributed, Bash/SLURM, static HTML.

## Global Constraints

- Compare only exact matches for runtime, model, domain, temperature, top-p, batch size, ISL, OSL, context profile, position encoding, CUDA graph mode, and setup.
- Never compute speedup across AngelSlim and vLLM-native runtimes.
- Keep completed, partial, timeout, OOM, and failed rows visibly distinct.
- Use `/lustre/fsw/coreai_dlalgo_llm/users/sna` for Lyris artifacts.
- Use Lyris `--segment=1` and no `--gres` for one-node jobs.
- Commit and push before submitting jobs; monitor every new job for five minutes.

---

### Task 1: CPU-Only DFlare Result Transport

**Files:**
- Create: `experiments/vllm_024_dynamicsd/angelslim_dflare_transport.py`
- Create: `tests/test_angelslim_dflare_transport.py`

**Interfaces:**
- Produces: `compact_response_map(responses: Mapping[int, Any]) -> dict[int, CompactResponse]`
- Produces: `write_rank_partial(path: Path, rank: int, responses: Sequence[Mapping[int, CompactResponse]]) -> Path`
- Produces: `CompactResponse(time_per_output_token: float, acceptance_lengths: list[int], num_input_tokens: int, num_output_tokens: int)`

- [ ] **Step 1: Write failing transport tests**

Test that compact records retain scalar timing and acceptance data, derive token counts from a fake CUDA-like tensor shape, contain no `output_ids`, serialize with `json.dumps`, and atomically create `result.json.rank2.partial.json`.

- [ ] **Step 2: Run the focused test and confirm failure**

Run: `python3 -m pytest -q tests/test_angelslim_dflare_transport.py`

Expected: import failure because `angelslim_dflare_transport.py` does not exist.

- [ ] **Step 3: Implement the pure transport module**

Use a frozen dataclass, integer/float normalization, JSON conversion helpers,
and `Path.replace()` for atomic writes. Do not import torch.

- [ ] **Step 4: Run the focused test**

Run: `python3 -m pytest -q tests/test_angelslim_dflare_transport.py`

Expected: all transport tests pass.

### Task 2: Patch and Stage the AngelSlim Runner

**Files:**
- Create: `experiments/vllm_024_dynamicsd/patches/angelslim_compact_result_transport.patch`
- Modify: `experiments/vllm_024_dynamicsd/stage_extended_method_assets_in_container.sh`
- Modify: `tests/test_vllm024_dynamicsd.py`

**Interfaces:**
- Consumes: Task 1 transport module.
- Produces: patched `tools/dflash_benchmark.py` that gathers only compact CPU records and writes rank partial files before `_dist_gather`.

- [ ] **Step 1: Add failing staging assertions**

Assert the staging script copies `angelslim_dflare_transport.py`, applies
`angelslim_compact_result_transport.patch`, and that the patch replaces
`responses.append(response)` with compact records plus a pre-gather partial
write.

- [ ] **Step 2: Run the focused staging tests and confirm failure**

Run: `python3 -m pytest -q tests/test_vllm024_dynamicsd.py`

Expected: assertions fail because the new transport and patch are not staged.

- [ ] **Step 3: Add the minimal AngelSlim patch**

Import the staged transport module, compact each response after local text
decoding, write a rank partial JSON immediately after the local loop, gather
compact records, and keep the existing final result JSON fields unchanged.

- [ ] **Step 4: Validate patch application**

Run `patch --dry-run` against the pinned AngelSlim commit
`6a97dab2f17c0a3c031065329f092c4f61108a6f`, then run the focused pytest file.

Expected: dry-run and tests pass.

### Task 3: Normalize vLLM-Native Profile Results

**Files:**
- Create: `scripts/vllm024_profile_report.py`
- Create: `tests/test_vllm024_profile_report.py`

**Interfaces:**
- Produces: `load_profile_results(paths: Iterable[Path]) -> pd.DataFrame`
- Produces: `match_profile_baselines(rows: pd.DataFrame) -> pd.DataFrame`
- Produces: `render_profile_section(rows: pd.DataFrame) -> str`

- [ ] **Step 1: Write failing parser tests**

Cover baseline and speculative JSON fixtures for Native 32K, YaRN 64K, and
YaRN total-128K. Verify one row per batch size, exact setup matching,
throughput speedup, latency speedup, acceptance, mean accepted length, K, and
unmatched-baseline behavior.

- [ ] **Step 2: Run the focused parser test and confirm failure**

Run: `python3 -m pytest -q tests/test_vllm024_profile_report.py`

Expected: import failure because the profile module does not exist.

- [ ] **Step 3: Implement normalization and matching**

Normalize `config` and each `results` item into canonical columns. Derive the
profile from ISL/OSL and position encoding, derive method/K from
`speculative_config`, and join against exact baseline keys.

- [ ] **Step 4: Implement compact profile HTML**

Render separate Native 32K, YaRN 64K, and total-128K tables with methodology,
throughput, speedup, latency speedup, acceptance, mean length, and source.

- [ ] **Step 5: Run the focused parser tests**

Run: `python3 -m pytest -q tests/test_vllm024_profile_report.py`

Expected: all profile tests pass.

### Task 4: Collect and Publish Completed Results

**Files:**
- Create: `experiments/vllm_024_dynamicsd/report/20260704_vllm_native_completed/`
- Create: `experiments/vllm_024_dynamicsd/report/vllm024_profiles_latest.csv`
- Create: `public/data/vllm024_profiles_latest.csv`
- Add completed DFlare JSON files below `experiments/vllm_024_dynamicsd/report/20260704_dflare_completed/`

**Interfaces:**
- Consumes: Task 3 normalizer and the 60 remote vLLM-native JSON files.
- Produces: one canonical CSV containing all complete Native 32K, 64K, and 128K rows.

- [ ] **Step 1: Pull only result JSON and job manifests from Lyris**

Use `rsync` from the three pinned result roots. Do not copy model caches or
logs. Preserve profile/domain/method path components.

- [ ] **Step 2: Pull completed DFlare retry JSON files**

Collect jobs `2272937`, `2272938`, and `2272941`; retain jobs `2272239`,
`2272939`, `2272942`, and `2272943` as status-only rows.

- [ ] **Step 3: Generate and validate canonical CSVs**

Assert the raw vLLM-native file count is 60, the DFlare completed job set is
exact, every speedup row has an exact baseline key, and no runtime-crossing
comparison exists.

### Task 5: Integrate HTML and Failure Status

**Files:**
- Modify: `scripts/build_latest_specdec_html_pages.py`
- Modify: `scripts/vllm024_dflare_report.py`
- Modify: `scripts/build_pages_index.py`
- Modify: `tests/test_vllm024_dflare_report.py`
- Modify: `public/reports/vllm_standalone_results_latest.html` through the builder

**Interfaces:**
- Consumes: profile CSV/renderer from Tasks 3-4 and DFlare completed/status rows.
- Produces: latest local and Pages HTML containing all complete rows and all failed job root causes.

- [ ] **Step 1: Add failing integration assertions**

Require the latest HTML to contain the three profile headings, all completed
DFlare retry IDs, and a status table containing TIMEOUT and gather OOM rows.

- [ ] **Step 2: Run report tests and confirm failure**

Run: `python3 -m pytest -q tests/test_vllm024_dflare_report.py tests/test_vllm024_profile_report.py`

- [ ] **Step 3: Wire both renderers into the builder**

Keep the vLLM-native section separate from AngelSlim DFlare. Publish both CSVs
under `public/data`, update the report index counts, and make failure rows
non-performance status entries.

- [ ] **Step 4: Rebuild and parse HTML**

Run:

```bash
python3 scripts/build_latest_specdec_html_pages.py
python3 scripts/build_pages_index.py
python3 -m py_compile scripts/build_latest_specdec_html_pages.py scripts/build_pages_index.py scripts/vllm024_profile_report.py
```

Expected: commands exit zero and the HTML parser accepts every generated page.

### Task 6: Remote Staging and DFlare Retries

**Files:**
- Modify: `experiments/vllm_024_dynamicsd/report/README.md`

**Interfaces:**
- Consumes: committed Task 2 patch and existing submission scripts.
- Produces: staged runner, smoke evidence, and retry job IDs.

- [ ] **Step 1: Run the complete local verification suite**

Run: `python3 -m pytest -q tests/test_angelslim_dflare_transport.py tests/test_vllm024_profile_report.py tests/test_vllm024_dflare_report.py tests/test_vllm024_dynamicsd.py`

- [ ] **Step 2: Commit and push only verified files**

Use signed commits on `codex/vllm024-dynamicsd` and push before remote staging.

- [ ] **Step 3: Pull and stage on Lyris**

Pull the branch in
`/lustre/fsw/coreai_dlalgo_llm/users/sna/Nemo-RL_Qwen3_Roadmap-vllm024`, run
the staging job, and verify the compact transport import and six-hour process
group timeout in the staged source.

- [ ] **Step 4: Run a short multi-rank gather smoke**

Use a short OSL with four ranks and verify four partial files plus one final
result JSON are written without CUDA-backed object collection.

- [ ] **Step 5: Submit missing long-context profiles**

Use `sbatch --test-only` first. Retry the gather-OOM profile on `gb200`; submit
the three wall-time profiles on `gb200-backfill` with eight hours. Keep
`--segment=1`, no `--gres`, and the same model/data/temperature setup.

- [ ] **Step 6: Monitor five minutes and record status**

Check SLURM state and severe log patterns. Add the job IDs, exact config, and
current state to the report README; do not add running rows to performance
tables.
