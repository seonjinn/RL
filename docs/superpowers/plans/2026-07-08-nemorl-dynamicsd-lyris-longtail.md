# NeMo-RL DynamicSD Lyris and 32K Long-Tail Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch matched baseline, Eagle-3 K5, and DynamicSD NeMo-RL performance-recipe runs on Lyris for the native recipe limits and a 32K-output long-tail profile.

**Architecture:** Extend the existing experiment launcher with an explicit profile dimension instead of changing shared NeMo-RL defaults or duplicating recipes. The launcher emits profile-specific Hydra overrides, provenance, W&B identities, and Lyris-compatible SLURM commands; the existing collector validates profile and length identity before computing matched speedups.

**Tech Stack:** Bash, Python 3.13, pytest, Hydra/OmegaConf, NeMo-RL GRPO, vLLM 0.24, Eagle-3 DynamicSD, SLURM/Pyxis, W&B.

## Global Constraints

- Use the upstream Qwen3-30B-A3B, Qwen3-32B, and Qwen3-235B synchronous performance recipes.
- Compare `baseline`, `eagle3_k5`, and `dynamic` with temperature/top-p `1.0/1.0` and PIECEWISE CUDA Graphs.
- Set `num_speculative_tokens=5`, draft TP1, and DynamicSD schedule `[[1,16,5],[17,32,4],[33,64,3],[65,128,1],[129,512,0]]`.
- Set vLLM `max_num_batched_tokens=16384` for every matched variant so target and draft-token scheduler budgets are explicit and identical.
- The recipe profile changes no recipe-owned sequence limit.
- The long-tail profile sets max output 32,768 and max total/model length 36,864.
- Disable checkpoint writes and retain Steps 2-20 for final means.
- Use Lyris account `coreai_dlalgo_llm`, partition `gb200`, no `--gres`, and segment equal to node count.

---

### Task 1: Add Profile and Draft-Budget Launcher Contracts

**Files:**
- Modify: `tests/test_vllm_024_launch_scripts.py`
- Modify: `experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh`

**Interfaces:**
- Consumes: environment `PROFILE=recipe|longtail32k` and existing launcher CLI `MODE MODEL VARIANT`.
- Produces: profile-specific Hydra overrides and manifest fields `profile`, `max_new_tokens`, `max_total_sequence_length`, and `max_num_batched_tokens`.

- [ ] **Step 1: Write failing launcher tests**

Add tests that render both profiles and assert:

```python
assert "vllm_kwargs.max_num_batched_tokens=16384" in output
assert "policy.max_total_sequence_length=36864" in longtail_output
assert "policy.generation.max_new_tokens=32768" in longtail_output
assert "policy.generation.vllm_cfg.max_model_len=36864" in longtail_output
assert "policy.max_total_sequence_length=" not in recipe_output
assert "--gres" not in lyris_output
assert "--account=coreai_dlalgo_llm" in lyris_output
assert "--partition=gb200" in lyris_output
```

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --no-sync pytest tests/test_vllm_024_launch_scripts.py -q
```

Expected: new profile assertions fail because the launcher does not yet render profile or length overrides.

- [ ] **Step 3: Implement profile parsing and overrides**

Add:

```bash
PROFILE="${PROFILE:-recipe}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
case "${PROFILE}" in
  recipe)
    max_new_tokens="recipe"
    max_total_sequence_length="recipe"
    ;;
  longtail32k)
    max_new_tokens=32768
    max_total_sequence_length=36864
    ;;
  *)
    echo "ERROR: PROFILE must be recipe or longtail32k" >&2
    exit 2
    ;;
esac
```

Pass `++policy.generation.vllm_kwargs.max_num_batched_tokens=16384` to all variants. Append the three long-tail sequence overrides only for `longtail32k`. Include the profile in run directories, job names, W&B IDs, and manifest rows.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run --no-sync pytest tests/test_vllm_024_launch_scripts.py -q
bash -n experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh
```

Expected: PASS with no shell syntax errors.

### Task 2: Protect Result Matching Across Profiles

**Files:**
- Modify: `tests/test_vllm_024_dynamicsd_summary.py`
- Modify: `experiments/vllm_024_upgrade/summarize_eagle3_dynamicsd.py`

**Interfaces:**
- Consumes: profile-aware `submissions.tsv` rows.
- Produces: comparison validation that rejects mixed profile or sequence-length rows for the same model.

- [ ] **Step 1: Write failing manifest-validation tests**

Create baseline and DynamicSD manifest rows that differ only in `profile`, `max_new_tokens`, `max_total_sequence_length`, or `max_num_batched_tokens`; assert `_validate_manifest_rows()` returns `mismatched setup` for each difference.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --no-sync pytest tests/test_vllm_024_dynamicsd_summary.py -q
```

Expected: new tests fail because the four provenance fields are not part of `MATCHED_SETUP_FIELDS`.

- [ ] **Step 3: Extend the matching identity**

Append these exact names to `MATCHED_SETUP_FIELDS`:

```python
"profile",
"max_new_tokens",
"max_total_sequence_length",
"max_num_batched_tokens",
```

- [ ] **Step 4: Verify GREEN**

Run the focused summary tests and expect PASS.

### Task 3: Document and Verify the Lyris Launch Contract

**Files:**
- Modify: `experiments/vllm_024_upgrade/README.md`

**Interfaces:**
- Consumes: the profile-aware launcher.
- Produces: exact test-only, one-step smoke, and 20-step Lyris commands.

- [ ] **Step 1: Add Lyris commands**

Document `ACCOUNT=coreai_dlalgo_llm`, `PARTITION=gb200`, `USE_GRES=false`, Lyris root/container/HF paths, the two profile names, and W&B projects.

- [ ] **Step 2: Run repository verification**

Run:

```bash
uv run --no-sync pytest tests/test_vllm_024_launch_scripts.py tests/test_vllm_024_dynamicsd_summary.py -q
git diff --check
```

Expected: all tests pass and no whitespace errors are reported.

- [ ] **Step 3: Commit and push**

Commit with sign-off and push `sna/nemorl-vllm024-dynamicsd-20260708` before remote submission.

### Task 4: Submit and Gate the Lyris Cohorts

**Files:**
- Generated remotely: `experiments/vllm_024_upgrade/runs/<run-tag>/submissions.tsv`

**Interfaces:**
- Consumes: pushed branch, staged nightly container, model/drafter checkpoints, W&B credentials.
- Produces: scheduler-tested smoke and 20-step job manifests for both profiles.

- [ ] **Step 1: Sync the remote worktree and validate assets**

Pull the pushed branch into a dedicated Lyris worktree and verify the container plus all six target/drafter paths before scheduling.

- [ ] **Step 2: Run scheduler test-only**

Render and test-only submit the three models and three variants for both profiles with `MAX_STEPS=1`.

- [ ] **Step 3: Submit one-step smokes**

Submit baseline, K5, and DynamicSD for each model/profile. Monitor for five minutes and require rollout plus policy-training progress, positive draft/accept counters, resolved `max_num_batched_tokens=16384`, and no CUDA Graph fallback error.

- [ ] **Step 4: Promote passing profiles to 20 steps**

Submit `MAX_STEPS=20` only for model/profile triplets whose smokes pass. Record job IDs and W&B URLs from each `submissions.tsv`.
