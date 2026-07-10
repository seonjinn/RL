# NeMo-RL Probabilistic Draft Sampling Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run NeMo-RL with the same `standard` rejection plus `probabilistic` draft-sampling contract as the vLLM 0.24 standalone benchmark, with fail-closed correctness and matched Qwen3-30B-A3B performance evidence.

**Architecture:** Keep vLLM's target-distribution rejection mode at `standard` and explicitly pass `draft_sample_method=probabilistic` for model-based drafters. Extend the existing version-gated NeMo-RL vLLM source adapter to expand per-request temperature metadata for flattened parallel-draft logits, then keep the existing missing-q(token) fail-closed patch. Make launcher provenance explicit and validate with local TDD, a three-step GB200 smoke, and a matched 20-step comparison. The EfficientRollout-style runtime controller is Phase 2 because its policy must be calibrated from Phase 1 telemetry.

**Tech Stack:** Python 3.13, pytest, Ruff, Bash, NeMo-RL GRPO, vLLM 0.24, Ray, SLURM, GB200, W&B.

## Global Constraints

- Use branch `sna/nemorl-vllm024-upgrade` in the isolated worktree `.worktrees/nemorl-vllm024-upgrade`.
- Keep `rejection_sample_method=standard`; do not alias `rejection_sample_method=probabilistic`.
- Use `draft_sample_method=probabilistic` for EAGLE, PARD, DFlash, and other model-based drafters; do not apply it to suffix decoding.
- Keep CUDA graphs enabled with `policy.generation.vllm_cfg.enforce_eager=false` and `cudagraph_mode=PIECEWISE`.
- Keep the performance recipe unchanged except for explicit SpecDec and benchmark logging overrides.
- Keep rollout MoE backend behavior unchanged and use the existing Triton configuration.
- Keep `checkpointing.enabled=false`.
- Fail closed when probabilistic draft metadata is incomplete or shape-incompatible.
- Commit with sign-off, push before submission, initialize recursive submodules, and monitor submitted jobs for at least five minutes.
- Do not modify `pyproject.toml`, `uv.lock`, model weights, reward computation, policy logprobs, or training updates.
- Host-compatible launcher tests run with `python3 -m pytest`; NeMo-RL unit tests run with `/opt/nemo_rl_venv/bin/python -m pytest` inside the Linux nightly container because the repository lockfile does not support macOS.

---

### Task 1: Port the parallel probabilistic-draft temperature fix

**Files:**
- Modify: `nemo_rl/models/generation/vllm/patches.py`
- Modify: `tests/unit/models/generation/test_vllm_patches.py`

**Interfaces:**
- Consumes: vLLM 0.24 `compute_probs_and_sample_next_token(logits, sampling_metadata, use_fp64_gumbel)` and the existing `_apply_vllm_patches()` version-gated adapter.
- Produces: `_patch_vllm_parallel_probabilistic_draft_temperature(logger) -> None`, installed only for `rejection_sample_method == "standard"` and `draft_sample_method == "probabilistic"`.

- [ ] **Step 1: Write failing source-adapter tests**

Add tests that create a temporary `llm_base_proposer.py` containing the exact vLLM 0.24 temperature block, apply the patch, and assert that the patched source:

```python
assert "temperature_count = temperature.numel()" in patched
assert "logits_count % temperature_count != 0" in patched
assert "temperature.repeat_interleave(logits_count // temperature_count)" in patched
```

Also assert that a second call is idempotent and that an unknown source layout raises `RuntimeError` containing `vLLM source layout changed`.

- [ ] **Step 2: Extend the patch-runner selection test**

In `test_apply_patches_only_installs_required_specdec_patches`, add a mock for `_patch_vllm_parallel_probabilistic_draft_temperature`. Require one call for `standard + probabilistic`, and no call for greedy draft sampling, synthetic rejection, empty SpecDec, or baseline.

- [ ] **Step 3: Commit and push the test-only red state**

```bash
git add tests/unit/models/generation/test_vllm_patches.py
git commit -s -m "test(vllm): cover probabilistic parallel drafts"
git push fork sna/nemorl-vllm024-upgrade
```

- [ ] **Step 4: Run the tests and confirm they fail**

Run:

```bash
/opt/nemo_rl_venv/bin/python -m pytest tests/unit/models/generation/test_vllm_patches.py \
  -k 'parallel_probabilistic or apply_patches_only' -vv
```

Run this command in the AWS nightly container. Expected: failure because `_patch_vllm_parallel_probabilistic_draft_temperature` is not defined or not invoked.

- [ ] **Step 5: Implement the minimal version-gated source patch**

Add a patch function next to `_patch_vllm_missing_draft_probs_fail_closed`. Replace only the vLLM 0.24 temperature block with logic equivalent to:

```python
temperature = sampling_metadata.temperature
temperature_count = temperature.numel()
logits_count = logits.shape[0]
if temperature_count != logits_count:
    if temperature_count <= 0 or logits_count % temperature_count != 0:
        raise RuntimeError(
            "parallel draft logits count is not divisible by the sampling "
            f"temperature count: logits={logits_count}, "
            f"temperatures={temperature_count}"
        )
    temperature = temperature.repeat_interleave(
        logits_count // temperature_count
    )
```

Preserve the existing mixed greedy/random handling after expansion. Refuse to patch an unknown source layout and make repeat application idempotent.

- [ ] **Step 6: Install both probabilistic correctness patches together**

Inside `_apply_vllm_patches`, under the existing `standard + probabilistic` condition, call:

```python
_patch_vllm_parallel_probabilistic_draft_temperature(patch_logger)
_patch_vllm_missing_draft_probs_fail_closed(patch_logger)
```

Do not install either patch for greedy draft sampling.

- [ ] **Step 7: Run focused tests and lint**

Run:

```bash
/opt/nemo_rl_venv/bin/python -m pytest tests/unit/models/generation/test_vllm_patches.py \
  -k 'parallel_probabilistic or missing_probabilistic or apply_patches_only' -vv
/opt/nemo_rl_venv/bin/python -m ruff check nemo_rl/models/generation/vllm/patches.py \
  tests/unit/models/generation/test_vllm_patches.py
```

Run these commands in the AWS nightly container. Expected: all selected tests pass and Ruff exits zero.

- [ ] **Step 8: Commit Task 1**

```bash
git add nemo_rl/models/generation/vllm/patches.py \
  tests/unit/models/generation/test_vllm_patches.py
git commit -s -m "fix(vllm): support probabilistic parallel drafts"
```

---

### Task 2: Make the NeMo-RL benchmark launcher sampling contract explicit

**Files:**
- Modify: `experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh`
- Modify: `tests/test_vllm_024_launch_scripts.py`

**Interfaces:**
- Consumes: environment variables `REJECTION_SAMPLE_METHOD` and `DRAFT_SAMPLE_METHOD`.
- Produces: explicit Hydra overrides for model-based SpecDec methods and manifest columns recording both resolved values.

- [ ] **Step 1: Write failing launcher contract tests**

Add tests requiring EAGLE and PARD dry runs to contain:

```text
speculative_config.rejection_sample_method=standard
speculative_config.draft_sample_method=probabilistic
```

Require suffix to contain only `rejection_sample_method=standard`, and baseline to remain free of every `speculative_config` override. Add subprocess tests proving any rejection value other than `standard`, or any draft value outside `greedy|probabilistic`, exits nonzero with an actionable error.

- [ ] **Step 2: Run the tests and confirm they fail**

Run:

```bash
python3 -m pytest tests/test_vllm_024_launch_scripts.py \
  -k 'sampling or probabilistic or suffix or baseline' -vv
```

Expected: failures because the launcher currently relies on vLLM defaults and does not render the requested sampling fields.

- [ ] **Step 3: Add centralized launcher defaults and validation**

Near the existing `STATIC_K` and `DYNAMIC_SCHEDULE` settings, add:

```bash
REJECTION_SAMPLE_METHOD="${REJECTION_SAMPLE_METHOD:-standard}"
DRAFT_SAMPLE_METHOD="${DRAFT_SAMPLE_METHOD:-probabilistic}"
```

Reject non-`standard` rejection values and draft values outside `greedy` and `probabilistic` before model or job construction.

- [ ] **Step 4: Render method-appropriate overrides**

For suffix, render only:

```bash
"++policy.generation.vllm_kwargs.speculative_config.rejection_sample_method=${REJECTION_SAMPLE_METHOD}"
```

For EAGLE and PARD, additionally render:

```bash
"++policy.generation.vllm_kwargs.speculative_config.draft_sample_method=${DRAFT_SAMPLE_METHOD}"
```

Keep the baseline path unchanged.

- [ ] **Step 5: Record sampling provenance**

Add `rejection_sample_method` and `draft_sample_method` columns to `submissions.tsv`. Record `not_applicable` for baseline and suffix draft sampling instead of implying probabilistic draft sampling was active.

- [ ] **Step 6: Run launcher tests, shell syntax, and lint**

Run:

```bash
bash -n experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh
python3 -m pytest tests/test_vllm_024_launch_scripts.py -vv
ruff check tests/test_vllm_024_launch_scripts.py
```

Expected: all launcher tests pass, Bash syntax exits zero, and Ruff exits zero.

- [ ] **Step 7: Commit Task 2**

```bash
git add experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh \
  tests/test_vllm_024_launch_scripts.py
git commit -s -m "feat(bench): enable probabilistic draft sampling"
```

---

### Task 3: Verify the complete local correctness contract

**Files:**
- Verify: `nemo_rl/models/generation/__init__.py`
- Verify: `nemo_rl/models/generation/vllm/patches.py`
- Verify: `tests/unit/models/generation/test_vllm_generation.py`
- Verify: `tests/unit/models/generation/test_vllm_patches.py`
- Verify: `tests/test_vllm_024_launch_scripts.py`

**Interfaces:**
- Consumes: explicit launcher fields from Task 2 and the vLLM adapters from Task 1.
- Produces: evidence that config validation, runtime-contract logging, probability-row failure, and launcher rendering agree.

- [ ] **Step 1: Run focused generation and patch tests**

```bash
/opt/nemo_rl_venv/bin/python -m pytest \
  tests/unit/models/generation/test_vllm_generation.py \
  tests/unit/models/generation/test_vllm_patches.py \
  tests/test_vllm_024_launch_scripts.py -vv
```

Run this command in the AWS nightly container. Expected: all tests pass.

- [ ] **Step 2: Verify the exact Qwen3-30B-A3B commands**

```bash
RUN_TAG=probabilistic-contract-check \
DRAFT_SAMPLE_METHOD=probabilistic \
bash experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh \
  dry-run qwen30ba3b eagle3_k5

RUN_TAG=probabilistic-contract-check \
DRAFT_SAMPLE_METHOD=probabilistic \
bash experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh \
  dry-run qwen30ba3b pard_k5
```

Expected: both commands show CUDA graphs on, temperature/top-p 1.0, standard rejection, probabilistic draft sampling, and the correct draft checkpoint and K.

- [ ] **Step 3: Run repository checks for touched files**

```bash
git diff --check
ruff check nemo_rl/models/generation/vllm/patches.py \
  tests/unit/models/generation/test_vllm_patches.py \
  tests/test_vllm_024_launch_scripts.py
```

Expected: both commands exit zero.

---

### Task 4: Push and run matched GB200 smoke jobs

**Files:**
- Update: `experiments/vllm_024_upgrade/runs/nemorl-v024-q30-probdraft-smoke-20260709/submissions.tsv`
- Create: `experiments/vllm_024_upgrade/runs/nemorl-v024-q30-probdraft-smoke-20260709/smoke-summary.json`

**Interfaces:**
- Consumes: pushed `sna/nemorl-vllm024-upgrade`, recursively initialized submodules, staged nightly image, performance recipe, and Qwen3-30B-A3B/EAGLE/PARD checkpoints.
- Produces: three-step baseline, greedy-draft, and probabilistic-draft correctness/performance evidence with W&B links.

- [ ] **Step 1: Push the branch and update the cluster checkout**

```bash
git push fork sna/nemorl-vllm024-upgrade
git submodule update --init --recursive
```

On `aws-dfw-cs-001-login-01.nvidia.com`, update `/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/sna/RL-vllm024-upgrade-20260707` to the pushed branch with `git pull --ff-only fork sna/nemorl-vllm024-upgrade`, then run `git submodule update --init --recursive`. Record the main commit, every recursive submodule commit, container path, and container SHA256.

- [ ] **Step 2: Run SLURM scheduling preflight**

Run the launcher in `test-only` mode for Qwen3-30B-A3B baseline and EAGLE K5 before submitting. Require no dependency, the correct account/partition, four nodes, `--segment=4`, and the cluster-appropriate GPU allocation flags.

- [ ] **Step 3: Submit matched three-step controls**

Submit separate run tags with identical recipe, container, commit, topology, sampling inputs, and CUDA graph mode:

```bash
MAX_STEPS=3 \
RUN_TAG=nemorl-v024-q30-probdraft-smoke-20260709-baseline \
bash experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh \
  submit qwen30ba3b baseline

MAX_STEPS=3 DRAFT_SAMPLE_METHOD=greedy \
RUN_TAG=nemorl-v024-q30-probdraft-smoke-20260709-greedy \
bash experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh \
  submit qwen30ba3b eagle3_k5

MAX_STEPS=3 DRAFT_SAMPLE_METHOD=probabilistic \
RUN_TAG=nemorl-v024-q30-probdraft-smoke-20260709-probabilistic \
bash experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh \
  submit qwen30ba3b eagle3_k5
```

After EAGLE confirms the common sampling path, submit PARD K5 with `MAX_STEPS=3`, `DRAFT_SAMPLE_METHOD=probabilistic`, and `RUN_TAG=nemorl-v024-q30-pard-k5-probdraft-smoke-20260709`; it exercises the parallel-draft temperature expansion specifically.

- [ ] **Step 4: Monitor every submitted job for five minutes**

Check `squeue`, startup logs, and W&B. Require:

```text
rejection_sample_method=standard
draft_sample_method=probabilistic
cuda_graph_enabled=true
```

Reject any run with OOM, NCCL failure, Ray actor restart, CUDA graph fallback, missing q(token), incompatible temperature shape, non-finite logprob, or absent SpecDec telemetry.

- [ ] **Step 5: Compare stochastic correctness**

Check generated-token counts, one finite chosen-token logprob per generated token, reward statistics, and later-position token distributions. Do not require token-for-token equality at temperature 1.0.

---

### Task 5: Run the matched 20-step comparison and prepare Phase 2 calibration

**Files:**
- Create: `experiments/vllm_024_upgrade/runs/nemorl-v024-q30-probdraft-step20-20260709/step20-summary.json`
- Create: `experiments/vllm_024_upgrade/runs/nemorl-v024-q30-probdraft-step20-20260709/controller-calibration-input.json`
- Update: `/Users/sna/Nemo-RL_Qwen3_Roadmap/public/reports/lyris_nemorl_perfcfg_specdec_live_status_latest.html`

**Interfaces:**
- Consumes: passed smoke gates and existing scalar W&B rollout metrics.
- Produces: step 2-20 averages and the empirical inputs for the separate runtime-controller implementation plan.

- [ ] **Step 1: Submit matched 20-step runs**

Run baseline, EAGLE K5 greedy, EAGLE K5 probabilistic, and PARD K5 probabilistic using run tags `nemorl-v024-q30-probdraft-step20-20260709-baseline`, `nemorl-v024-q30-probdraft-step20-20260709-greedy`, `nemorl-v024-q30-probdraft-step20-20260709-probabilistic`, and `nemorl-v024-q30-pard-k5-probdraft-step20-20260709`. Use the unmodified Qwen3-30B-A3B performance recipe and keep commit, container, topology, OSL, temperature/top-p, CUDA graph mode, and W&B project matched.

- [ ] **Step 2: Compute step 2-20 metrics**

Report for each run:

```text
E2E step time
generation time
logprob time
policy-training time
E2E throughput
generation throughput
acceptance rate
mean accepted length
drafted/accepted/emitted token counts
```

Compute baseline-relative time reduction and throughput speedup only from matched completed steps.

- [ ] **Step 3: Build calibration input**

Store active request count, scheduled token count, sequence-length statistics, selected K, acceptance, mean accepted length, proposer/verification timing when exposed, and generation timing. Include exact model, draft model, target/draft TP, GPU, container digest, vLLM version, and commits.

- [ ] **Step 4: Decide the Phase 2 gate**

Proceed to the EfficientRollout-style controller plan only if Phase 1 proves correct stochastic behavior and complete telemetry. Use the measured GB200 TP2 costs to choose the controller's K set, break-even rule, hysteresis, and fallback behavior; do not hard-code thresholds from A100 or standalone serving.
