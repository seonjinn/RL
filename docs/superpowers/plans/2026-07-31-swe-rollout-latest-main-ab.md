# SWE Rollout Latest-Main A/B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether the existing node-local OpenHands optimization still provides independent SWE rollout wall-time savings after upgrading to NeMo-RL latest main, vLLM 0.25.1, and the relevant upstream rollout optimizations.

**Architecture:** Freeze one SWE2 rollout-only workload, then create immutable NeMo-RL revisions for upstream main, PR #3390, rebased PR #3283, and the node-local Gym patch. Run a short correctness canary before a matched ABBA job-level comparison, parse allocation-to-result wall time and OpenHands phase timings, and publish both successful results and failed attempts in the existing HTML report.

**Tech Stack:** NeMo-RL, NeMo Gym, nv-OpenHands, vLLM 0.25.1, Ray, Hydra, SLURM, Apptainer, Python, pytest, static HTML.

## Global Constraints

- Use upstream NeMo-RL `main` at `e08fc276e41b9f4fef69a0b91f6f384ad1964f3d` as the frozen baseline; record a new SHA if upstream is intentionally refreshed before submission.
- Use vLLM 0.25.1, CUDA Graph enabled, and an explicit dense `cudagraph_capture_sizes` list covering every speculative verification batch.
- Hold model, drafter, K, prompt rows, seed, number of generations, concurrency, token limits, container, worker venvs, and SLURM resources constant across arms.
- Use Qwen3-30B-A3B-Thinking-2507, the three SWE2 validation rows used by `val-mini3.jsonl`, eight generations per row, and concurrency 24 for the primary gate.
- Compare allocation-to-result job wall time. Report rollout-duration sums separately and include the one-time node-local staging cost.
- Preserve the legacy path as the fallback and keep node-local staging opt-in through `NRL_OH_NODE_LOCAL=1`.
- Do not claim a workspace-cache improvement: that optimization is not implemented or validated yet.
- Do not mutate the dirty roadmap worktree beyond the files listed in this plan. Stage and commit exact paths only, with `git commit -s`, author `seonjinn <sna@nvidia.com>`.
- Before every SLURM submission, run scheduling/fair-share preflight, `git pull --ff-only` on the remote experiment branch, and monitor a running job for at least five minutes.

---

### Task 1: Freeze Reproduction Provenance

**Files:**
- Create: `docs/superpowers/plans/2026-07-31-swe-rollout-latest-main-ab.md`
- Create: `experiments/swe_rollout_latest_main_ab/README.md`
- Create: `experiments/swe_rollout_latest_main_ab/provenance.json`
- Create: `experiments/swe_rollout_latest_main_ab/attempts.md`

**Interfaces:**
- Consumes: `HANDOFF_CODEX.md`, `experiments/dflash_loss_ab/report/data/patch_ab_n24.csv`, and the Lyris `RL-wt-nemogym-opt` worktree.
- Produces: a machine-readable frozen workload and the exact source commit of `_node_local_mirror` for Tasks 2–5.

- [ ] **Step 1: Recover the remote worktree, commit, and launcher without changing remote state**

```bash
ssh -O check login-lyris
ssh -o ConnectTimeout=5 login-lyris 'for d in /lustre/fsw/coreai_dlalgo_llm/users/sna/RL-wt-nemogym-opt /project/coreai_dlalgo_llm/users/sna/RL-wt-nemogym-opt; do [ -d "$d" ] || continue; git -C "$d" status --short; git -C "$d" rev-parse HEAD; git -C "$d" log --all -S NRL_OH_NODE_LOCAL --oneline -- responses_api_agents/swe_agents/app.py 3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/app.py; find "$d/experiments/nemogym_swe1_specdec" -maxdepth 3 -type f -print 2>/dev/null | sort; done'
```

Expected: one worktree path, its clean/dirty status, a NeMo-RL SHA, the patch commit or working-tree diff location, and the prior SWE launcher/config paths.

- [ ] **Step 2: Recover the exact Gym patch and workload settings**

```bash
ssh login-lyris 'd=/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-wt-nemogym-opt; [ -d "$d" ] || d=/project/coreai_dlalgo_llm/users/sna/RL-wt-nemogym-opt; git -C "$d" diff -- 3rdparty/Gym-workspace/Gym; rg -n "NRL_OH_NODE_LOCAL|_node_local_mirror|NUM_PROMPTS|NUM_GENS|val-mini3|cudagraph_capture_sizes|speculative|draft" "$d/experiments/nemogym_swe1_specdec" "$d/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/app.py" 2>/dev/null | head -400'
```

Expected: the complete opt-in patch and enough launcher lines to prove the model, data, concurrency, K, and CUDA Graph coverage.

- [ ] **Step 3: Write the frozen provenance files**

`provenance.json` must contain these keys with observed values: `nemo_rl_sha`, `gym_sha`, `nv_openhands_sha`, `container_path`, `container_sha256`, `worker_venv_path`, `vllm_version`, `model`, `drafter`, `speculative_tokens`, `dataset_path`, `dataset_sha256`, `num_prompts`, `num_generations`, `concurrency`, `seed`, `max_model_len`, `max_new_tokens`, `cudagraph_capture_sizes`, `slurm_cluster`, `slurm_partition`, `slurm_account`, and `source_patch_commit`.

- [ ] **Step 4: Validate the provenance schema**

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("experiments/swe_rollout_latest_main_ab/provenance.json")
data = json.loads(path.read_text())
required = {
    "nemo_rl_sha", "gym_sha", "nv_openhands_sha", "container_path",
    "container_sha256", "worker_venv_path", "vllm_version", "model",
    "drafter", "speculative_tokens", "dataset_path", "dataset_sha256",
    "num_prompts", "num_generations", "concurrency", "seed",
    "max_model_len", "max_new_tokens", "cudagraph_capture_sizes",
    "slurm_cluster", "slurm_partition", "slurm_account",
    "source_patch_commit",
}
missing = sorted(required - data.keys())
assert not missing, f"missing provenance keys: {missing}"
assert data["vllm_version"] == "0.25.1"
assert data["num_prompts"] == 3
assert data["num_generations"] == 8
assert data["concurrency"] == 24
assert data["cudagraph_capture_sizes"], "CUDA Graph coverage is empty"
PY
```

Expected: exit code 0.

- [ ] **Step 5: Commit the reproduction manifest**

```bash
git add docs/superpowers/plans/2026-07-31-swe-rollout-latest-main-ab.md experiments/swe_rollout_latest_main_ab/README.md experiments/swe_rollout_latest_main_ab/provenance.json experiments/swe_rollout_latest_main_ab/attempts.md
git commit -s -m "docs: freeze SWE rollout latest-main benchmark"
```

### Task 2: Build Immutable Upstream Variants

**Files:**
- Create: `.worktrees/swe-ab-main/` as a git worktree, not a tracked directory.
- Create: `.worktrees/swe-ab-3390/` as a git worktree, not a tracked directory.
- Create: `.worktrees/swe-ab-3390-3283/` as a git worktree, not a tracked directory.
- Modify: `.worktrees/swe-ab-3390-3283/nemo_rl/algorithms/distillation.py`
- Modify: `.worktrees/swe-ab-3390-3283/nemo_rl/algorithms/grpo.py`
- Modify: `.worktrees/swe-ab-3390-3283/nemo_rl/environments/nemo_gym.py`
- Modify: `.worktrees/swe-ab-3390-3283/nemo_rl/experience/rollout_manager.py`
- Modify: `.worktrees/swe-ab-3390-3283/nemo_rl/experience/rollouts.py`
- Modify: the latest-main single-controller NemoGym setup call site discovered by `rg -n "spinup_nemo_gym_actor|NemoGym\.options" nemo_rl`.
- Test: `.worktrees/swe-ab-3390-3283/tests/unit/algorithms/test_distillation.py`
- Test: `.worktrees/swe-ab-3390-3283/tests/unit/environments/test_nemo_gym.py`
- Test: `.worktrees/swe-ab-3390-3283/tests/unit/single_controller/test_single_controller_setup.py`

**Interfaces:**
- Consumes: frozen upstream SHA and PR head SHAs from GitHub.
- Produces: three pushed, immutable NeMo-RL experiment branches whose only differences are visible in `git range-diff`.

- [ ] **Step 1: Create the clean baseline worktree**

```bash
git -C tmp/RL_latest_main fetch origin main
git -C tmp/RL_latest_main worktree add -b seonjinn/swe-ab-main-e08fc276 /Users/sna/Nemo-RL_Qwen3_Roadmap/.worktrees/swe-ab-main e08fc276e41b9f4fef69a0b91f6f384ad1964f3d
git -C .worktrees/swe-ab-main submodule update --init 3rdparty/Gym-workspace/Gym
```

Expected: clean status and Gym at the gitlink recorded by the baseline commit.

- [ ] **Step 2: Create and cherry-pick PR #3390**

```bash
git -C tmp/RL_latest_main fetch origin pull/3390/head:refs/remotes/origin/pr-3390
git -C tmp/RL_latest_main worktree add -b seonjinn/swe-ab-3390 /Users/sna/Nemo-RL_Qwen3_Roadmap/.worktrees/swe-ab-3390 e08fc276e41b9f4fef69a0b91f6f384ad1964f3d
git -C .worktrees/swe-ab-3390 cherry-pick -x 5fc345f8 c3129078 e5fa7802
```

Expected: the cherry-pick changes token metadata handling and removes awaited per-turn `/tokenize` calls without unrelated diffs.

- [ ] **Step 3: Run the PR #3390 unit tests**

```bash
cd .worktrees/swe-ab-3390
uv run pytest tests/unit/models/generation/test_vllm_generation.py tests/unit/models/generation/test_vllm_utils.py -q
```

Expected: exit code 0. Record dependency or collection failures verbatim in `experiments/swe_rollout_latest_main_ab/attempts.md`; do not convert them into performance failures.

- [ ] **Step 4: Rebase PR #3283 on top of the verified #3390 branch**

```bash
git -C tmp/RL_latest_main fetch origin pull/3283/head:refs/remotes/origin/pr-3283
git -C tmp/RL_latest_main worktree add -b seonjinn/swe-ab-3390-3283 /Users/sna/Nemo-RL_Qwen3_Roadmap/.worktrees/swe-ab-3390-3283 seonjinn/swe-ab-3390
git -C .worktrees/swe-ab-3390-3283 cherry-pick -x f78b0a48
```

If the cherry-pick conflicts, preserve latest-main prompt-group streaming from #3000/#3360, construct the tokenizer once in the driver, pass it through the `NemoGym` actor constructor, and remove only the redundant per-call tokenizer argument. Do not revert streaming or change rollout ordering.

- [ ] **Step 5: Run the rebased PR #3283 tests**

```bash
cd .worktrees/swe-ab-3390-3283
uv run pytest tests/unit/algorithms/test_distillation.py tests/unit/environments/test_nemo_gym.py tests/unit/single_controller/test_single_controller_setup.py -q
uv run ruff check nemo_rl/algorithms/distillation.py nemo_rl/algorithms/grpo.py nemo_rl/environments/nemo_gym.py nemo_rl/experience/rollout_manager.py nemo_rl/experience/rollouts.py tests/unit/algorithms/test_distillation.py tests/unit/environments/test_nemo_gym.py tests/unit/single_controller/test_single_controller_setup.py
```

Expected: exit code 0 and no tokenizer serialization in `run_rollouts.remote(...)` call sites.

- [ ] **Step 6: Commit conflict resolution, push experiment branches, and record SHAs**

```bash
git -C .worktrees/swe-ab-3390-3283 add nemo_rl tests
git -C .worktrees/swe-ab-3390-3283 diff --cached --quiet || git -C .worktrees/swe-ab-3390-3283 commit -s -m "perf: rebase tokenizer reuse for SWE rollout A/B"
git -C .worktrees/swe-ab-main push fork seonjinn/swe-ab-main-e08fc276
git -C .worktrees/swe-ab-3390 push fork seonjinn/swe-ab-3390
git -C .worktrees/swe-ab-3390-3283 push fork seonjinn/swe-ab-3390-3283
```

Expected: three fork branches with SHAs copied into `provenance.json` under `arms.main`, `arms.pr3390`, and `arms.pr3390_pr3283`.

### Task 3: Port and Test the Node-Local Gym Patch

**Files:**
- Modify: `.worktrees/swe-ab-3390-3283/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/app.py`
- Test: `.worktrees/swe-ab-3390-3283/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/tests/test_node_local_mirror.py`
- Modify: `.worktrees/swe-ab-3390-3283/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/configs/swebench_openhands_training.yaml` only if an explicit opt-in config field is required in addition to `NRL_OH_NODE_LOCAL`.

**Interfaces:**
- Consumes: the exact recovered `_node_local_mirror` implementation and latest Gym at the NeMo-RL baseline gitlink.
- Produces: a Gym fork commit plus a NeMo-RL gitlink commit for the fourth benchmark arm.

- [ ] **Step 1: Write isolated filesystem tests before porting the implementation**

The test must use `tmp_path` and monkeypatching to prove: disabled mode is a no-op; the first enabled call copies OpenHands and miniforge exactly once; a second call reuses the completed mirror; a partial mirror is never published; the returned paths are container paths under `/openhands_setup`; and separate source identities do not share a mirror.

- [ ] **Step 2: Run the focused test and verify it fails**

```bash
cd .worktrees/swe-ab-3390-3283/3rdparty/Gym-workspace/Gym
uv run pytest responses_api_agents/swe_agents/tests/test_node_local_mirror.py -q
```

Expected: failure because the latest Gym source does not define the recovered node-local helper.

- [ ] **Step 3: Port the minimal opt-in implementation**

Implement the recovered behavior in `responses_api_agents/swe_agents/app.py` with these invariants: lock one mirror population per node/job; copy into a temporary sibling; atomically rename only after both trees are complete; return `/openhands_setup/OpenHands` and `/openhands_setup/miniforge3` for command generation; keep original Lustre paths when disabled or on any recoverable failure; log staging duration and reuse status once.

- [ ] **Step 4: Run focused Gym verification**

```bash
cd .worktrees/swe-ab-3390-3283/3rdparty/Gym-workspace/Gym
uv run pytest responses_api_agents/swe_agents/tests/test_node_local_mirror.py responses_api_agents/swe_agents/tests -q
uv run ruff check responses_api_agents/swe_agents/app.py responses_api_agents/swe_agents/tests/test_node_local_mirror.py
```

Expected: exit code 0.

- [ ] **Step 5: Commit Gym and then the NeMo-RL gitlink**

```bash
git -C .worktrees/swe-ab-3390-3283/3rdparty/Gym-workspace/Gym add responses_api_agents/swe_agents/app.py responses_api_agents/swe_agents/tests/test_node_local_mirror.py
git -C .worktrees/swe-ab-3390-3283/3rdparty/Gym-workspace/Gym commit -s -m "perf(swe): stage OpenHands on node-local storage"
git -C .worktrees/swe-ab-3390-3283/3rdparty/Gym-workspace/Gym push git@github.com:seonjinn/Gym.git HEAD:seonjinn/swe-node-local-staging
git -C .worktrees/swe-ab-3390-3283 add 3rdparty/Gym-workspace/Gym
git -C .worktrees/swe-ab-3390-3283 commit -s -m "test: add node-local SWE rollout A/B arm"
git -C .worktrees/swe-ab-3390-3283 push fork HEAD:seonjinn/swe-ab-3390-3283-node-local
```

Expected: the fourth arm differs from the third only by the Gym gitlink.

### Task 4: Create the Matched SLURM Harness and Submit Canaries

**Files:**
- Create: `experiments/swe_rollout_latest_main_ab/submit_lyris.sh`
- Create: `experiments/swe_rollout_latest_main_ab/parse_results.py`
- Create: `experiments/swe_rollout_latest_main_ab/tests/test_parse_results.py`
- Create remotely through git checkout: `experiments/swe_rollout_latest_main_ab/jobs/${SLURM_JOB_ID}/` result directories.

**Interfaces:**
- Consumes: four immutable branch SHAs and the Task 1 provenance.
- Produces: one JSON result per job with job wall, staging, Connect, Initialize, Framework, LLM generation, final evaluation, valid rollout count, reward, and generation throughput.

- [ ] **Step 1: Write parser tests with a synthetic complete log and a failed-rollout log**

The complete fixture must assert exact extraction of `allocation_to_result_s`, `staging_s`, `connect_s`, `initialize_s`, `framework_s`, `llm_generation_s`, `final_eval_s`, `valid_rollouts`, `failed_rollouts`, and `generation_tok_s`. The failed fixture must assert that missing phases remain `null` and the job is marked incomplete instead of silently becoming zero.

- [ ] **Step 2: Run parser tests and verify they fail**

```bash
uv run pytest experiments/swe_rollout_latest_main_ab/tests/test_parse_results.py -q
```

Expected: failure because `parse_results.py` does not exist.

- [ ] **Step 3: Implement the parser and a branch-parameterized launcher**

`submit_lyris.sh` must require `ARM`, `NEMO_RL_SHA`, and `NODE_LOCAL`; reject a dirty remote checkout; pass `NRL_OH_NODE_LOCAL=$NODE_LOCAL`; enable CUDA Graph with the frozen explicit list; write `date +%s.%N` immediately before service setup and after the result manifest is complete; save `git rev-parse HEAD`, Gym SHA, container SHA, dataset SHA, `pip show vllm`, resolved Hydra config, and `sacct` fields in the result directory.

- [ ] **Step 4: Run local harness verification**

```bash
bash -n experiments/swe_rollout_latest_main_ab/submit_lyris.sh
uv run pytest experiments/swe_rollout_latest_main_ab/tests/test_parse_results.py -q
```

Expected: exit code 0.

- [ ] **Step 5: Commit and push the harness**

```bash
git add experiments/swe_rollout_latest_main_ab
git commit -s -m "bench: add latest-main SWE rollout A/B harness"
git push
```

- [ ] **Step 6: Run SSH and scheduling preflight**

```bash
ssh -O check login-lyris
ssh login-lyris 'sacctmgr -nP show assoc where user=$(whoami) format=account,fairshare; sinfo -p gb200 -t idle,alloc -o "%15P|%10t|%6D|%N"; lfs quota -hu sna /lustre/fsw | head -20'
```

Expected: a usable account, GB200 capacity or a recorded pending reason, and sufficient quota. Use `coreai_dlalgo_llm` and partition `gb200`; do not use the obsolete `batch` partition from old notes.

- [ ] **Step 7: Submit one n=1 correctness canary per arm**

Run `submit_lyris.sh` with the same frozen settings except `NUM_PROMPTS=1`, `NUM_GENS=1`, and `CONCURRENCY=1`. A canary passes only if the result is valid, the resolved vLLM version is 0.25.1, CUDA Graph capture covers the realized verification batch, and the node-local arm records both staging and container-path rewriting.

- [ ] **Step 8: Monitor each running canary for at least five minutes**

```bash
ssh login-lyris 'squeue -u sna -h -o "%i|%j|%T|%r|%M|%S" | grep "swe-ab-"; find /lustre/fsw/coreai_dlalgo_llm/users/sna/swe_rollout_latest_main_ab/jobs -type f -name slurm.out -print0 2>/dev/null | xargs -0 ls -1t | head -4 | xargs -r -n1 tail -n 100'
```

Expected: no version mismatch, Ray worker lease failure, eager fallback, mount collision, import error, or missing result manifest. Record every failed canary, cause, and exact corrective commit in `attempts.md` before resubmission.

### Task 5: Run the A/B Gate and Publish the Decision

**Files:**
- Create: `experiments/swe_rollout_latest_main_ab/report/results.csv`
- Create: `experiments/swe_rollout_latest_main_ab/report/summary.json`
- Create: `experiments/swe_rollout_latest_main_ab/report/README.md`
- Modify: `docs/nemogym_init_framework_fixes.html`
- Modify: `docs/nemogym_swe_efficiency_report.html`
- Modify: `docs/specdec_reports_index_latest.html`

**Interfaces:**
- Consumes: passing canaries and parsed per-job JSON.
- Produces: an evidence-backed keep/drop decision for the node-local patch and separate estimates for #3390 and #3283.

- [ ] **Step 1: Submit the primary n=24 sequence in ABBA order**

Use four independent job pairs per comparison. For upstream-software effects, compare main against main+#3390 and main+#3390 against main+#3390+#3283. For our patch, compare main+#3390+#3283 against the same revision plus node-local staging. Keep identical requested walltime and resources; do not run two arms inside one allocation because the node-local cache would contaminate setup accounting.

- [ ] **Step 2: Monitor startup and collect completed job records**

```bash
ssh login-lyris 'squeue -u sna -h -o "%i|%j|%T|%r|%M|%S" | grep "swe-ab-"; sacct -X -S 2026-07-31 --name swe-ab-main,swe-ab-3390,swe-ab-3283,swe-ab-node-local --format=JobID,JobName,State,ExitCode,Elapsed,Start,End -P -n'
```

Expected: at least four valid paired jobs per comparison. Failed or invalid jobs stay in `results.csv` with their failure class and are excluded only by an explicit `included=false` field.

- [ ] **Step 3: Compute paired effects and setup-inclusive break-even**

`summary.json` must report paired median and bootstrap 95% confidence interval for allocation-to-result wall, each phase, valid rate, reward, and generation throughput. For node-local staging, report both observed job wall and the phase-sum view, and compute `break_even_rollouts = ceil(staging_s / per_rollout_connect_framework_savings)` from paired medians.

- [ ] **Step 4: Apply the decision gates**

Keep the node-local optimization as PR-worthy only if the setup-inclusive n=24 paired median is non-regressive, the 95% interval does not include a regression larger than 2%, valid rollout rate falls by no more than one percentage point, generation throughput falls by no more than 2%, and no new failure class appears. If n=24 is inconclusive but per-rollout Connect+Framework savings remain at least 5 seconds, run the same four-pair gate at n=80 before deciding.

- [ ] **Step 5: Update the HTML pages with measured and unmeasured results clearly separated**

Add a dated “Latest-main revalidation” section with the four SHAs, container and vLLM version, CUDA Graph evidence, full wall-time breakdown, paired effect intervals, break-even, failures, and final keep/drop decision. Preserve the historical 47%/49.3-second breakdown as historical evidence and label the workspace-cache target as projected.

- [ ] **Step 6: Verify report consistency**

```bash
python - <<'PY'
import csv
import json
from pathlib import Path

root = Path("experiments/swe_rollout_latest_main_ab/report")
rows = list(csv.DictReader((root / "results.csv").open()))
summary = json.loads((root / "summary.json").read_text())
assert rows, "results.csv is empty"
assert {"main", "pr3390", "pr3390_pr3283", "node_local"} <= {row["arm"] for row in rows}
assert summary["vllm_version"] == "0.25.1"
assert summary["cudagraph_enabled"] is True
for page in [
    Path("docs/nemogym_init_framework_fixes.html"),
    Path("docs/nemogym_swe_efficiency_report.html"),
    Path("docs/specdec_reports_index_latest.html"),
]:
    text = page.read_text()
    assert "Latest-main revalidation" in text
    assert summary["baseline_sha"] in text
PY
```

Expected: exit code 0.

- [ ] **Step 7: Commit the complete evidence bundle**

```bash
git add experiments/swe_rollout_latest_main_ab/report experiments/swe_rollout_latest_main_ab/attempts.md docs/nemogym_init_framework_fixes.html docs/nemogym_swe_efficiency_report.html docs/specdec_reports_index_latest.html
git commit -s -m "docs: report latest-main SWE rollout overhead A/B"
git push
```
