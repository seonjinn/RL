# Llama and Qwen Async Force-On-Policy Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Submit eight native-four-GPU-node, 20-step Pre-Tyche jobs that compare `force_on_policy_ratio=false` and `true` for Llama 3.1 8B sync/async and Qwen3-30B-A3B/Qwen3-32B async-1off recipes.

**Architecture:** Add a new committed experiment directory without changing the pinned benchmark source. A TSV manifest drives a resolved-config validator, immutable runner, and test-only/real submitter. Existing pinned-container unit-test job `2338113` supplies behavioral-test evidence; the new validator proves the four recipe pairs and the submitter records every SLURM job before five-minute monitoring.

**Tech Stack:** NeMo-RL, OmegaConf/Hydra, Bash, Python 3.13, SLURM, Ray, Pyxis/Enroot, W&B, Pre-Tyche GB200-NVL36.

## Global Constraints

- Source branch: `sna/nemorl-main-pr3030-q235-20260701`.
- Source SHA: `d4cfecf90db41cdf142629963b54b67ab479ab02`.
- Container: `/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260630_0215.sqsh`.
- Container SHA-256: `bf841732e6615aca7a00a6c4ba47d7298a118137fc914296a4083172132ff510`.
- Pre-Tyche account `coreai_dlalgo_llm`, partition `36x2-a01r`, exclusive allocations, `--comment=metrics`, and no `--gres`.
- Only filenames with native `4g` topology are in scope.
- Both sides use `policy.train_global_batch_size=2048`, 20 steps, and disabled checkpoints.
- The only paired difference is `loss_fn.force_on_policy_ratio`.
- Do not add timeout, backend, HybridEP, or source-code overrides.
- Do not run a two-step model smoke; run config validation and `sbatch --test-only` before direct 20-step submission.

---

### Task 1: Create and validate the eight-case contract

**Files:**
- Create: `experiments/pretyche_force_on_policy_ratio_llama_qwen_async_20260707/README.md`
- Create: `experiments/pretyche_force_on_policy_ratio_llama_qwen_async_20260707/manifests/config_contract.tsv`
- Create: `experiments/pretyche_force_on_policy_ratio_llama_qwen_async_20260707/scripts/test_matrix_contract.sh`
- Create: `experiments/pretyche_force_on_policy_ratio_llama_qwen_async_20260707/scripts/validate_config_contract.py`

**Interfaces:**
- Consumes: the four approved native-4g recipe names and pinned source checkout.
- Produces: eight manifest rows and `CONFIG_OK`/`PAIR_OK` output proving paired equality.

- [ ] **Step 1: Write the failing static contract test**

The test must require eight rows, four control and four force cases, GBS 2048,
20 steps, four recipe names, native four-GPU topology, expected node/segment
pairs `(2,2)`, `(4,4)`, and `(8,8)`, plus the validator file.

Run:

```bash
bash experiments/pretyche_force_on_policy_ratio_llama_qwen_async_20260707/scripts/test_matrix_contract.sh
```

Expected: non-zero because the manifest and implementation files do not exist.

- [ ] **Step 2: Add the exact TSV matrix**

Use these rows after a tab-separated header:

```text
llama_2n4g_sync_control       llama3.1-8b  grpo-llama3.1-8b-instruct-2n4g                 sync       false  2  4  2  2048  20  02:00:00
llama_2n4g_sync_force         llama3.1-8b  grpo-llama3.1-8b-instruct-2n4g                 sync       true   2  4  2  2048  20  02:00:00
llama_2n4g_async1off_control  llama3.1-8b  grpo-llama3.1-8b-instruct-2n4g-async-1off      async1off  false  2  4  2  2048  20  02:00:00
llama_2n4g_async1off_force    llama3.1-8b  grpo-llama3.1-8b-instruct-2n4g-async-1off      async1off  true   2  4  2  2048  20  02:00:00
q30_4n4g_async1off_control    qwen3-30ba3b grpo-qwen3-30ba3b-4n4g-async-1off             async1off  false  4  4  4  2048  20  03:00:00
q30_4n4g_async1off_force      qwen3-30ba3b grpo-qwen3-30ba3b-4n4g-async-1off             async1off  true   4  4  4  2048  20  03:00:00
q32_8n4g_async1off_control    qwen3-32b     grpo-qwen3-32b-8n4g-async-1off                 async1off  false  8  4  8  2048  20  04:00:00
q32_8n4g_async1off_force      qwen3-32b     grpo-qwen3-32b-8n4g-async-1off                 async1off  true   8  4  8  2048  20  04:00:00
```

- [ ] **Step 3: Implement the resolved-config validator**

For each row, call `load_config`, apply only these paired overrides, resolve the
OmegaConf object, and assert every design invariant:

```python
overrides = [
    "policy.train_global_batch_size=2048",
    f"loss_fn.force_on_policy_ratio={str(case.force).lower()}",
    f"cluster.segment_size={case.segment}",
]
```

The validator must normalize only the force flag and logger run fields before
comparing each control/force resolved dictionary.

- [ ] **Step 4: Run RED/GREEN contract verification**

Run:

```bash
bash experiments/pretyche_force_on_policy_ratio_llama_qwen_async_20260707/scripts/test_matrix_contract.sh
```

Expected after the manifest and validator exist: `MATRIX_CONTRACT_OK`.

---

### Task 2: Add the immutable runner and direct submitter

**Files:**
- Create: `experiments/pretyche_force_on_policy_ratio_llama_qwen_async_20260707/scripts/run_force_on_policy_benchmark.sbatch`
- Create: `experiments/pretyche_force_on_policy_ratio_llama_qwen_async_20260707/scripts/submit_force_on_policy_matrix.sh`

**Interfaces:**
- Consumes: one manifest row through exported SLURM variables and an expected runner SHA.
- Produces: isolated run logs, W&B runs, and `results/jobs.tsv` containing eight job IDs.

- [ ] **Step 1: Extend the static test and verify RED**

Require pinned source/container identities, GBS 2048, exact force override,
20 steps only, checkpoint disablement, W&B enablement, `TEST_ONLY`, manifest
recording, and duplicate protection. Forbid `--gres`, timeout overrides,
backend overrides, HybridEP overrides, and source mutations.

- [ ] **Step 2: Implement the immutable runner**

The runner must verify source HEAD/upstream/origin/clean recursive submodules,
container SHA, runner SHA, native node count, and token files before running:

```bash
NRL_FORCE_REBUILD_VENVS=false uv run --locked examples/run_grpo.py \
  --config "examples/configs/recipes/llm/performance/${CONFIG_NAME}.yaml" \
  grpo.max_num_steps=20 \
  checkpointing.enabled=false \
  policy.train_global_batch_size=2048 \
  loss_fn.force_on_policy_ratio="${FORCE_ON_POLICY_RATIO}" \
  cluster.segment_size="${SEGMENT_SIZE}"
```

Use W&B project `sna-force-on-policy-llama-qwen-async-gb200` and a unique name
containing run key and `20s`.

- [ ] **Step 3: Implement test-only and real submission**

The submitter must hash the runner, iterate the TSV, and call `sbatch` with the
row's node, segment, and walltime. `TEST_ONLY=1` uses `--test-only`; `TEST_ONLY=0`
uses `--parsable` and writes exactly eight rows to `results/jobs.tsv`.

- [ ] **Step 4: Verify syntax and the complete contract**

Run:

```bash
bash -n experiments/pretyche_force_on_policy_ratio_llama_qwen_async_20260707/scripts/*.sh
bash -n experiments/pretyche_force_on_policy_ratio_llama_qwen_async_20260707/scripts/*.sbatch
bash experiments/pretyche_force_on_policy_ratio_llama_qwen_async_20260707/scripts/test_matrix_contract.sh
git diff --check
```

Expected: all commands pass and the contract prints `MATRIX_CONTRACT_OK`.

- [ ] **Step 5: Commit and push only the new experiment directory**

Run:

```bash
git add experiments/pretyche_force_on_policy_ratio_llama_qwen_async_20260707
git commit -s -m "test: add native-4g force-ratio performance sweep"
git push fork HEAD:sna/q30-q32-force-on-policy-benchmark-20260707
```

Expected: local HEAD equals `@{u}` and the worktree is clean aside from ignored
collected result files.

---

### Task 3: Validate, schedule, submit, and monitor on Pre-Tyche

**Files:**
- Create remotely: `pretyche_force_on_policy_ratio_llama_qwen_async_20260707/results/jobs.tsv`
- Fetch locally: `experiments/pretyche_force_on_policy_ratio_llama_qwen_async_20260707/results/jobs.tsv`

**Interfaces:**
- Consumes: committed harness files and prior pinned-container unit-test evidence from job `2338113`.
- Produces: eight queued/running 20-step SLURM jobs with immutable provenance.

- [ ] **Step 1: Recheck authentication and immutable inputs**

Verify principal, SLURM access, account, FairShare, source pull/no-op at the
pinned SHA, clean submodules, container SHA, and readable token file.

- [ ] **Step 2: Sync committed harness and compare every SHA-256**

Rsync the experiment directory without deleting remote results. Compare sorted
local `shasum -a 256` output with remote `sha256sum` output and require equality.

- [ ] **Step 3: Run the validator and retain unit-test evidence**

Run the validator in the pinned source environment. Record that validation job
`2338113` completed `0:0` with the two selected unit tests and the same source
and container identities.

- [ ] **Step 4: Check all eight allocations with `sbatch --test-only`**

Run:

```bash
TEST_ONLY=1 scripts/submit_force_on_policy_matrix.sh
```

Expected: eight schedulability lines and no manifest creation.

- [ ] **Step 5: Submit the eight direct 20-step jobs**

Run:

```bash
TEST_ONLY=0 scripts/submit_force_on_policy_matrix.sh
```

Expected: eight job IDs and a header plus eight rows in `results/jobs.tsv`.

- [ ] **Step 6: Monitor for five minutes**

Poll all eight IDs every 30 seconds. For running jobs, require source/container
identity lines, W&B URL, actor startup, and no fatal signature. For pending jobs,
record reason and estimated start time.

---

### Task 4: Collect paired correctness and performance results

**Files:**
- Create: `experiments/pretyche_force_on_policy_ratio_llama_qwen_async_20260707/results/metrics.csv`
- Create: `experiments/pretyche_force_on_policy_ratio_llama_qwen_async_20260707/REPORT.md`

**Interfaces:**
- Consumes: terminal logs and TensorBoard events for each completed pair.
- Produces: pair-only speedups and correctness classifications.

- [ ] **Step 1: Require terminal paired outcomes**

Do not calculate a speedup unless both sides have `COMPLETED 0:0`, step 20,
and no fatal signatures.

- [ ] **Step 2: Extract matched steady-state metrics**

Exclude steps 1, 10, and 20. Record step time, tokens/s/GPU, samples/s,
generation, logprob, policy training, preparation, and weight transfer for
steps 2-9 and 11-19.

- [ ] **Step 3: Check correctness curves**

Compare matched reward, loss, generation KL, reference KL, sampling importance
ratio, token probability error, force markers, reference-logprob timing, and
finite-value status. Keep async-1off and sync results separate.

- [ ] **Step 4: Write and verify the final report**

Include job/W&B links, identities, terminal states, sample counts, mean/median,
time reduction, throughput speedup, correctness findings, and limitations.
Run `git diff --check` and verify all reported values against `metrics.csv`.
