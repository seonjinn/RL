# Qwen3-30B-A3B HybridEP 200-Step Resumed A/B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete 200 comparable all-to-all and HybridEP Qwen3-30B-A3B steps by checkpointing before each four-hour allocation expires and resuming in dependent rounds.

**Architecture:** Keep the existing 20-step launcher defaults unchanged while adding opt-in step, walltime, and checkpoint overrides. A dedicated experiment wrapper creates dispatcher-specific checkpoint roots and round-specific log roots; NeMo-RL automatically resumes from the latest checkpoint.

**Tech Stack:** Bash, SLURM, NeMo-RL synchronous GRPO, Megatron distributed checkpoints, offline W&B, GCP-NRT B200.

## Global Constraints

- Use the canonical `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n8g.yaml` recipe at four nodes and eight GPUs per node.
- Use NeMo-RL `541413bd2912561950413b39809db40590a652bb`, MCore `34b55f24f0826c9aebd6693ecb60648cd934737d`, and DeepEP `17cfb817bccec3a9c247013360cc550c2bac441e`.
- Set `grpo.max_num_steps=200`, SLURM walltime `04:00:00`, and internal checkpoint deadline `00:03:15:00`.
- Save only at the internal deadline or final Step 200, retain only the latest full checkpoint, and include optimizer state.
- Exclude `pool0-0167,pool0-0272,pool0-0337` and request all eight GPUs on every node.
- Compare Steps 2–200 after merging and deduplicating all round histories.

---

### Task 1: Parameterize Step Count, Experiment Root, and Walltime

**Files:**
- Modify: `experiment_logs/pr2964-20step-20260807/performance_case.sh`
- Modify: `experiment_logs/pr2964-20step-20260807/submit_performance_20step.sh`
- Test: `experiment_logs/pr2964-20step-20260807/tests/test_performance_case.sh`

**Interfaces:**
- Consumes: `MAX_NUM_STEPS_OVERRIDE`, `EXPERIMENT_ROOT_OVERRIDE`, and `TIME_LIMIT_OVERRIDE`.
- Produces: resolved driver arguments and `submission.env` provenance while preserving the prior 20-step defaults.

- [x] **Step 1: Write and run failing override tests**
- [x] **Step 2: Implement minimal optional overrides**
- [x] **Step 3: Run tests and verify GREEN**
- [x] **Step 4: Commit with DCO sign-off**

Commit: `d7412b040` (`test: parameterize long HybridEP performance runs`).

### Task 2: Add Checkpoint and Resume Overrides

**Files:**
- Modify: `experiment_logs/pr2964-20step-20260807/performance_case.sh`
- Modify: `experiment_logs/pr2964-20step-20260807/tests/test_performance_case.sh`
- Modify: `experiment_logs/pr2964-q30-4hour-20260809/PLAN.md`
- Modify: `experiment_logs/pr2964-q30-4hour-20260809/submit_q30_4hour.sh`

**Interfaces:**
- Consumes: checkpoint-enabled flag, checkpoint directory, save period, internal deadline, metric name, retention count, and optimizer-save flag.
- Produces: `driver_args` that enable checkpointing only when explicitly requested; a wrapper accepting `{baseline|hybridep} {test-only|submit} ROUND`; separate run names per round; one persistent checkpoint root per dispatcher.

- [x] **Step 1: Write the failing checkpoint contract tests**

Assert that default rendering still contains `checkpointing.enabled=false`. With checkpoint overrides, require:

```text
checkpointing.enabled=true
checkpointing.checkpoint_dir=/tmp/q30-checkpoints
checkpointing.save_period=200
checkpointing.checkpoint_must_save_by=00:03:15:00
checkpointing.metric_name=null
checkpointing.keep_top_k=1
checkpointing.ft_keep_latest_k=1
checkpointing.save_optimizer=true
```

Require the experiment wrapper to set `MAX_NUM_STEPS_OVERRIDE=200`, accept a round number, and derive round-specific run names while keeping a dispatcher-specific checkpoint path.

- [x] **Step 2: Run tests and verify RED**

```bash
bash experiment_logs/pr2964-20step-20260807/tests/test_performance_case.sh
```

Expected: failure because checkpoint overrides and round-aware wrapper behavior are absent.

- [x] **Step 3: Implement optional checkpoint arguments**

When `CHECKPOINTING_ENABLED_OVERRIDE=true`, replace the default disabled argument and append the seven exact checkpoint arguments from Step 1. Reject an empty checkpoint directory. Leave every default invocation unchanged.

- [x] **Step 4: Implement the round-aware wrapper**

Validate `ROUND` as `1`, `2`, or `3`; set:

```bash
RUN_NAME_OVERRIDE=qwen3-30ba3b-sync-${dispatcher}-pr2964-200step-round${round}
CHECKPOINT_DIR_OVERRIDE=${EXPERIMENT_ROOT_OVERRIDE}/checkpoints/${dispatcher}
```

Pass a caller-supplied `JOB_DEPENDENCY` through to the generic launcher.

- [x] **Step 5: Verify GREEN and commit**

Run the Bash test, chart unit tests, `bash -n` for all modified launchers, and `git diff --check`. Commit with DCO sign-off.

### Task 3: Publish, Validate, and Submit Two Resume Chains

**Files:**
- Remote launcher checkout: `/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/experiments/pr2964-20step-20260807/RL-report-launcher-acaa33bcd`
- Remote validation checkout: `/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/experiments/pr2964-20step-20260807/RL-latest-bridge-validation-20260808`
- Remote experiment root: `/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/experiments/pr2964-q30-4hour-20260809`

**Interfaces:**
- Consumes: pushed launcher branch and frozen validation/MCore source.
- Produces: six SLURM job IDs arranged as two independent three-round `afterok` chains, plus immutable submission records and checkpoints.

- [ ] **Step 1: Push and pull the exact launcher branch**

Push `sna/pr2964-hybridep17cf-20step-20260807`, fast-forward the remote launcher checkout, attach the validation checkout to its tracking branch, and run `git pull --ff-only` before submission.

- [ ] **Step 2: Run scheduler probes**

Run `test-only` for baseline Round 1 and HybridEP Round 1. Require both to request four nodes × eight GPUs without a partial-node allocation.

- [ ] **Step 3: Submit the baseline chain**

Submit Round 1 without a dependency, Round 2 with `afterok:<round1>`, and Round 3 with `afterok:<round2>`. Record every parsable job ID.

- [ ] **Step 4: Submit the HybridEP chain**

Use the same dependency structure with its independent checkpoint directory and record every job ID.

- [ ] **Step 5: Monitor active startup for five minutes**

Poll bounded `squeue` and log tails. Verify source/container/wheel imports, Ray startup, and no fatal before leaving the chains scheduled.

- [ ] **Step 6: Measure checkpoint overhead**

After the first successful save, record checkpoint bytes, file count, save duration, saved step, and whether the next round reports loading that exact step.

### Task 4: Analyze Steps 2–200

**Files:**
- Create after completion: `experiment_logs/pr2964-q30-4hour-20260809/results/summary.json`
- Modify after completion: `experiment_logs/pr2964-20step-20260807/report/index.html`

**Interfaces:**
- Consumes: authoritative offline W&B histories, checkpoint metadata, and SLURM logs from every round.
- Produces: 200-step correctness and non-checkpoint performance comparisons with included/missing counts.

- [ ] **Step 1: Merge round histories by training step**

Require Steps 2–200 exactly once after deterministic deduplication and disclose every missing, duplicate, null, NaN, or Inf record.

- [ ] **Step 2: Compare correctness metrics**

Summarize reward, loss, `train/gen_kl_error`, valid samples, token work, and all common validation checkpoints, including pre/post-resume continuity.

- [ ] **Step 3: Compare performance metrics**

Use logged E2E, generation, policy-training, and LogProb timing/throughput. Exclude steps that include checkpoint saving from dispatcher-only means and report their operational overhead separately.

- [ ] **Step 4: Update and validate the HTML report**

Add the 200-step resumed result, checkpoint storage/time overhead, and accuracy conclusion; parse the HTML, run chart tests, visually inspect the page, and push the report commit.
