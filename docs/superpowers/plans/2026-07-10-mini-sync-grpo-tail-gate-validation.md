# Mini Sync-GRPO Tail-Gate Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and run a two-step Qwen3-32B synchronous GRPO smoke that proves FastRL-style tail gating activates at an observable inflight batch without changing the production performance recipe topology.

**Architecture:** Extend the existing cumulative vLLM telemetry with an activation-only scheduler tick, expose it through the current W&B derivation, and reuse the matched tail-gate launcher through a smoke-sized wrapper. The wrapper derives local scheduler capacity from global rollouts and DP, while the collector and validator consume the resolved local threshold from manifest provenance. The three-arm baseline/always-on/threshold matrix is validated before the exact fetched fork commit is submitted on Pre-Tyche.

**Tech Stack:** Python 3.13, Bash, NeMo-RL, vLLM 0.24.0, pytest, W&B, SLURM/Pyxis, Pre-Tyche GB200.

## Global Constraints

- Work on `sna/nemorl-vllm024-tail-gating`; push only to personal remote `fork`, never NVIDIA `origin`.
- Use Qwen3-32B `examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml` and preserve its model and parallel topology.
- Use four nodes, four GPUs per node, `--segment=4`, no `--gres`, and Triton MoE.
- V2 arms use `VLLM_USE_V2_MODEL_RUNNER=1`, `enforce_eager=false`, and `FULL_AND_PIECEWISE` CUDA graphs.
- Sampling remains temperature 1.0, top-p 1.0, standard rejection, and probabilistic Eagle drafting.
- Smoke geometry defaults to 16 prompts, 4 generations, train GBS 64, OSL/total length 1024, max model length 1056, and 2 steps.
- The mini FastRL threshold defaults to 4 local active requests and 10
  consecutive decode checks. With 64 global rollouts and DP8, each local
  scheduler starts with eight requests; global rollout count is not the gate
  threshold.
- Production `all` selections exclude Qwen3-30B-A3B V2 arms until
  commit-scoped V2 smoke evidence exists.
- Submit fetches `fork/<current-branch>` and requires exact equality between
  local `HEAD` and `refs/remotes/fork/<current-branch>`.
- Checkpointing remains disabled and no experimental checkpoint is written.
- Preserve untracked `tests/unit/unit_results.json` and `tests/unit/unit_results/`.

---

### Task 1: Activation Scheduler-Tick Telemetry

**Files:**
- Modify: `nemo_rl/models/generation/vllm/patches.py`
- Modify: `nemo_rl/models/generation/vllm/utils.py`
- Modify: `tests/unit/models/generation/test_vllm_patches.py`
- Modify: `tests/unit/models/generation/test_vllm_utils.py`
- Modify fixture snapshots under `tests/unit/models/generation/fixtures/vllm_v0_24_0/`

**Interfaces:**
- Consumes: `SchedulerOutput.tail_gate_tick` and `tail_gate_just_activated`.
- Produces: raw cumulative `vllm:spec_decode_tail_gate_activation_tick_sum/count` and derived `vllm/tail_gate_activation_tick`.

- [ ] **Step 1: Add failing producer tests**

Extend the self-contained V1/V2 patch fixtures so one non-activation decision
followed by one activation decision proves:

```python
assert metrics["vllm:spec_decode_tail_gate_activation_tick_sum"] == 17.0
assert metrics["vllm:spec_decode_tail_gate_activation_tick_count"] == 1.0
```

The non-activation tick must not contribute.

- [ ] **Step 2: Verify producer RED**

Run:

```bash
uv run --no-sync pytest --noconftest -o addopts='' \
  tests/unit/models/generation/test_vllm_patches.py -k activation_tick -q
```

Expected: missing activation tick counters.

- [ ] **Step 3: Add activation-only counters**

In both V1 and V2 telemetry patches, update tick sum/count inside the existing
`tail_gate_just_activated` block. Preserve the all-step tick/state counters and
all fixed-K behavior.

- [ ] **Step 4: Add failing W&B derivation tests**

Use cumulative snapshots where all scheduler ticks total 100 but the activation
tick is 17. Assert:

```python
assert metrics["vllm/tail_gate_activation_tick"] == 17.0
```

Also assert zero count returns 0.0.

- [ ] **Step 5: Implement derivation and verify GREEN**

Delta activation tick sum/count with the current cumulative metric machinery,
then derive their zero-safe ratio. Run the two focused test files, Ruff, format,
and `git diff --check`.

- [ ] **Step 6: Commit**

```bash
git commit -s -m "bench(vllm): report tail gate activation tick"
```

### Task 2: Mini Sync-GRPO Launcher

**Files:**
- Modify: `experiments/vllm_024_upgrade/submit_tail_gated_specdec_step20.sh`
- Create: `experiments/vllm_024_upgrade/submit_tail_gated_specdec_mini_sync_grpo.sh`
- Modify: `tests/test_vllm_024_tail_gate_launch.py`
- Create: `tests/test_vllm_024_tail_gate_mini_sync_grpo.py`
- Modify: `experiments/vllm_024_upgrade/README.md`

**Interfaces:**
- Consumes: existing matched launcher variants and cluster environment overrides.
- Produces: `dry-run`, `test-only`, and `submit` for the exact three-arm mini matrix.

- [ ] **Step 1: Add failing parameterization tests**

Assert the main launcher honors:

```text
TAIL_GATE_THRESHOLD=32
TAIL_GATE_CONSECUTIVE_CHECKS=10
```

and records the resolved values in commands and manifests rather than using
hard-coded values. Also assert submit fetches `fork/<current-branch>` and
rejects ahead, behind, diverged, fetch-failed, and stale states unless local
`HEAD` exactly equals the fetched remote-tracking ref.

- [ ] **Step 2: Add failing mini-wrapper tests**

Dry-run the wrapper and assert exactly three independent arms:

```text
baseline_v2 always_on_v2_k5 fastrl_threshold_v2_k5
```

Every arm must render Qwen3-32B 4n4g, 16x4 rollouts, GBS64, OSL/sequence 1024,
max model length 1056, max steps 2, V2, FULL_AND_PIECEWISE, no checkpoints,
four nodes, `--segment=4`, no `--gres`, and explicit W&B metadata. DP8 gives
eight initial requests per local scheduler, so only the threshold arm sets the
gate scheduler with local threshold 4 and ten checks. Assert the wrapper rejects
a threshold greater than or equal to the computed local capacity.

- [ ] **Step 3: Implement environment-backed threshold settings**

Add validated positive integer environment values to the main launcher and use
them for threshold and roofline variants. Keep current production 32/10
defaults. Exclude Qwen3-30B-A3B V2 arms whenever either model or variant
selection uses `all`; explicit model-plus-variant selection remains available
for commit-scoped support smoke evidence.

- [ ] **Step 4: Implement the mini wrapper**

The wrapper exports smoke defaults while allowing explicit caller overrides,
then invokes the existing launcher once per selected arm. It must propagate a
failure immediately and use a shared run tag/project so one manifest contains
all three jobs. Override the production threshold with local threshold 4 and
validate it is positive and strictly below
`(NUM_PROMPTS * NUM_GENERATIONS) / DP`.

- [ ] **Step 5: Verify and commit**

Run both launcher suites, `bash -n` on both scripts, Ruff, and diff checks.

```bash
git commit -s -m "bench(vllm): add mini sync grpo tail gate smoke"
```

### Task 3: Activation Event Report and Functional Validator

**Files:**
- Modify: `experiments/vllm_024_upgrade/summarize_tail_gated_specdec.py`
- Modify: `tests/test_vllm_024_tail_gate_summary.py`
- Create: `experiments/vllm_024_upgrade/validate_mini_sync_grpo_tail_gate.py`
- Create: `tests/test_vllm_024_tail_gate_mini_validation.py`

**Interfaces:**
- Consumes: mini manifest and production W&B histories.
- Produces: validated mini JSON/CSV plus an activation-event HTML scatter plot.

- [ ] **Step 1: Add failing activation-metric tests**

Require `train/vllm/tail_gate_activation_tick` only for gated arms. Add it to
the row schema and preserve variant-aware baseline behavior.

- [ ] **Step 2: Add failing functional-gate tests**

For a completed threshold run, require positive activation tick, activation
batch in `[1, threshold]`, both enabled and advance-only ratios in `(0, 1)`, K0
and K5 scheduler counters, positive proposals/accepts, finite accuracy metrics,
and completion through policy training. Every failure must mark the row partial
or health-failed and return nonzero.

- [ ] **Step 3: Add failing chart tests**

Render deterministic HTML with scheduler tick on x, inflight batch on y, a
reference line taken from each manifest row's local threshold, and an annotated
OFF-to-ON activation point. Test that both axis labels, the manifest threshold,
and exact event values appear and shuffled input produces byte-identical
output. Do not hard-code 32 or infer the threshold from global rollout count.

- [ ] **Step 4: Implement collector and validator changes**

Reuse the collector's exact cohort schema and atomic output rules. The mini
validator may call collector helpers but must not duplicate baseline matching.
Do not claim stable speedup from two steps.

- [ ] **Step 5: Verify and commit**

Run both summary/mini validation suites, Ruff, Pyright, and diff checks.

```bash
git commit -s -m "report: validate mini sync grpo tail gating"
```

### Task 4: Local Integration and Pre-Tyche Smoke

**Files:**
- Generated only: Lustre run manifest, SLURM logs, W&B URLs, validated summary.

**Interfaces:**
- Consumes: Tasks 1-3 at one pushed commit.
- Produces: three completed two-step functional smoke rows.

- [ ] **Step 1: Run focused local integration**

Run all tail-gate, patch, metric, launcher, collector, and mini validation tests.
Require Ruff/format/Bash/diff checks and a clean tracked worktree.

- [ ] **Step 2: Push personal fork and update Pre-Tyche**

Push only `fork/sna/nemorl-vllm024-tail-gating`. On `login-ptyche`, update
`/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-vllm024-upgrade-20260707`, initialize
recursive submodules, and verify HEAD and container provenance. Fetch the
current branch and require local `HEAD` to equal
`refs/remotes/fork/sna/nemorl-vllm024-tail-gating` exactly before submission.

- [ ] **Step 3: Run Pre-Tyche test-only**

Use account `coreai_dlalgo_llm`, partition `batch`, four nodes, `--segment=4`,
no GRES, 20260705 nightly image, Lustre HF/W&B paths, and the mini wrapper's
`test-only` mode. Require all three scheduler checks to pass.

- [ ] **Step 4: Submit and monitor**

Submit the three arms, record job IDs and W&B URLs, and monitor for at least five
minutes and through the first policy-training step. Stop and fix on any stale
draft ID, invalid token, NaN, OOM, NCCL, q-cache, graph fallback, or missing
telemetry error.

- [ ] **Step 5: Validate and report**

Run the mini validator after completion, render the activation chart, and record
the exact activation tick/inflight batch. Only after this passes may the full
20-step matrix be submitted.
