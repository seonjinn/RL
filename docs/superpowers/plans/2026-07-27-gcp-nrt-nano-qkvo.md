# GCP-NRT Nano QKVO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tested GCP-NRT B200 launch profile for the five-arm Nemotron 3 Nano GRPO suite and submit 20-step jobs.

**Architecture:** Generalize the existing Nano submitter from a fixed 4x4 Lyris allocation to an exact 16-GPU invariant, following the existing Qwen3-235B GCP profile pattern. Keep cluster-specific paths and Slurm options in a thin GCP wrapper while the shared batch script receives resolved topology, repository SHA, cache, and W&B metadata through exported variables.

**Tech Stack:** Bash, Slurm `sbatch`, Pytest, PyYAML, NeMo-RL, Ray, vLLM, W&B

## Global Constraints

- Use exactly 16 GPUs: 2 GCP-NRT B200 nodes with 8 GPUs per node.
- Use `batch`, `--gpus-per-node=8`, four-hour wall time, and no Slurm `--segment` or `--network`.
- Run BF16, MoE MXFP8 baseline/optimized, and MoE+QKVO MXFP8 baseline/optimized for 20 steps.
- Keep real importance sampling, vLLM TP1, trainer TP2/PP2/CP2/EP8, global batch size 16, seed 42, and checkpointing disabled.
- Use `nvidia/sna-mxfp8-qkvo-nano-gcp-nrt` for W&B.
- Do not initialize submodules on GCP-NRT because the shared filesystem is nearly out of inodes.
- Commit and push the exact source revision before submission and monitor jobs for at least five minutes.

---

### Task 1: Test the Portable Nano Launcher Contract

**Files:**
- Modify: `tests/test_mxfp8_qkvo_nano_recipe.py`

**Interfaces:**
- Consumes: `experiments/mxfp8_qkvo_nano/submit_suite.sh`
- Produces: tests for exact 16-GPU validation, dynamic application topology, immutable SHA propagation, and the GCP profile

- [x] **Step 1: Write failing tests**

Add assertions that the shared submitter accepts a 16-GPU topology, supports
`GPU_REQUEST_MODE=gpus-per-node`, allows empty Slurm network and segment
arguments, passes `EXPECTED_REPO_SHA`, and that `submit_gcp_nrt.sh` defines the
approved 2x8 paths and settings. Extend the fake-`sbatch` test to assert:

```python
assert "--nodes=2" in call
assert "--gpus-per-node=8" in call
assert "--segment=" not in call
assert "--network=" not in call
assert "NUM_NODES=2" in call
assert "GPUS_PER_NODE=8" in call
assert "EXPECTED_REPO_SHA=deadbeef" in call
```

- [x] **Step 2: Run the focused test and verify RED**

Run:

```bash
uv run --isolated --no-project --with pyyaml --with pytest \
  pytest -q tests/test_mxfp8_qkvo_nano_recipe.py
```

Expected: failures for the missing GCP profile and the fixed 4x4 launcher
contract.

- [x] **Step 3: Commit the failing test with the implementation**

The test and launcher form one reviewer-visible deliverable and are committed
together after the GREEN state in Task 2.

### Task 2: Implement the GCP-NRT Nano Profile

**Files:**
- Modify: `experiments/mxfp8_qkvo_nano/submit_suite.sh`
- Modify: `experiments/mxfp8_qkvo_nano/run_arm.sbatch`
- Create: `experiments/mxfp8_qkvo_nano/submit_gcp_nrt.sh`
- Test: `tests/test_mxfp8_qkvo_nano_recipe.py`

**Interfaces:**
- Consumes: `NUM_NODES`, `GPUS_PER_NODE`, `GPU_REQUEST_MODE`,
  `SLURM_NETWORK`, `SLURM_SEGMENT`, `NRL_HF_HOME`, `EXPECTED_REPO_SHA`,
  `WANDB_PROJECT`, `WANDB_ENTITY`, `EXPERIMENT_CLUSTER`, and
  `INIT_SUBMODULES`
- Produces: five Slurm submissions using an immutable repository/container
  pair and dynamic NeMo-RL cluster overrides

- [x] **Step 1: Generalize the shared submitter**

Replace the fixed 4x4 check with:

```bash
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))
if [[ "$TOTAL_GPUS" -ne 16 ]]; then
  echo "Nano suite requires 16 GPUs total, got $TOTAL_GPUS" >&2
  exit 2
fi
```

Add `none`, `gres`, and `gpus-per-node` request modes; optional Slurm network,
segment, comment, and dependency arguments; optional submodule
initialization; an immutable resolved container path; and expected SHA,
topology, cache, W&B, and cluster metadata in `--export`.

- [x] **Step 2: Make the batch script topology- and revision-aware**

Require `NUM_NODES`, `GPUS_PER_NODE`, and `EXPECTED_REPO_SHA`, validate the
checkout with `git rev-parse --git-dir`, and fail if `git rev-parse HEAD`
differs from the submitted SHA. Pass:

```bash
cluster.num_nodes='$NUM_NODES'
cluster.gpus_per_node='$GPUS_PER_NODE'
cluster.segment_size='$NUM_NODES'
```

Keep real importance sampling, vLLM TP1, global batch size 16, seed 42, and
checkpointing disabled. Record topology, precision scope, refit controls,
cluster, container, source SHA, and W&B identity in `metadata.env`.

- [x] **Step 3: Add the GCP wrapper**

Set the verified GCP checkout, Nano model, immutable nightly container, output
root, `batch` account/partition, 2x8 allocation,
`GPU_REQUEST_MODE=gpus-per-node`, empty network and Slurm segment, four-hour
wall time, 120-minute idle-reaper exemption, `INIT_SUBMODULES=0`, and
`nvidia/sna-mxfp8-qkvo-nano-gcp-nrt`.

- [x] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
uv run --isolated --no-project --with pyyaml --with pytest \
  pytest -q tests/test_mxfp8_qkvo_nano_recipe.py
bash -n experiments/mxfp8_qkvo_nano/submit_suite.sh
bash -n experiments/mxfp8_qkvo_nano/submit_gcp_nrt.sh
bash -n experiments/mxfp8_qkvo_nano/run_arm.sbatch
git diff --check
```

Expected: every command exits zero.

- [x] **Step 5: Commit and push**

Run:

```bash
git add \
  docs/superpowers/specs/2026-07-27-gcp-nrt-nano-qkvo-design.md \
  docs/superpowers/plans/2026-07-27-gcp-nrt-nano-qkvo.md \
  experiments/mxfp8_qkvo_nano/submit_suite.sh \
  experiments/mxfp8_qkvo_nano/submit_gcp_nrt.sh \
  experiments/mxfp8_qkvo_nano/run_arm.sbatch \
  tests/test_mxfp8_qkvo_nano_recipe.py
git commit -s -m "perf: add GCP-NRT Nano QKVO suite"
git push fork sna/mxfp8-qkvo-refit-pr3294-ab
```

Expected: the fork branch points to the tested commit.

### Task 3: Validate and Submit on GCP-NRT

**Files:**
- Runtime output: `/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/experiments/mxfp8-qkvo-nano-gcp-nrt/`

**Interfaces:**
- Consumes: pushed branch and the existing GCP checkout/container/model
- Produces: one submission manifest, five Slurm jobs, and five W&B runs

- [ ] **Step 1: Fast-forward the prepared checkout**

Run through the `gcp-nrt` SSH alias:

```bash
git -C /lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/experiments/refit-opt-qwen30b/nemo-rl-refit-opt-r2 \
  pull --ff-only --recurse-submodules=no
```

Expected: the remote checkout reaches the pushed commit without creating
submodule files.

- [ ] **Step 2: Run the scheduler dry run**

```bash
ACTION=test-only \
MAX_STEPS=20 \
RUN_SUFFIX=gcp-nrt-nano-$(date +%Y%m%d-%H%M%S) \
./experiments/mxfp8_qkvo_nano/submit_gcp_nrt.sh
```

Expected: five accepted test-only requests, each with 2 nodes,
`--gpus-per-node=8`, no Slurm segment/network option, and the same source SHA.

- [ ] **Step 3: Submit the five jobs**

```bash
ACTION=submit \
MAX_STEPS=20 \
RUN_SUFFIX=gcp-nrt-nano-$(date +%Y%m%d-%H%M%S) \
./experiments/mxfp8_qkvo_nano/submit_gcp_nrt.sh
```

Expected: the manifest records five numeric Slurm job IDs.

- [ ] **Step 4: Monitor startup for five minutes**

Inspect `squeue`, `sacct`, and bounded tails of each Slurm log. Verify each job
requests 2 nodes and 16 GPUs total, imports NeMo-RL from the expected checkout,
starts the W&B run, and has no traceback, OOM, SHA mismatch, or scheduler
option failure.

- [ ] **Step 5: Report the live state**

Return the source SHA, manifest path, job IDs/states, W&B project link, and any
startup error with the relevant log path.
