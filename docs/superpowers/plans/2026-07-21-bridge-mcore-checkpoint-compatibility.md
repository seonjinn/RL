# Bridge–MCore Checkpoint Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the verified PR5672 MCore commit compatible with NeMo-RL's Bridge checkpointing path, then resume the bounded Nano CUDA Graph validation matrix from a source-provenance-correct fresh clone.

**Architecture:** A small Bridge fork branch replaces removed MCore default strategy helpers with direct `TorchDistLoadShardedStrategy` and `TorchDistSaveShardedStrategy` construction. NeMo-RL pins that Bridge branch and places its `src` directory before the container copy at runtime. A strict locked probe must prove every framework module comes from the fresh worktree before a training job is submitted.

**Tech Stack:** Megatron-Bridge, Megatron-Core, NeMo-RL, uv, PyTorch, Transformer Engine, SLURM/Ptyche.

## Global Constraints

- Preserve MCore `experiment/pr5672-nano-packed-support-20260720` at `5be63d9f4ed589cb4dcfefd0c50a2bab85d26b3b`; do not restore its removed deprecated checkpoint APIs.
- Bridge compatibility branch: `experiment/pr5672-nano-checkpoint-compat-20260721` on `seonjinn/Megatron-Bridge`.
- Forward-port only the direct strategy migration from upstream Bridge; do not merge unrelated current-Bridge changes.
- NeMo-RL must pin the Bridge gitlink and `.gitmodules` URL to the seonjinn fork so fresh recursive initialization can fetch both non-upstream submodule commits.
- The strict runtime probe must require `megatron.bridge.training.checkpointing.__file__` under the fresh Bridge `src` directory, never `/opt/nemo-rl`.
- Use `uv run --locked --extra mcore`; update `uv.lock` only if a temporary Linux regeneration yields a small Bridge-only metadata delta with no package version/source-map drift.
- All remote worktrees must `git pull --ff-only`, initialize recursive submodules, and pass scheduler `--test-only` before each actual job.
- Warmup is exactly three; checkpoint saving is disabled. Run no-CG five-step smoke before Mamba five-step smoke; run attn/router only as preflight checks; submit matched 20/40-step jobs only after both smokes pass.

---

### Task 1: Forward-port the direct Bridge checkpoint strategy migration

**Files:**
- Modify: `src/megatron/bridge/training/checkpointing.py`
- Modify: `tests/unit_tests/training/test_checkpointing.py`

**Consumes:** Bridge base `554c7b9324225aa863eee52e8b8fdde7abced2b1`; MCore `5be63d9f4` where both default strategy helpers are absent.

**Produces:** A signed Bridge fork commit importable with MCore `5be63d9f4`.

- [ ] **Step 1: Create the isolated Bridge branch**

```bash
git -C /Users/sna/CudaGraph_PR/RL-pr5672-nano-extension-20260720/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge \
  worktree add -b experiment/pr5672-nano-checkpoint-compat-20260721 \
  /Users/sna/CudaGraph_PR/Megatron-Bridge-pr5672-checkpoint-compat-20260721 \
  554c7b9324225aa863eee52e8b8fdde7abced2b1
git -C /Users/sna/CudaGraph_PR/Megatron-Bridge-pr5672-checkpoint-compat-20260721 \
  remote set-url origin git@github-seonjinn:seonjinn/Megatron-Bridge.git
```

- [ ] **Step 2: Establish the red import against the pinned MCore source**

Run the following in a fresh Ptyche test container with `PYTHONPATH` ordered as MCore `5be63d9f4`, then Bridge `src`:

```bash
python -c 'import megatron.bridge.training.checkpointing'
```

Expected: fail because `get_default_load_sharded_strategy` or `get_default_save_sharded_strategy` is unavailable from `megatron.core.dist_checkpointing.serialization`.

- [ ] **Step 3: Apply the minimal direct-strategy migration**

Replace the obsolete imports with:

```python
from megatron.core.dist_checkpointing.serialization import StateDict
from megatron.core.dist_checkpointing.strategies.torch import (
    TorchDistLoadShardedStrategy,
    TorchDistSaveShardedStrategy,
    _get_filesystem_reader,
)
```

Replace all three helper call sites with these exact direct constructions:

```python
# save_checkpoint(), non-torch_dist branch
save_strategy = TorchDistSaveShardedStrategy()

# _load_model_state_dict() and load_checkpoint()
load_strategy = TorchDistLoadShardedStrategy()
```

Update affected checkpoint unit-test decorators and injected arguments from
`get_default_save_sharded_strategy` to `TorchDistSaveShardedStrategy`, then
make their mocked constructor return the same strategy mock.

- [ ] **Step 4: Verify green and push Bridge**

Run the focused Bridge checkpoint test with MCore `5be63d9f4`, then the direct import command. Both must pass. Commit and push:

```bash
git add src/megatron/bridge/training/checkpointing.py tests/unit_tests/training/test_checkpointing.py
git commit -s -m "fix: use direct MCore checkpoint strategies"
git push -u origin experiment/pr5672-nano-checkpoint-compat-20260721
```

### Task 2: Pin Bridge and prove runtime source provenance

**Files:**
- Modify: `.gitmodules`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge`
- Modify: `experiments/cuda_graph/launch_nanov3_packed_cg_scope_ptyche.sh`
- Create: `experiments/cuda_graph/probe_nemo_mcore_bridge_runtime_ptyche.sbatch`
- Modify: `tests/unit/experiments/test_nanov3_cuda_graph_launcher.py`
- Modify: `uv.lock` only if the constrained lock check requires it.

**Consumes:** Task 1 Bridge commit and MCore `5be63d9f4`.

**Produces:** Fresh recursive clone resolves Bridge/MCore fork commits, and the probe proves Bridge code originates from the fresh checkout.

- [ ] **Step 1: Write failing launcher/provenance tests**

Add a launcher contract assertion that `BRIDGE_SRC` is the `Megatron-Bridge/src`
directory and appears before the Bridge project root in the printed/exported
Python path. Add a probe-script test that requires these output keys:

```text
strict_locked_combined_import=passed
bridge_checkpointing_file=<fresh-worktree>/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src/megatron/bridge/training/checkpointing.py
```

Run the focused test before the launcher/probe change. Expected: fail because
the launcher does not define `BRIDGE_SRC` and the probe script does not exist.

- [ ] **Step 2: Pin submodule and add explicit Bridge source precedence**

Set the Bridge submodule URL and branch in `.gitmodules`:

```ini
[submodule "3rdparty/Megatron-Bridge"]
    path = 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
    url = https://github.com/seonjinn/Megatron-Bridge.git
    branch = experiment/pr5672-nano-checkpoint-compat-20260721
```

Check out the Task 1 Bridge commit in the root submodule. In the Nano launcher,
define and prepend the source directory:

```bash
BRIDGE_SRC="${WORKTREE}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src"
export PYTHONPATH="${WORKTREE}:${WORKTREE}/3rdparty/Megatron-LM-workspace/Megatron-LM:${BRIDGE_SRC}:${WORKTREE}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge:${PYTHONPATH:-}"
```

The probe script must execute `uv run --locked --extra mcore`, import
`nemo_rl`, `mamba_ssm`, `megatron.core`, `transformer_engine.pytorch`, and
`megatron.bridge.training.checkpointing`, then assert the resolved checkpoint
module path is under `${BRIDGE_SRC}` before printing the required keys.

- [ ] **Step 3: Reconcile lock only when required**

Run `uv lock --check`. If it succeeds, do not modify `uv.lock`. If it fails,
regenerate in an isolated Linux copy, require no package version/source-map
change and a Bridge-only metadata diff, then apply exactly that diff to the
real `uv.lock`. If the generated diff changes package versions or unrelated
sources, stop and report the lock drift rather than committing it.

- [ ] **Step 4: Verify and push NeMo-RL integration**

Run launcher/probe tests, commit with `-s`, and push seonjinn. In a fresh
Ptyche HTTPS clone, run `git pull --ff-only`, recursive init, and assert both
submodule source heads. Run `sbatch --test-only` then the strict provenance
probe; it must print the fresh Bridge `src` path and pass.

### Task 3: Resume gated Nano smoke and matched matrix

**Files:**
- Use the persisted launcher and probe scripts from Task 2.
- Write results only under `experiments/cuda_graph/logs/task6-20260721/`.

**Consumes:** Task 2 strict provenance probe.

**Produces:** Valid smoke/preflight evidence and, only after success, matched
20/40-step no-CG versus Mamba results.

- [ ] **Step 1: Submit ordered five-step gates**

From the fresh remote worktree, submit `SCOPE_CASE=nocg STEPS=5 SUBMIT=1` and
observe it for five minutes. Only after it stays healthy, submit
`SCOPE_CASE=mamba STEPS=5 SUBMIT=1` and observe it for five minutes. Both use
the exact launcher defaults `cuda_graph_warmup_steps=3`, checkpoint saving
disabled, max packed sequences 512, and bucket `[8192]`.

- [ ] **Step 2: Run rejected scope preflights**

Submit attn and moe-router only as bounded preflight invocations. Require the
Nano attention and FP64 router errors before `make_graphed_callables`; do not
submit 20/40-step jobs for either rejected scope.

- [ ] **Step 3: Submit matched matrix only after the gates pass**

```bash
SCOPE_CASE=nocg STEPS=20 SUBMIT=1 ./experiments/cuda_graph/launch_nanov3_packed_cg_scope_ptyche.sh
SCOPE_CASE=mamba STEPS=20 SUBMIT=1 ./experiments/cuda_graph/launch_nanov3_packed_cg_scope_ptyche.sh
SCOPE_CASE=nocg STEPS=40 SUBMIT=1 ./experiments/cuda_graph/launch_nanov3_packed_cg_scope_ptyche.sh
SCOPE_CASE=mamba STEPS=40 SUBMIT=1 ./experiments/cuda_graph/launch_nanov3_packed_cg_scope_ptyche.sh
```

Use the same seed and topology. Report only completed post-warmup medians for
E2E/generation/logprob/policy time and tokens/s/GPU, then 40-step reward,
accuracy, policy loss, KL, clip ratio, NaN/invalid count, and completion rate.
