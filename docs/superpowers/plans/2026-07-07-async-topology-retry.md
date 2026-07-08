# Async Topology Retry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the six Async-1off control/force jobs to use the official r0.7.0 segment topology, submit them in a fresh Pre-Tyche result namespace, and verify that they pass the prior setup failure.

**Architecture:** Keep the pinned NeMo-RL source and container unchanged. Make the experiment manifest express an optional application/Slurm segment, validate topology after the non-colocated training/inference split, and make the runner and submitter conditionally add segment arguments only for Qwen3-30B-A3B. Preserve the failed jobs and completed Sync jobs by writing the retry to a new remote experiment root.

**Tech Stack:** Bash, Python 3.13, OmegaConf/Hydra, NeMo-RL `ray.sub`, SLURM, Pre-Tyche GB200-NVL36.

## Global Constraints

- NeMo-RL source remains `d4cfecf90db41cdf142629963b54b67ab479ab02` with no source or submodule changes.
- Container remains `nemo_rl_nightly_20260630_0215.sqsh`, SHA-256 `bf841732e6615aca7a00a6c4ba47d7298a118137fc914296a4083172132ff510`.
- Cluster remains Pre-Tyche partition `36x2-a01r`, account `coreai_dlalgo_llm`, four GPUs per node, exclusive allocation, and no `--gres`.
- Both sides use global batch size 2048, 20 steps, Async-1off, and in-flight weight updates.
- Qwen3-30B-A3B uses application and Slurm segment 2; Llama 3.1 8B and Qwen3-32B omit both segment settings.
- Control and treatment differ only in `loss_fn.force_on_policy_ratio` after logging fields are normalized.
- The retry uses remote root `/lustre/fsw/coreai_dlalgo_llm/users/sna/pretyche_force_on_policy_ratio_async_retry_20260707`.

---

### Task 1: Encode the corrected topology contract

**Files:**
- Modify: `experiments/pretyche_force_on_policy_ratio_llama_qwen_async_20260707/manifests/config_contract.tsv`
- Modify: `experiments/pretyche_force_on_policy_ratio_llama_qwen_async_20260707/scripts/test_matrix_contract.sh`
- Modify: `experiments/pretyche_force_on_policy_ratio_llama_qwen_async_20260707/scripts/validate_config_contract.py`

**Interfaces:**
- Consumes: official r0.7.0 topology values from the approved design.
- Produces: six `Case` records where `segment` is `int | None`, and resolved-config checks against derived training nodes.

- [ ] **Step 1: Write the failing contract test**

Change the shell contract test to require six Async rows, three control/force pairs, segments `none/2/none`, and post-split topology assertions in the Python validator. It must reject the current eight-row `segment == nodes` manifest.

- [ ] **Step 2: Run the contract test to verify RED**

Run:

```bash
bash experiments/pretyche_force_on_policy_ratio_llama_qwen_async_20260707/scripts/test_matrix_contract.sh
```

Expected: FAIL because the manifest still contains eight rows and total-node segments 2, 4, and 8.

- [ ] **Step 3: Implement optional segments and derived-node validation**

Make `Case.segment` optional, parse `none` as `None`, and append the Hydra override only when a segment is present:

```python
segment: int | None

segment_value = row["segment"]
segment = None if segment_value == "none" else int(segment_value)

overrides = [
    f"grpo.max_num_steps={self.steps}",
    "checkpointing.enabled=false",
    f"policy.train_global_batch_size={self.global_batch_size}",
    f"loss_fn.force_on_policy_ratio={force_value}",
]
if self.segment is not None:
    overrides.append(f"cluster.segment_size={self.segment}")
```

Resolve `policy.generation.colocated.resources.num_nodes`, compute `train_nodes = total_nodes - inference_nodes`, and assert that every non-null segment divides both total and training nodes. Assert the exact model topology: Llama `None`, Qwen3-30B-A3B `2`, Qwen3-32B `None`.

- [ ] **Step 4: Run the contract test to verify GREEN**

Run the shell test again. Expected: `MATRIX_CONTRACT_OK`.

### Task 2: Make submission and execution segment-aware

**Files:**
- Modify: `experiments/pretyche_force_on_policy_ratio_llama_qwen_async_20260707/scripts/test_matrix_contract.sh`
- Modify: `experiments/pretyche_force_on_policy_ratio_llama_qwen_async_20260707/scripts/submit_force_on_policy_matrix.sh`
- Modify: `experiments/pretyche_force_on_policy_ratio_llama_qwen_async_20260707/scripts/run_force_on_policy_benchmark.sbatch`
- Modify: `experiments/pretyche_force_on_policy_ratio_llama_qwen_async_20260707/README.md`

**Interfaces:**
- Consumes: manifest `segment` values `none` or `2`.
- Produces: optional `sbatch --segment`, optional Hydra segment override, and a fresh retry result namespace.

- [ ] **Step 1: Extend the failing shell test**

Require the submitter to build `segment_args` conditionally, require the runner to build `topology_override` conditionally, and require the retry root string. Reject unconditional `--segment="$segment"` and `cluster.segment_size=${SEGMENT_SIZE}`.

- [ ] **Step 2: Run the test to verify RED**

Run the shell test. Expected: FAIL on the unconditional segment patterns.

- [ ] **Step 3: Implement conditional SLURM and Hydra arguments**

In the submitter, convert `none` to an empty exported value and add the SLURM option only for an integer segment:

```bash
local -a segment_args=()
local segment_export=
if [[ $segment != none ]]; then
    segment_args=(--segment="$segment")
    segment_export=$segment
fi
```

Place `"${segment_args[@]}"` in the `sbatch` command and export `SEGMENT_SIZE=${segment_export}`. In the runner, allow an empty segment and build the command suffix:

```bash
topology_override=
if [[ -n $SEGMENT_SIZE ]]; then
    topology_override="cluster.segment_size=$SEGMENT_SIZE"
fi
```

Pass `$topology_override` after the force override. Change experiment paths and W&B run names to the retry namespace. Preserve no-`--gres`, immutable-source, container-SHA, runner-SHA, and duplicate-submission guards.

- [ ] **Step 4: Run shell and resolved-config tests to verify GREEN**

Run:

```bash
bash experiments/pretyche_force_on_policy_ratio_llama_qwen_async_20260707/scripts/test_matrix_contract.sh
PYTHONPATH=. /tmp/q30-q32-run-compare-venv/bin/python \
  experiments/pretyche_force_on_policy_ratio_llama_qwen_async_20260707/scripts/validate_config_contract.py
```

Expected: six `CONFIG_OK`, three `PAIR_OK`, and `MATRIX_CONTRACT_OK`.

### Task 3: Commit, push, and stage the retry

**Files:**
- Modify only the experiment files listed in Tasks 1 and 2.
- Preserve uncommitted result-report files outside the implementation commit.

**Interfaces:**
- Consumes: locally green topology contract.
- Produces: pushed harness commit and byte-identical remote scripts.

- [ ] **Step 1: Verify the implementation diff**

Run `git diff --check`, `bash -n` on both shell scripts, `python -m py_compile` on the validator, and the full matrix contract test.

- [ ] **Step 2: Commit only implementation files**

Use `git commit -s` with message `fix: correct async retry segment topology`.

- [ ] **Step 3: Push the harness branch**

Push `sna/q30-q32-force-on-policy-benchmark-20260707` to its configured upstream before submission.

- [ ] **Step 4: Stage files on Pre-Tyche**

Create the fresh retry root and transfer the manifest and scripts. Verify remote SHA-256 values against local values before executing them.

### Task 4: Validate, submit, and monitor six jobs

**Files:**
- Remote manifest and scripts under `/lustre/fsw/coreai_dlalgo_llm/users/sna/pretyche_force_on_policy_ratio_async_retry_20260707`.
- Remote results manifest `results/jobs.tsv`.

**Interfaces:**
- Consumes: staged scripts, pinned source/container, and credentials already present on Pre-Tyche.
- Produces: six SLURM job IDs plus initial setup/step evidence.

- [ ] **Step 1: Run remote resolved-config validation**

Run the validator from the pinned container/source environment. Expected: six `CONFIG_OK` and three `PAIR_OK` records.

- [ ] **Step 2: Run all six `sbatch --test-only` checks**

Execute `TEST_ONLY=1 scripts/submit_force_on_policy_matrix.sh`. Expected: all six cases accepted, with only the two Qwen3-30B-A3B commands carrying `--segment=2`.

- [ ] **Step 3: Submit all six jobs exactly once**

Execute `TEST_ONLY=0 scripts/submit_force_on_policy_matrix.sh`. Confirm six unique job IDs in `results/jobs.tsv`.

- [ ] **Step 4: Monitor for at least five minutes**

Poll `squeue`/`sacct`, inspect only concise log tails, and verify that no job reproduces `num_nodes (...) must be divisible by segment_size (...)`. For jobs that start, require compute-cluster setup to pass and record the highest training step reached.

- [ ] **Step 5: Report actual terminal or running state**

Report job IDs, topology values, W&B URLs when available, and any first fatal signature. Do not claim performance results until both members of a pair complete 20 steps.
