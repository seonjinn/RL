# Qwen Safe CUDA Graph Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a persistent, fail-closed Qwen3-30B-A3B and Qwen3-235B-A22B launch/reporting path, then submit the safe Router Replay and `moe_router` CUDA Graph arms on OCI-HSG.

**Architecture:** Extend the existing selector and leaf-launch system. Make Router Replay part of run identity, reject Router Replay combined with router-containing graphs, and use committed A/B/C/E condition leaves. Carry exact source, runtime, job, R3, graph, performance, and correctness provenance into the HTML report.

**Tech Stack:** Bash, Python 3.12, pytest, Hydra, NeMo-RL, Megatron-Core, Transformer Engine, SLURM, Enroot, W&B, TensorBoard.

## Global Constraints

- Merge latest NeMo-RL `main` before submission.
- Run OCI-HSG `batch` with four GPUs per node.
- Use one digest-pinned nightly container and an attestation for the submitted commit.
- Enable packing, use exactly three optimizer warmups, and disable checkpoints.
- Use W&B project `sna-cg-study`.
- Run five-step smokes before 20-step performance runs.
- Reject `R3 on + moe_router/moe_preprocess graph on` before scheduler contact.
- Preserve unrelated files and commit only listed paths.

---

### Task 1: Synchronize with latest main and establish the test baseline

**Files:**
- Merge: `origin/main`
- Verify: `examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g.yaml`

**Interfaces:**
- Consumes: branch `experiment/thd-cg-hybrid-nemotron-20260731` at `d94ccc8d8`.
- Produces: latest outer main with the reviewed nested gitlinks preserved.

- [ ] **Step 1: Record revisions and tracked state**

```bash
git status --short
git rev-parse HEAD
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge rev-parse HEAD
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM rev-parse HEAD
```

Expected: no tracked modifications; only session and plan files are untracked.

- [ ] **Step 2: Merge latest main**

```bash
git fetch origin main
git merge --no-edit origin/main
```

Expected: merge succeeds. Stop if either nested gitlink changes unexpectedly.

- [ ] **Step 3: Run the pre-feature test baseline**

```bash
python3 -m py_compile experiments/cuda_graph/nemotron_thd_te_graph_20260731/scope_matrix.py
python3 -m pytest \
  tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py \
  tests/unit/experiments/test_matrix_submitters.py \
  tests/unit/experiments/test_nemotron_thd_te_graph_reporting.py \
  tests/unit/experiments/test_export_tensorboard.py -q
```

Expected: pass, with platform-only dependency skips recorded rather than hidden.

### Task 2: Add Qwen235 selection and Router Replay safety

**Files:**
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/models/qwen3_235b.env`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scope_matrix.py`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/run_scope.sh`
- Modify: `tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py`

**Interfaces:**
- Consumes: `MODEL=qwen3_235b`, `ROUTER_REPLAY=off|on`, and an existing scope leaf.
- Produces: 16n4g HybridEP Qwen235 selection, explicit R3 command/metadata identity, and fail-closed unsafe combinations.

- [ ] **Step 1: Write failing selector tests**

Add:

```python
spec = module.load_model_spec("qwen3_235b")
assert spec.nemorl_recipe.endswith("grpo-qwen3-235b-16n4g.yaml")
assert (spec.num_nodes, spec.gpus_per_node) == (16, 4)
assert spec.dispatcher == "hybridep"
assert spec.moe_preprocess_graph_ready is False
assert module.classify_scope(
    module.find_scope_row("moe_router"), model="qwen3_235b"
).status == "runnable"
assert module.classify_scope(
    module.find_scope_row("moe_router,moe_preprocess"), model="qwen3_235b"
).status == "capacity-blocked"
```

Extend the exact selector map with `qwen3_235b.env`.

- [ ] **Step 2: Write failing R3 safety tests**

Add:

```python
arguments = shlex.split(
    module.render_scope_command(
        model="qwen3_30ba3b",
        scope=(),
        steps=5,
        run_name="qwen30-baseline-r3on",
        cuda_graph_enabled=False,
        router_replay_enabled=True,
    )
)
assert "++policy.router_replay.enabled=true" in arguments
assert "NRL_ROUTER_REPLAY_VALIDATE=1" in arguments
assert "NRL_R3_TRACE_VERIFY_FORWARD=1" in arguments
with pytest.raises(ValueError, match="Router Replay.*router CUDA Graph"):
    module.render_scope_command(
        model="qwen3_30ba3b",
        scope=("moe_router",),
        steps=5,
        run_name="unsafe",
        router_replay_enabled=True,
    )
```

Add shell assertions that invalid R3 values and R3-on router graphs exit 2 before `SBATCH:`.

- [ ] **Step 3: Confirm red tests**

```bash
python3 -m pytest tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py \
  -k 'qwen3_235b or router_replay or model_selectors' -q
```

Expected: fail because selector and R3 contract do not exist.

- [ ] **Step 4: Implement the selector**

Create exactly:

```text
NEMORL_LAUNCHER=examples/run_grpo.py
NEMORL_LAUNCHER_VALIDATED=true
NEMORL_RECIPE=examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g.yaml
MCORE_RECIPE=__REQUIRED_QWEN235_MCORE_RECIPE__
DISPATCHER=hybridep
SUPPORTED_MODULES=attn,moe,moe_router,moe_preprocess
WHOLE_MOE_CAPACITY_READY=false
MOE_PREPROCESS_GRAPH_READY=false
REQUIRES_ULTRA_EXTERNALS=false
NUM_NODES=16
GPUS_PER_NODE=4
THD_MAX_PACKED_SEQUENCES=16
```

Add the model to Python and Bash allowlists. Do not add it to accuracy defaults.

- [ ] **Step 5: Implement the R3 contract**

Add `router_replay_enabled: bool = False` to command rendering. Always emit one `policy.router_replay.enabled` override. R3-on also emits:

```text
NRL_ROUTER_REPLAY_VALIDATE=1
NRL_R3_TRACE=1
NRL_R3_TRACE_STEPS=5
NRL_R3_TRACE_VERIFY_FORWARD=1
++policy.generation.vllm_cfg.enable_prefix_caching=false
++policy.generation.vllm_kwargs.enable_chunked_prefill=false
```

Reject R3-on with a graph scope containing `moe_router` or `moe_preprocess`. Validate `ROUTER_REPLAY`, add `r3on|r3off` to run names, and persist it in `run-metadata.env`.

- [ ] **Step 6: Verify and commit**

```bash
python3 -m pytest tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py \
  -k 'qwen3_235b or router_replay or model_selectors or rendered_nemorl_command' -q
git add \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/models/qwen3_235b.env \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/scope_matrix.py \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/run_scope.sh \
  tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py
git commit -s -m "feat: add safe Qwen router graph selection"
```

Expected: targeted tests pass and only listed files enter the commit.

### Task 3: Add persistent A/B/C/E conditions

**Files:**
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/conditions/qwen_A_baseline_r3off.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/conditions/qwen_B_moe_router_r3off.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/conditions/qwen_C_baseline_r3on.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/conditions/qwen_E_attn_r3on.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_qwen_router_validation.sh`
- Modify: `tests/unit/experiments/test_matrix_submitters.py`

**Interfaces:**
- Consumes: Qwen model, `PHASE=smoke|performance`, `REPEATS=1|3`, cluster, and optional arm list.
- Produces: independently rerunnable safe jobs with paired baselines.

- [ ] **Step 1: Write failing submitter tests**

Assert default smoke order is A/B/C/E with five steps and one repeat. Assert explicit performance arms A/B use 20 steps. Assert three repeats use distinct repeat indices. Reject arm D, path traversal, invalid phase, and non-Qwen models before any leaf invocation.

- [ ] **Step 2: Confirm red tests**

```bash
python3 -m pytest tests/unit/experiments/test_matrix_submitters.py \
  -k 'qwen_router_validation' -q
```

Expected: fail because the submitter does not exist.

- [ ] **Step 3: Implement leaves and submitter**

Each leaf validates `qwen3_30ba3b|qwen3_235b`, exports its fixed R3 value, and invokes an existing scope leaf. The submitter resolves committed paths with traversal protection, maps smoke to 5 and performance to 20 steps, defaults to one repeat, and uses separate R3-on/off run groups so A pairs with B and C pairs with E.

- [ ] **Step 4: Verify and commit**

```bash
python3 -m pytest tests/unit/experiments/test_matrix_submitters.py \
  -k 'qwen_router_validation' -q
CLUSTER=oci-hsg MODEL=qwen3_30ba3b PHASE=smoke TEST_ONLY=1 \
  bash experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_qwen_router_validation.sh
CLUSTER=oci-hsg MODEL=qwen3_235b PHASE=smoke TEST_ONLY=1 \
  bash experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_qwen_router_validation.sh
git add \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/conditions \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_qwen_router_validation.sh \
  tests/unit/experiments/test_matrix_submitters.py
git commit -s -m "feat: add Qwen router validation matrix"
```

Expected: four jobs render for each model and no scheduler call occurs.

### Task 4: Preserve R3 identity through export and HTML

**Files:**
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/export_tensorboard.py`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/collect_results.py`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/render_report.py`
- Modify: `tests/unit/experiments/test_export_tensorboard.py`
- Modify: `tests/unit/experiments/test_nemotron_thd_te_graph_reporting.py`

**Interfaces:**
- Consumes: `router_replay=off|on` from run metadata.
- Produces: JSONL, CSV, aggregation, matching, and HTML that never cross-pair R3 states.

- [ ] **Step 1: Write failing export and pairing tests**

Pass `router_replay="on"` to export and assert all rows contain it. Build otherwise-identical A/B/C/E fixtures and assert B pairs only with A while E pairs only with C.

- [ ] **Step 2: Confirm red tests**

```bash
python3 -m pytest \
  tests/unit/experiments/test_export_tensorboard.py \
  tests/unit/experiments/test_nemotron_thd_te_graph_reporting.py \
  -k 'router_replay' -q
```

Expected: fail because R3 is not an identity or match field.

- [ ] **Step 3: Implement the canonical field**

Add R3 to exporter function/CLI choices, default legacy exports to `off`, and add it to collector and renderer identity, run key, match key, comparison group, and visible columns. Reject values other than `off|on` during export.

- [ ] **Step 4: Verify and commit**

```bash
python3 -m pytest \
  tests/unit/experiments/test_export_tensorboard.py \
  tests/unit/experiments/test_nemotron_thd_te_graph_reporting.py -q
git add \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/export_tensorboard.py \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/collect_results.py \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/render_report.py \
  tests/unit/experiments/test_export_tensorboard.py \
  tests/unit/experiments/test_nemotron_thd_te_graph_reporting.py
git commit -s -m "feat: report router replay experiment identity"
```

Expected: reporting tests pass and no R3 cross-pair is possible.

### Task 5: Verify, review, document, and push

**Files:**
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/README.md`
- Modify: `session/20260802_122126/session_state.md`
- Modify: `session/20260802_122126/timeline.md`
- Modify: `session/20260802_122126/files.md`
- Modify: `session/20260802_122126/handoff.md`

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces: reviewed and pushed source ready for OCI attestation.

- [ ] **Step 1: Document commands and safety boundary**

Document A/B/C/E, Qwen235 16n4g, R3+router rejection, five-step and 20-step commands, R3 trace checking, and result collection.

- [ ] **Step 2: Run the full local gate**

```bash
bash -n \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/run_scope.sh \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_qwen_router_validation.sh \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/conditions/*.sh
python3 -m py_compile \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/scope_matrix.py \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/export_tensorboard.py \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/collect_results.py \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/render_report.py
python3 -m pytest \
  tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py \
  tests/unit/experiments/test_matrix_submitters.py \
  tests/unit/experiments/test_export_tensorboard.py \
  tests/unit/experiments/test_nemotron_thd_te_graph_reporting.py -q
git diff --check
```

Expected: syntax and targeted tests pass, with any platform skip disclosed.

- [ ] **Step 3: Run a dedicated code review**

Review selector constraints, unsafe-arm rejection, shell escaping, R3 matching, and backward compatibility. Apply only verified fixes and rerun Step 2.

- [ ] **Step 4: Commit docs/session and push**

```bash
git add \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/README.md \
  session/20260802_122126
git commit -s -m "docs: record Qwen CUDA graph campaign launch"
git push seonjinn experiment/thd-cg-hybrid-nemotron-20260731
```

Expected: the branch push succeeds and no tracked edits remain.

### Task 6: Create an attested OCI runtime and submit Qwen30 smokes

**Files:**
- Generate remotely: immutable source snapshot and runtime attestation.
- Generate remotely: fresh OCI profile, job logs, and run metadata.

**Interfaces:**
- Consumes: pushed revisions, nightly digest `f863be73380afea5c545614612bcec9a38c9f59be54e88d9431fda4acba717aa`, and cached Qwen models.
- Produces: Qwen30 A/B/C/E smoke job IDs from one provenance.

- [ ] **Step 1: Check scheduling**

Compare OCI FairShare and `sbatch --test-only` for `nemotron_sw_post` and `nemotron_n3_post`; select the better runnable account.

- [ ] **Step 2: Pull a clean remote checkout**

Use one SSH command to clone/fetch the pushed branch, check out the exact local commit, initialize submodules recursively, and verify all three revisions. Do not edit remote source.

- [ ] **Step 3: Create snapshot and preflight**

Run `scripts/create_source_snapshot.sh`, submit `scripts/validate_oci_container_runtime.sub`, capture the job ID, and require a successful attestation JSON.

- [ ] **Step 4: Run both dry-run gates**

Run A/B/C/E with `TEST_ONLY=1`, then `SBATCH_TEST_ONLY=1` using the fresh profile. Verify 4n4g, `batch`, no singleton dependency, unique log paths, warmup 3, and checkpoints off.

- [ ] **Step 5: Submit and monitor**

```bash
CLUSTER=oci-hsg MODEL=qwen3_30ba3b PHASE=smoke REPEATS=1 \
  bash experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_qwen_router_validation.sh
```

Record job IDs and monitor scheduler state, first errors, Ray connections, initialization, and GPU activity for at least five minutes. Cancel jobs that cannot progress and waste GPUs.

### Task 7: Gate Qwen235, promote A/B, and render results

**Files:**
- Generate remotely: Qwen235 route-completeness and smoke logs.
- Generate: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/raw/*.jsonl`
- Generate: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/results.json`
- Generate: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/results.csv`
- Generate: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/report.html`

**Interfaces:**
- Consumes: passing Qwen30 smokes and the same attested runtime.
- Produces: Qwen235 smoke result, one 20-step A/B pair per model, and HTML.

- [ ] **Step 1: Gate Qwen30**

Require five optimizer steps, finite metrics, expected graph coverage, zero undeclared fallbacks, and passing R3 traces for C/E.

- [ ] **Step 2: Check Qwen235 routes**

Run `tools/model_diagnostics/6.vllm_routed_experts_completeness.py` with the 16n4g vLLM configuration. Require complete route tensors before C/E.

- [ ] **Step 3: Submit Qwen235 smoke**

Run scheduler test-only and submit A/B on 16n4g. Submit C/E only after route completeness passes. Monitor each job for five minutes.

- [ ] **Step 4: Promote passing A/B arms**

```bash
CLUSTER=oci-hsg MODEL=qwen3_30ba3b PHASE=performance REPEATS=1 \
  bash experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_qwen_router_validation.sh A B
CLUSTER=oci-hsg MODEL=qwen3_235b PHASE=performance REPEATS=1 \
  bash experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_qwen_router_validation.sh A B
```

- [ ] **Step 5: Export and render**

Create provenance JSON, export each run with explicit `--router-replay`, validate R3 traces, then run:

```bash
python3 experiments/cuda_graph/nemotron_thd_te_graph_20260731/collect_results.py
python3 experiments/cuda_graph/nemotron_thd_te_graph_20260731/render_report.py
```

If Qwen235 TensorBoard fails, stop collection and add a separately tested W&B adapter; do not invent graph or parity fields.

- [ ] **Step 6: Record the stop condition**

Record all job IDs, included steps, performance, correctness, graph coverage, failures, and HTML path. Stop after one valid Qwen30 A/B 20-step pair and one valid Qwen235 A/B 20-step pair, plus safe R3 controls that passed smoke.
