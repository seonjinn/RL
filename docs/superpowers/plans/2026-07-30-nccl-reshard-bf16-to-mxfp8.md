# NCCL-Reshard BF16-to-MXFP8 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transfer BF16 Megatron weights as MXFP8 value/scale tensor pairs through NCCL-Reshard and measure the refit-time reduction.

**Architecture:** Reuse the existing trainer MXFP8 quantizer and vLLM MXFP8 post-load path. Expand each eligible FFN weight into two typed NCCL-Reshard transfers without changing `xferdtensor`, and reject layouts whose K-axis shards are not aligned to the MXFP8 block size.

**Tech Stack:** Python 3.13, PyTorch DTensor placements, Ray, NCCL/nccl4py, Megatron Bridge, vLLM, pytest, SLURM, W&B.

## Global Constraints

- Work only in `/Users/sna/MXFP8_generation/nemo-rl-pr3294-nccl-mxfp8-prequant`.
- Preserve existing BF16 and matching blockwise-FP8 NCCL-Reshard behavior.
- Restrict transform-aware support to Megatron BF16 storage and vLLM MXFP8 rollout.
- Keep `xferdtensor` one-source/one-destination and invoke it separately for values and scales.
- Quantization and scale semantics must match `mxfp8_e4m3_quantize_for_refit`.
- Every K-axis source and destination shard must be aligned to 32 values.
- Do not cache a full quantized model copy.
- Write a failing test before each production behavior change.
- Use `uv run` for Python and pytest commands.
- Sign every commit with `git commit -s`.

---

### Task 1: Initialization Handshake And Transform Metadata

**Files:**
- Modify: `nemo_rl/weight_sync/nccl_reshard_weight_synchronizer.py`
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- Modify: `nemo_rl/weight_sync/nccl_reshard_utils.py`
- Test: `tests/unit/weight_sync/test_nccl_reshard_weight_synchronizer.py`
- Test: `tests/unit/weight_sync/test_nccl_reshard_utils.py`

**Interfaces:**
- Consumes: `policy.prepare_refit_info()`, `generation.prepare_refit_info()`, and `policy.enable_refit_prequantize()`.
- Produces: Per-parameter `refit_transform`, `scale_global_shape`, `scale_dtype`, `scale_src_placements`, and `scale_dst_placements`.

- [ ] **Step 1: Add a failing synchronizer handshake test**

Create a test that records this exact order:

```python
[
    "policy.prepare_refit_info",
    "generation.prepare_refit_info:bf16",
    "policy.enable_refit_prequantize",
    "generation.prepare_refit_info:mxfp8",
    "policy.prepare_nccl_reshard_refit_info",
    "generation.prepare_nccl_reshard_refit_info",
]
```

- [ ] **Step 2: Run the handshake test and verify RED**

Run:

```bash
uv run --group test pytest -q \
  tests/unit/weight_sync/test_nccl_reshard_weight_synchronizer.py
```

Expected: failure because `NcclReshardWeightSynchronizer.init_communicator`
does not perform the MXFP8 negotiation.

- [ ] **Step 3: Implement the two-stage initialization handshake**

Mirror the validated checkpoint-engine handshake. Require non-`None` updated
metadata when generation requests prequantization, then build NCCL-Reshard
metadata only after every trainer worker has enabled the selected names.

- [ ] **Step 4: Add failing metadata-pair and alignment tests**

Cover:

```python
value_shape = (64, 128)
scale_shape = (64, 4)
value_dtype = torch.float8_e4m3fn
scale_dtype = torch.uint8
```

Also require setup-time failure when a source or destination K shard has a
local width that is not divisible by 32.

- [ ] **Step 5: Run metadata tests and verify RED**

Run:

```bash
uv run --group test pytest -q \
  tests/unit/weight_sync/test_nccl_reshard_utils.py
```

- [ ] **Step 6: Pair bulk value/scale metadata and validate placements**

When the metadata iterator emits
`<name>_scale_from_checkpoint`, attach it to the preceding eligible bulk
parent instead of routing it through misc packed broadcast. Keep scales for
non-bulk parameters on the misc path. Derive scale placements from the parent
name and reject non-32-aligned K sharding.

- [ ] **Step 7: Verify Task 1**

Run:

```bash
uv run --group test pytest -q \
  tests/unit/weight_sync/test_nccl_reshard_utils.py \
  tests/unit/weight_sync/test_nccl_reshard_weight_synchronizer.py
```

- [ ] **Step 8: Commit Task 1**

```bash
git add \
  nemo_rl/weight_sync/nccl_reshard_weight_synchronizer.py \
  nemo_rl/models/policy/workers/megatron_policy_worker.py \
  nemo_rl/weight_sync/nccl_reshard_utils.py \
  tests/unit/weight_sync/test_nccl_reshard_weight_synchronizer.py \
  tests/unit/weight_sync/test_nccl_reshard_utils.py
git commit -s -m "feat(refit): negotiate MXFP8 NCCL reshard metadata"
```

### Task 2: Trainer-Side Quantization And Dual Transfer

**Files:**
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- Test: `tests/unit/models/megatron/test_group_experts.py`
- Test: `tests/unit/models/generation/test_mxfp8_prequant.py`

**Interfaces:**
- Consumes: Transform metadata from Task 1.
- Produces: `RefitCtx.buf` containing E4M3 values and
  `RefitCtx.extra["scale_buf"]` containing row-major E8M0 scales.

- [ ] **Step 1: Run the existing trainer contract tests and verify RED**

Run:

```bash
uv run --group test pytest -q \
  tests/unit/models/megatron/test_group_experts.py \
  tests/unit/models/generation/test_mxfp8_prequant.py
```

Expected: `test_build_mxfp8_source_specs_quantize_direct_and_grouped_once`
fails because the source mapping returns BF16 tensors.

- [ ] **Step 2: Implement transformed source specs**

For direct and grouped parameters:

```python
value, scale = mxfp8_e4m3_quantize_for_refit(source)
return RefitCtx(buf=value, extra={"scale_buf": scale})
```

Grouped experts must stack once and quantize the stacked tensor once.

- [ ] **Step 3: Add a failing dual-transfer test**

Patch `xferdtensor` with a recording implementation and require, for each
transformed parent, value transfer followed by scale transfer with the
metadata-defined global shapes and placements.

- [ ] **Step 4: Run the dual-transfer test and verify RED**

Run:

```bash
uv run --group test pytest -q \
  tests/unit/models/megatron/test_group_experts.py
```

- [ ] **Step 5: Implement the scale transfer**

Construct a second `DTensorRef` from `ctx.extra["scale_buf"]` and invoke
`xferdtensor` with `scale_global_shape`, `scale_src_placements`, and
`scale_dst_placements` on the same CUDA stream.

- [ ] **Step 6: Verify Task 2**

Run:

```bash
uv run --group test pytest -q \
  tests/unit/models/megatron/test_group_experts.py \
  tests/unit/models/generation/test_mxfp8_prequant.py
```

- [ ] **Step 7: Commit Task 2**

```bash
git add \
  nemo_rl/models/policy/workers/megatron_policy_worker.py \
  tests/unit/models/megatron/test_group_experts.py \
  tests/unit/models/generation/test_mxfp8_prequant.py
git commit -s -m "feat(refit): prequantize MXFP8 reshard sources"
```

### Task 3: Generation Value/Scale Destinations And Atomic Load

**Files:**
- Modify: `nemo_rl/models/generation/vllm/vllm_backend.py`
- Test: `tests/unit/models/generation/test_nccl_reshard_backend.py`

**Interfaces:**
- Consumes: E4M3 value and E8M0 scale transfers from Task 2.
- Produces: Canonical vLLM value and `*_scale_from_checkpoint` tensors ready
  for the existing post-load processing.

- [ ] **Step 1: Run the existing destination tests and verify RED**

Run:

```bash
uv run --group test pytest -q \
  tests/unit/models/generation/test_nccl_reshard_backend.py
```

Expected: MXFP8 value/scale destination tests fail because only value targets
are currently mapped.

- [ ] **Step 2: Implement direct and merged value/scale specs**

Resolve the scale target from the actual vLLM value parameter name:

```text
<value_parameter_name>_scale_from_checkpoint
```

Use the same merged row slice for value and scale. Validate destination
presence, dtype, local shape, and slice shape during mapping construction.

- [ ] **Step 3: Add a failing receiver dual-transfer test**

Require the receive loop to call `xferdtensor` for value and scale before the
spec's post hook copies either merged pair into its live destinations.

- [ ] **Step 4: Run the receiver transfer test and verify RED**

Run:

```bash
uv run --group test pytest -q \
  tests/unit/models/generation/test_nccl_reshard_backend.py
```

- [ ] **Step 5: Implement paired receive and post-load ordering**

Receive value and scale on the same stage stream, run the merged post hook
after both calls, synchronize all stage streams, receive misc parameters, and
invoke the existing `process_weights_after_loading` once.

- [ ] **Step 6: Verify Task 3**

Run:

```bash
uv run --group test pytest -q \
  tests/unit/models/generation/test_nccl_reshard_backend.py
```

- [ ] **Step 7: Commit Task 3**

```bash
git add \
  nemo_rl/models/generation/vllm/vllm_backend.py \
  tests/unit/models/generation/test_nccl_reshard_backend.py
git commit -s -m "feat(refit): receive MXFP8 reshard tensor pairs"
```

### Task 4: Functional Correctness And Regression Verification

**Files:**
- Modify: `tests/functional/grpo_nccl_reshard_refit.sh`
- Modify: `tests/functional/L1_Functional_Tests_GB200_MXFP8.sh`
- Test: `tests/unit/weight_sync/test_nccl_reshard_utils.py`
- Test: `tests/unit/models/megatron/test_group_experts.py`
- Test: `tests/unit/models/generation/test_nccl_reshard_backend.py`

**Interfaces:**
- Consumes: Complete transform-aware refit path.
- Produces: A Blackwell functional smoke that executes at least two refits.

- [ ] **Step 1: Add an MXFP8 NCCL-Reshard functional invocation**

Use:

```text
policy.generation.refit_transport=nccl_reshard
policy.generation.vllm_cfg.precision=fp8
policy.generation.vllm_cfg.is_mx=true
policy.generation.vllm_cfg.refit_prequantize=true
policy.megatron_cfg.fp8_cfg.fp8_param=false
```

- [ ] **Step 2: Run formatting and static checks**

Run:

```bash
uv run --group test ruff check \
  nemo_rl/weight_sync/nccl_reshard_utils.py \
  nemo_rl/weight_sync/nccl_reshard_weight_synchronizer.py \
  nemo_rl/models/policy/workers/megatron_policy_worker.py \
  nemo_rl/models/generation/vllm/vllm_backend.py \
  tests/unit/weight_sync/test_nccl_reshard_utils.py \
  tests/unit/models/megatron/test_group_experts.py \
  tests/unit/models/generation/test_nccl_reshard_backend.py
uv run --group test ruff format --check \
  nemo_rl/weight_sync/nccl_reshard_utils.py \
  nemo_rl/weight_sync/nccl_reshard_weight_synchronizer.py \
  nemo_rl/models/policy/workers/megatron_policy_worker.py \
  nemo_rl/models/generation/vllm/vllm_backend.py
git diff --check
```

- [ ] **Step 3: Run the focused unit suite in the Linux NeMo-RL container**

Run:

```bash
uv run --group test pytest -q \
  tests/unit/weight_sync/test_nccl_reshard_utils.py \
  tests/unit/weight_sync/test_nccl_reshard_weight_synchronizer.py \
  tests/unit/models/megatron/test_group_experts.py \
  tests/unit/models/generation/test_mxfp8_prequant.py \
  tests/unit/models/generation/test_nccl_reshard_backend.py
```

- [ ] **Step 4: Run the two-refit Blackwell functional smoke**

Run the updated functional driver in the cluster's nightly NeMo-RL container
and require both refits, post-load processing, generation, logprob, and policy
training to complete.

- [ ] **Step 5: Commit Task 4**

```bash
git add \
  tests/functional/grpo_nccl_reshard_refit.sh \
  tests/functional/L1_Functional_Tests_GB200_MXFP8.sh
git commit -s -m "test(refit): cover MXFP8 NCCL reshard end to end"
```

### Task 5: Controlled Refit Performance A/B

**Files:**
- Create: `experiments/nccl_reshard_mxfp8_prequant/README.md`
- Create: `experiments/nccl_reshard_mxfp8_prequant/PLAN.md`
- Create: `experiments/nccl_reshard_mxfp8_prequant/launch_ab.sh`
- Create: `experiments/nccl_reshard_mxfp8_prequant/report.md`
- Create: `experiments/nccl_reshard_mxfp8_prequant/results/metrics.csv`

**Interfaces:**
- Consumes: Verified branch from Task 4.
- Produces: Reproducible 5-step smoke and 20-step steady-state A/B results.

- [ ] **Step 1: Record immutable experiment provenance**

Capture:

```text
git commit
container path and SHA256
cluster and GPU type
model path
recipe path and full overrides
node and GPU allocation
seed
W&B project and run URLs
SLURM job IDs
```

- [ ] **Step 2: Check scheduling before submission**

Run the cluster's FairShare/test-only workflow, use `batch`, and request the
smallest node allocation that preserves the selected performance recipe's
parallelism.

- [ ] **Step 3: Submit paired 5-step smoke runs**

Submit:

```text
baseline: existing NCCL-Reshard BF16 storage path
optimized: BF16 trainer + MXFP8 rollout + NCCL-Reshard prequantization
```

Monitor both jobs for at least five minutes and inspect the first refit logs.

- [ ] **Step 4: Validate correctness gates**

Require:

```text
no transfer or shape mismatch
no missing scale destination
two or more completed refits
finite reward, loss, and logprob metrics
no token-probability regression beyond the existing MXFP8 parity tolerance
```

- [ ] **Step 5: Submit paired 20-step runs**

Keep every setting identical except the precision/refit path under test. Do not
save checkpoints. Use unique W&B names under one project.

- [ ] **Step 6: Extract steady-state metrics**

Exclude initialization and steps 1-2. Write one row per run with:

```text
transfer_update_s
refit_total_s
generation_s
logprob_s
policy_training_s
e2e_step_s
tokens_per_second_per_gpu
mean_rollout_reward
```

- [ ] **Step 7: Write the report**

Report absolute values, percentage changes, run URLs, job IDs, and the exact
comparison boundary. Separate transport-only conclusions from end-to-end
conclusions.

- [ ] **Step 8: Commit Task 5**

```bash
git add experiments/nccl_reshard_mxfp8_prequant
git commit -s -m "docs(experiments): report MXFP8 NCCL reshard refit"
```

