# Cross-Precision NCCL-Reshard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace MXFP8-specific NCCL-Reshard transfer branches with a compact component-based transform contract while preserving the validated Megatron BF16-storage to vLLM MXFP8-rollout behavior and performance.

**Architecture:** A small transform module defines structured requests, ordered components, and a registry keyed by source and target storage format. Megatron encodes source tensors, NCCL-Reshard moves every component through one precision-independent loop, and vLLM maps and finalizes the received component set. The first runtime codec remains BF16-to-MXFP8; a mock four-component codec proves future NVFP4 extensibility.

**Tech Stack:** Python 3.13, PyTorch DTensor placements, Ray, vLLM, pytest, Ruff, Pyright, SLURM on GCP-NRT B200.

## Global Constraints

- Preserve commit `a854dbdbe33eccee80c32c8e2025fb0ac59d26d5` as branch `sna/nccl-reshard-bf16-mxfp8-stable-v1` and annotated tag `nccl-reshard-bf16-mxfp8-stable-v1`.
- Develop on `sna/nccl-reshard-cross-precision` in a separate worktree.
- Runtime support in this plan is BF16 parameter storage to MXFP8 rollout only.
- Training compute precision must not substitute for parameter storage format.
- Add one focused production module and replace existing hard-coded branches instead of retaining parallel implementations.
- Unknown format pairs and incomplete component sets fail before transfer or successful acknowledgement.
- A partial vLLM in-place update makes the affected generation actor unusable; transactional rollback is not part of this PR.
- Do not claim native `nccl.m2n` performance from the Python exact-transfer fallback.

## File Map

- Create `nemo_rl/weight_sync/refit_transforms.py`: request/component types, BF16-to-MXFP8 registry entry, and deterministic signatures.
- Modify `nemo_rl/weight_sync/nccl_reshard_utils.py`: generic component metadata and placement validation.
- Modify `nemo_rl/weight_sync/interfaces.py` and policy/generation interfaces: structured negotiation.
- Modify Megatron policy worker: source codec installation and generic component sender.
- Modify vLLM backend/worker/generation: target request, destination component mapping, and generic receiver.
- Modify NCCL synchronizer: shared handshake and plan-signature verification.
- Extend focused unit tests under `tests/unit/weight_sync`, `tests/unit/models/megatron`, and `tests/unit/models/generation`.
- Update `experiments/nccl_reshard_pr3294` and session records after GCP-NRT validation.

---

### Task 1: Freeze Baseline And Create Worktree

**Files:**
- Verify: `docs/superpowers/specs/2026-08-02-cross-precision-nccl-reshard-design.md`
- Verify: `experiments/nccl_reshard_pr3294/RESULTS.md`

**Interfaces:**
- Consumes: validated commit `a854dbdbe33eccee80c32c8e2025fb0ac59d26d5`.
- Produces: stable branch/tag and isolated development worktree.

- [ ] **Step 1: Verify the source checkout**

```bash
git status --short
git show -s --oneline a854dbdbe33eccee80c32c8e2025fb0ac59d26d5
```

Expected: clean status and the validated result commit.

- [ ] **Step 2: Create and push stable references**

```bash
git branch sna/nccl-reshard-bf16-mxfp8-stable-v1 a854dbdbe33eccee80c32c8e2025fb0ac59d26d5
git tag -a nccl-reshard-bf16-mxfp8-stable-v1 a854dbdbe33eccee80c32c8e2025fb0ac59d26d5 -m "Stable BF16-to-MXFP8 NCCL-Reshard baseline"
git push fork sna/nccl-reshard-bf16-mxfp8-stable-v1
git push fork nccl-reshard-bf16-mxfp8-stable-v1
```

- [ ] **Step 3: Create the development worktree**

```bash
git worktree add /Users/sna/MXFP8_generation/nemo-rl-nccl-reshard-cross-precision -b sna/nccl-reshard-cross-precision 9f234b51b
```

Expected: clean worktree on the development branch with the approved design.

- [ ] **Step 4: Run local source checks**

```bash
python3 -m compileall -q nemo_rl/weight_sync
git diff --check
```

Full pytest runs use the Linux container because the repository lockfile excludes local macOS.

---

### Task 2: Add Minimal Transform Contract

**Files:**
- Create: `nemo_rl/weight_sync/refit_transforms.py`
- Create: `tests/unit/weight_sync/test_refit_transforms.py`

**Interfaces:**
- Produces `RefitTransformRequest`, `TransformComponentSpec`, `RefitTransformPlan`, `resolve_transform`, and `plan_signature`.

- [ ] **Step 1: Write failing registry tests**

The primary test uses this API:

```python
request = RefitTransformRequest(
    parameter_names=("model.layers.0.mlp.down_proj.weight",),
    source_format="bf16",
    target_format="mxfp8_e4m3_e8m0",
)
codec = resolve_transform(request.source_format, request.target_format)
components = codec.describe_outputs((64, 128), "torch.bfloat16")
assert [(item.role, item.global_shape, item.dtype_name) for item in components] == [
    ("weight", (64, 128), "torch.float8_e4m3fn"),
    ("weight_scale", (64, 4), "torch.uint8"),
]
```

Also assert unknown pairs fail with both formats in the message and canonical plan signatures ignore dictionary insertion order.

- [ ] **Step 2: Verify RED**

```bash
pytest -q tests/unit/weight_sync/test_refit_transforms.py
```

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement the minimal types and registry**

Use frozen dataclasses and a protocol. Register one private BF16-to-MXFP8 codec:

```python
_TRANSFORM_CODECS = {
    ("bf16", "mxfp8_e4m3_e8m0"): _BF16ToMXFP8Codec(),
}
```

The codec validates BF16 input and K divisibility by 32. `plan_signature` hashes canonical JSON containing parameter names, transform IDs, component roles/shapes/dtypes, and finalization scope only.

- [ ] **Step 4: Prove multi-component extensibility**

Register a test-local codec whose roles are `weight`, `weight_scale`, `weight_scale_2`, and `input_scale`; assert the generic plan preserves that order. Do not import ModelOpt or claim NVFP4 correctness.

- [ ] **Step 5: Verify GREEN and commit**

```bash
pytest -q tests/unit/weight_sync/test_refit_transforms.py
ruff check nemo_rl/weight_sync/refit_transforms.py tests/unit/weight_sync/test_refit_transforms.py
git add nemo_rl/weight_sync/refit_transforms.py tests/unit/weight_sync/test_refit_transforms.py
git commit -s -m "refactor(refit): add cross-precision transform contract"
```

---

### Task 3: Replace MXFP8 Metadata With Ordered Components

**Files:**
- Modify: `nemo_rl/weight_sync/nccl_reshard_utils.py:231-535,892-1050`
- Modify: `tests/unit/weight_sync/test_nccl_reshard_utils.py:365-760`

**Interfaces:**
- Consumes: Task 2 registry and component descriptions.
- Produces: a `components` list for every bulk parameter and a canonical `plan_signature` on refit metadata.

- [ ] **Step 1: Write component-schema tests first**

MXFP8 metadata must become:

```python
assert param["components"] == [
    {
        "role": "weight",
        "global_shape": (64, 128),
        "dtype": "torch.float8_e4m3fn",
        "src_placements": [Shard(1)],
        "dst_placements": [Shard(1)],
    },
    {
        "role": "weight_scale",
        "global_shape": (64, 4),
        "dtype": "torch.uint8",
        "src_placements": [Shard(1)],
        "dst_placements": [Shard(1)],
    },
]
```

Add an identity BF16 test with one `weight` component and a regression test rejecting blockwise-FP8 storage that targets MXFP8 without an MXFP8 scale component.

- [ ] **Step 2: Verify RED**

```bash
pytest -q tests/unit/weight_sync/test_nccl_reshard_utils.py
```

Expected: current top-level scale metadata does not satisfy the new component assertions.

- [ ] **Step 3: Build generic metadata**

Resolve transformed parameters through the registry. Attach source/destination placements to each codec output. Keep MXFP8 K-shard alignment validation in the codec integration. Identity parameters use a one-component plan. Replace scale-specific dtype/placement restoration with a loop over `components`.

- [ ] **Step 4: Verify GREEN and commit**

```bash
pytest -q tests/unit/weight_sync/test_nccl_reshard_utils.py tests/unit/weight_sync/test_refit_transforms.py
ruff check nemo_rl/weight_sync/nccl_reshard_utils.py tests/unit/weight_sync/test_nccl_reshard_utils.py
git add nemo_rl/weight_sync/nccl_reshard_utils.py tests/unit/weight_sync/test_nccl_reshard_utils.py
git commit -s -m "refactor(refit): describe NCCL payloads as components"
```

---

### Task 4: Generalize Sender And Receiver Loops

**Files:**
- Modify: `nemo_rl/weight_sync/nccl_reshard_utils.py:65-130`
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py:2374-2760`
- Modify: `nemo_rl/models/generation/vllm/vllm_backend.py:947-1510`
- Modify: `tests/unit/models/megatron/test_group_experts.py:128-285`
- Modify: `tests/unit/models/generation/test_nccl_reshard_backend.py:358-650`

**Interfaces:**
- Produces `RefitCtx.transfer_tensors: tuple[torch.Tensor, ...] | None` and `tensors_for_transfer()`.

- [ ] **Step 1: Write failing sender tests**

Construct `RefitCtx(buf=value, transfer_tensors=(value, scale))` and component metadata. Assert two transfers in component order. Add a four-component context and assert the same sender loop makes four transfers without inspecting role names.

- [ ] **Step 2: Write failing receiver tests**

Expose destination value and scale regions through `transfer_tensors`. Assert both transfers finish before `post(ctx)`, misc transfer, and `process_weights_after_loading`. Add a component-count mismatch test that raises before success.

- [ ] **Step 3: Verify RED**

```bash
pytest -q tests/unit/models/megatron/test_group_experts.py tests/unit/models/generation/test_nccl_reshard_backend.py
```

Expected: `RefitCtx` rejects `transfer_tensors` and current loops branch on MXFP8.

- [ ] **Step 4: Implement the shared component loop**

Add:

```python
transfer_tensors: tuple[torch.Tensor, ...] | None = None

def tensors_for_transfer(self) -> tuple[torch.Tensor, ...]:
    return self.transfer_tensors or (self.buf,)
```

Megatron quantization returns `(value, scale)`. Both sender and receiver validate component count, zip tensors with component metadata, and transfer each item. Remove scale-specific transfer blocks. vLLM may retain MXFP8-specific local buffer mapping, but the transport loop must not know MXFP8 roles.

- [ ] **Step 5: Verify GREEN and commit**

```bash
pytest -q tests/unit/models/megatron/test_group_experts.py tests/unit/models/generation/test_nccl_reshard_backend.py tests/unit/weight_sync/test_nccl_reshard_utils.py
ruff check nemo_rl/models/policy/workers/megatron_policy_worker.py nemo_rl/models/generation/vllm/vllm_backend.py nemo_rl/weight_sync/nccl_reshard_utils.py
git add nemo_rl/weight_sync/nccl_reshard_utils.py nemo_rl/models/policy/workers/megatron_policy_worker.py nemo_rl/models/generation/vllm/vllm_backend.py tests/unit/models/megatron/test_group_experts.py tests/unit/models/generation/test_nccl_reshard_backend.py
git commit -s -m "refactor(refit): transfer ordered precision components"
```

---

### Task 5: Generalize Handshake And Verify Plan Agreement

**Files:**
- Modify: `nemo_rl/weight_sync/interfaces.py:35-80`
- Modify: `nemo_rl/models/policy/interfaces.py:185-210`
- Modify: `nemo_rl/models/generation/interfaces.py:330-365`
- Modify: `nemo_rl/models/policy/lm_policy.py:930-960`
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py:350-370,1856-1905`
- Modify: `nemo_rl/models/generation/vllm/vllm_backend.py:459-495`
- Modify: `nemo_rl/models/generation/vllm/vllm_generation.py:924-950,1044-1055`
- Modify: `nemo_rl/models/generation/vllm/vllm_worker.py:1093-1115,1205-1210`
- Modify: `nemo_rl/weight_sync/nccl_reshard_weight_synchronizer.py:200-226`
- Modify focused handshake tests.

**Interfaces:**
- Consumes: `RefitTransformRequest` and `plan_signature`.
- Produces: `enable_refit_transforms(requests)` and destination plan-signature acknowledgement.

- [ ] **Step 1: Write failing structured-handshake tests**

Use this request:

```python
RefitTransformRequest(
    parameter_names=("model.layers.0.mlp.down_proj.weight",),
    source_format="bf16",
    target_format="mxfp8_e4m3_e8m0",
)
```

Assert the shared helper calls `policy.enable_refit_transforms`, re-sends updated metadata, rejects non-Megatron policies clearly, and NCCL initialization rejects a destination signature mismatch.

- [ ] **Step 2: Verify RED**

```bash
pytest -q tests/unit/weight_sync/test_weight_synchronizer.py tests/unit/weight_sync/test_nccl_reshard_weight_synchronizer.py tests/unit/models/policy/test_policy_validation.py tests/unit/models/generation/test_vllm_backend.py
```

Expected: failures reference the old `list[str]`/`enable_refit_prequantize` contract.

- [ ] **Step 3: Implement structured negotiation**

Add this internal policy hook:

```python
def enable_refit_transforms(
    self, requests: list[RefitTransformRequest]
) -> Optional[dict[str, Any]]:
```

Megatron resolves and validates requests, then reuses the existing MXFP8 quantizer. vLLM unions worker names into one deterministic request. Other generation backends return `None`. Replace the NCCL synchronizer's duplicated handshake with `initialize_refit_metadata` and compare source/destination plan signatures before refit.

- [ ] **Step 4: Verify GREEN and commit**

```bash
pytest -q tests/unit/weight_sync/test_refit_transforms.py tests/unit/weight_sync/test_weight_synchronizer.py tests/unit/weight_sync/test_nccl_reshard_weight_synchronizer.py tests/unit/models/policy/test_policy_validation.py tests/unit/models/generation/test_vllm_backend.py
ruff check nemo_rl/weight_sync nemo_rl/models/policy/interfaces.py nemo_rl/models/generation/interfaces.py
git add nemo_rl/weight_sync/interfaces.py nemo_rl/weight_sync/nccl_reshard_weight_synchronizer.py nemo_rl/models/policy/interfaces.py nemo_rl/models/generation/interfaces.py nemo_rl/models/policy/lm_policy.py nemo_rl/models/policy/workers/megatron_policy_worker.py nemo_rl/models/generation/vllm/vllm_backend.py nemo_rl/models/generation/vllm/vllm_generation.py nemo_rl/models/generation/vllm/vllm_worker.py tests/unit/weight_sync/test_weight_synchronizer.py tests/unit/weight_sync/test_nccl_reshard_weight_synchronizer.py tests/unit/models/policy/test_policy_validation.py tests/unit/models/generation/test_vllm_backend.py
git commit -s -m "refactor(refit): negotiate cross-precision plans"
```

---

### Task 6: Full Verification And GCP-NRT Parity

**Files:**
- Modify: `experiments/nccl_reshard_pr3294/README.md`
- Modify: `experiments/nccl_reshard_pr3294/RESULTS.md`
- Modify: `session/20260801_233400/{session_state,handoff,timeline,files}.md`

**Interfaces:**
- Consumes: all implementation tasks and the existing experiment wrapper.
- Produces: Linux test record, 2-step correctness gate, 20-step performance result, and durable handoff.

- [ ] **Step 1: Run the focused Linux unit suite**

```bash
pytest -q tests/unit/weight_sync/test_refit_transforms.py tests/unit/weight_sync/test_nccl_reshard_utils.py tests/unit/weight_sync/test_nccl_reshard_weight_synchronizer.py tests/unit/weight_sync/test_weight_synchronizer.py tests/unit/models/megatron/test_group_experts.py tests/unit/models/generation/test_nccl_reshard_backend.py tests/unit/models/generation/test_mxfp8_prequant.py tests/unit/models/generation/test_vllm_backend.py tests/unit/models/policy/test_policy_validation.py
```

- [ ] **Step 2: Run static checks**

```bash
ruff check nemo_rl/weight_sync nemo_rl/models/policy nemo_rl/models/generation/vllm tests/unit/weight_sync tests/unit/models/megatron/test_group_experts.py tests/unit/models/generation/test_nccl_reshard_backend.py
git diff --check sna/nccl-reshard-bf16-mxfp8-stable-v1...HEAD
```

Run the repository Pyright CI target for every modified module and require zero new errors.

- [ ] **Step 3: Push and run the 2-step correctness gate**

```bash
git push -u fork sna/nccl-reshard-cross-precision
```

From a fresh GCP-NRT checkout:

```bash
CONTAINER=/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/containers/nemo-rl-nightly-refresh/nemo_rl_nightly_20260730_483099.sqsh ACTION=submit MODES=mxfp8-nccl-prequant ARMS=optimized MAX_STEPS=2 NUM_PROMPTS_PER_STEP=4 NUM_GENERATIONS_PER_PROMPT=4 TRAIN_GLOBAL_BATCH_SIZE=16 MAX_TOTAL_SEQUENCE_LENGTH=512 FORCE_ON_POLICY_RATIO=false USE_IMPORTANCE_SAMPLING_CORRECTION=true MXFP8_SHUFFLE_VERIFY=1 ./experiments/nccl_reshard_pr3294/submit_prequant_ab.sh
```

Expected: exit `0:0`, two optimizer/refit cycles, finite metrics, generation KL below `0.05`, and no plan/component failure.

- [ ] **Step 4: Run the matched 20-step development arm**

```bash
CONTAINER=/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/containers/nemo-rl-nightly-refresh/nemo_rl_nightly_20260730_483099.sqsh ACTION=submit MODES=mxfp8-nccl-prequant ARMS=optimized MAX_STEPS=20 ./experiments/nccl_reshard_pr3294/submit_prequant_ab.sh
```

Compare steps 3-20 against stable W&B run `8c2n3oj7`: refit `0.887 s`, E2E `172.10 s`, throughput `1205.04 tok/s/GPU`, reward `0.52713`, and generation KL `0.003974`.

- [ ] **Step 5: Record results and commit**

Document job IDs, W&B links, included steps, confidence intervals, source commit, container, and native-M2N availability.

```bash
git add experiments/nccl_reshard_pr3294 session/20260801_233400
git commit -s -m "docs(refit): validate generic NCCL transform path"
git push fork sna/nccl-reshard-cross-precision
```

Expected: clean worktree and reproducible stable-versus-development report.
